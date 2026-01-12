#include "MLIRGenerator.h"
#include "../semantics/Type.h"
#include <iostream>

namespace cool {

std::string MLIRGenerator::generate(const Program& program) {
    output.str("");
    output << "module {\n";
    indentLevel++;
    visitProgram(program);
    indentLevel--;
    output << "}\n";
    return output.str();
}

void MLIRGenerator::emit(const std::string& line) {
    for (int i = 0; i < indentLevel * 2; ++i) output << " ";
    output << line << "\n";
}

std::string MLIRGenerator::nextSSA() {
    return "%" + std::to_string(ssaCounter++);
}

void MLIRGenerator::enterScope() {
    symbolStack.push_back({});
}

void MLIRGenerator::exitScope() {
    if (!symbolStack.empty()) symbolStack.pop_back();
}

std::string MLIRGenerator::getSSA(const std::string& name) {
    for (auto it = symbolStack.rbegin(); it != symbolStack.rend(); ++it) {
        if (it->count(name)) return it->at(name);
    }
    return "%unknown_" + name;
}

void MLIRGenerator::setSSA(const std::string& name, const std::string& ssa) {
    if (!symbolStack.empty()) {
        symbolStack.back()[name] = ssa;
    }
}

void MLIRGenerator::visitProgram(const Program& prog) {
    for (const auto& decl : prog.decls) {
        if (auto func = dynamic_cast<const FunctionDecl*>(decl.get())) {
            visitFunction(*func);
        }
    }
}

void MLIRGenerator::visitFunction(const FunctionDecl& func) {
    // Reset SSA counter for each function? 
    // Usually SSA IDs are local to the region.
    ssaCounter = 0; 
    
    std::stringstream sig;
    sig << "func.func @" << func.name << "(";
    // Params
    for (size_t i = 0; i < func.params.size(); ++i) {
        if (i > 0) sig << ", ";
        // Simple mapping for now
        std::string type = "i32"; // Default mock
        if (func.params[i].typeName.find("view") == 0) type = "!cool.view<i32>";
        sig << "%arg" << i << ": " << type;
    }
    sig << ")";
    
    // Return type
    if (!func.returnType.empty() && func.returnType != "void") {
        sig << " -> " << "i32"; // Mock
    }
    
    sig << " {";
    emit(sig.str());
    
    indentLevel++;
    enterScope();
    
    // Register params
    for (size_t i = 0; i < func.params.size(); ++i) {
        setSSA(func.params[i].name, "%arg" + std::to_string(i));
        // Bump ssaCounter so we don't collide with %argN if we use %N
        if ((int)i >= ssaCounter) ssaCounter = i + 1;
    }

    visitBlock(func.body);
    
    // Ensure return if void and missing
    // (Simplification: if last stmt wasn't return)
    if (func.returnType.empty() || func.returnType == "void") {
        emit("return");
    }
    
    exitScope();
    indentLevel--;
    emit("}");
}

void MLIRGenerator::visitBlock(const std::vector<std::unique_ptr<Stmt>>& stmts) {
    for (const auto& stmt : stmts) {
        visitStmt(*stmt);
    }
}

void MLIRGenerator::visitStmt(const Stmt& stmt) {
    if (auto let = dynamic_cast<const LetStmt*>(&stmt)) {
        std::string val = visitExpr(*let->initializer);
        // cool.alloc is conceptual; for primitives we might just alias the SSA value
        // But to support re-assignment or mutable borrowing, we typically need an alloca.
        // For this milestone, we'll map the name directly to the value SSA (immutable binding).
        setSSA(let->name, val);
        
    } else if (auto ret = dynamic_cast<const ReturnStmt*>(&stmt)) {
        if (ret->value) {
            std::string val = visitExpr(*ret->value);
            emit("return " + val + " : i32");
        } else {
            emit("return");
        }
    } else if (auto exprStmt = dynamic_cast<const ExprStmt*>(&stmt)) {
        visitExpr(*exprStmt->expr);
    } else if (auto ifStmt = dynamic_cast<const IfStmt*>(&stmt)) {
        std::string cond = visitExpr(*ifStmt->condition);
        emit("scf.if " + cond + " {");
        indentLevel++;
        visitBlock(ifStmt->thenBlock);
        // implicit yield? scf.if doesn't yield if it doesn't return values.
        indentLevel--;
        if (!ifStmt->elseBlock.empty()) {
            emit("} else {");
            indentLevel++;
            visitBlock(ifStmt->elseBlock);
            indentLevel--;
        }
        emit("}");
    } else if (auto whileStmt = dynamic_cast<const WhileStmt*>(&stmt)) {
        // Simple while loop lowering
        // scf.while (%arg...) : (types...) -> (types...) {
        //   %cond = ...
        //   scf.condition(%cond) %args...
        // } do {
        //   ^bb0(%arg...):
        //   ... body ...
        //   scf.yield %updated_args...
        // }
        
        // For Milestone 1, we treat it as having NO loop-carried state (simplification).
        emit("scf.while () : () -> () {");
        indentLevel++;
        std::string cond = visitExpr(*whileStmt->condition);
        emit("scf.condition(" + cond + ")");
        indentLevel--;
        emit("} do {");
        indentLevel++;
        visitBlock(whileStmt->body);
        emit("scf.yield");
        indentLevel--;
        emit("}");
    }
}

std::string MLIRGenerator::visitExpr(const Expr& expr) {
    if (auto lit = dynamic_cast<const LiteralExpr*>(&expr)) {
        std::string ssa = nextSSA();
        emit(ssa + " = arith.constant " + lit->value + " : i32");
        return ssa;
    } else if (auto var = dynamic_cast<const VariableExpr*>(&expr)) {
        return getSSA(var->name);
    } else if (auto mem = dynamic_cast<const MemberAccessExpr*>(&expr)) {
        std::string objSSA = visitExpr(*mem->object);
        
        // Resolve Type
        auto type = mem->object->resolvedType;
        if (auto view = std::dynamic_pointer_cast<ViewType>(type)) {
            type = view->innerType;
        }

        if (auto structType = std::dynamic_pointer_cast<StructType>(type)) {
            int index = 0;
            for(const auto& f : structType->fields) {
                if(f.name == mem->member) break;
                index++;
            }
            std::string res = nextSSA();
            emit(res + " = cool.get_field " + objSSA + "[" + std::to_string(index) + "] : !cool.struct<" + structType->name + ">"); 
            return res;
        }
        return "%undef_member";
    } else if (auto call = dynamic_cast<const CallExpr*>(&expr)) {
        std::string funcName = "unknown";
        if (auto v = dynamic_cast<const VariableExpr*>(call->callee.get())) {
            funcName = v->name;
        }
        
        std::stringstream args;
        for (size_t i = 0; i < call->args.size(); ++i) {
            if (i > 0) args << ", ";
            std::string val = visitExpr(*call->args[i]->expr);
            
            // Handle move/view annotations
            if (call->args[i]->mode == Argument::Mode::Move) {
                std::string moveSSA = nextSSA();
                emit(moveSSA + " = cool.move " + val + " : i32");
                val = moveSSA;
            } else if (call->args[i]->mode == Argument::Mode::View) {
                std::string viewSSA = nextSSA();
                emit(viewSSA + " = cool.borrow " + val + " : i32");
                val = viewSSA;
            }
            
            args << val;
        }
        std::string res = nextSSA();
        emit(res + " = call @" + funcName + "(" + args.str() + ") : (...) -> i32");
        return res;
    }
    return "%undef";
}

} // namespace cool
