#include "MLIRGenerator.h"
#include "../semantics/Type.h"
#include <iostream>

namespace cool {

std::string MLIRGenerator::generate(const Program& program) {
    output.str("");
    wrappers.str("");
    output << "module {\n";
    indentLevel++;
    
    // Declare Runtime Functions
    emit("func.func private @cs_alloc(i64) -> !llvm.ptr");
    emit("func.func private @cs_free(!llvm.ptr) -> ()");
    emit("func.func private @cs_spawn((!llvm.ptr)->(), !llvm.ptr) -> ()");
    emit("func.func private @cs_print_int(i32) -> ()");
    emit("func.func private @cs_sleep(i32) -> ()");
    emit("func.func private @cs_chan_create(i64) -> !llvm.ptr");
    emit("func.func private @cs_chan_send(!llvm.ptr, !llvm.ptr) -> ()");
    emit("func.func private @cs_chan_receive(!llvm.ptr) -> !llvm.ptr");
    
    visitProgram(program);
    
    // Append wrappers
    output << wrappers.str();
    
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
    
    // Ensure a proper terminator (func.return or similar)
    // Check if last statement is already a return
    bool lastStmtIsReturn = false;
    if (!func.body.empty()) {
        lastStmtIsReturn = dynamic_cast<const ReturnStmt*>(func.body.back().get()) != nullptr;
    }
    
    // If the last statement is not a return, emit one
    // This handles both void functions and functions that should have returned
    // (semantic analyzer should catch unintended missing returns)
    if (!lastStmtIsReturn) {
        // For void or no-return functions, emit plain func.return
        if (func.returnType.empty() || func.returnType == "void") {
            emit("func.return");
        } else {
            // For functions with return type, emit a default return
            // This shouldn't happen if semantic analyzer is doing its job,
            // but we do it defensively to ensure valid MLIR
            emit("func.return %0 : i32"); // Default mock
        }
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
            emit("func.return " + val + " : i32");
        } else {
            emit("func.return");
        }
    } else if (auto exprStmt = dynamic_cast<const ExprStmt*>(&stmt)) {
        visitExpr(*exprStmt->expr);
    } else if (auto ifStmt = dynamic_cast<const IfStmt*>(&stmt)) {
        std::string thenLabel = "^bb_then_" + std::to_string(ssaCounter++);
        std::string elseLabel = "^bb_else_" + std::to_string(ssaCounter++);
        std::string contLabel = "^bb_cont_" + std::to_string(ssaCounter++);
        
        std::string cond = visitExpr(*ifStmt->condition);
        
        if (!ifStmt->elseBlock.empty()) {
            emit("cf.cond_br " + cond + ", " + thenLabel + ", " + elseLabel);
        } else {
            emit("cf.cond_br " + cond + ", " + thenLabel + ", " + contLabel);
        }
        
        emit(thenLabel + ":");
        indentLevel++;
        visitBlock(ifStmt->thenBlock);
        if (ifStmt->thenBlock.empty() || !dynamic_cast<const ReturnStmt*>(ifStmt->thenBlock.back().get())) {
            emit("cf.br " + contLabel);
        }
        indentLevel--;
        
        if (!ifStmt->elseBlock.empty()) {
            emit(elseLabel + ":");
            indentLevel++;
            visitBlock(ifStmt->elseBlock);
            if (ifStmt->elseBlock.empty() || !dynamic_cast<const ReturnStmt*>(ifStmt->elseBlock.back().get())) {
                emit("cf.br " + contLabel);
            }
            indentLevel--;
        } else {
            // elseLabel is implicit contLabel? No, defined above.
            // If no else block, we branched to contLabel or elseLabel? 
            // Logic above: if else empty, branch to contLabel.
            // So elseLabel is unused.
        }
        
        emit(contLabel + ":");
        
    } else if (auto whileStmt = dynamic_cast<const WhileStmt*>(&stmt)) {
        std::string condLabel = "^bb_while_cond_" + std::to_string(ssaCounter++);
        std::string bodyLabel = "^bb_while_body_" + std::to_string(ssaCounter++);
        std::string contLabel = "^bb_while_cont_" + std::to_string(ssaCounter++);
        
        emit("cf.br " + condLabel);
        
        emit(condLabel + ":");
        indentLevel++;
        std::string cond = visitExpr(*whileStmt->condition);
        emit("cf.cond_br " + cond + ", " + bodyLabel + ", " + contLabel);
        indentLevel--;
        
        emit(bodyLabel + ":");
        indentLevel++;
        visitBlock(whileStmt->body);
        emit("cf.br " + condLabel);
        indentLevel--;
        
        emit(contLabel + ":");
        
    } else if (auto spawnStmt = dynamic_cast<const SpawnStmt*>(&stmt)) {
        // 1. Resolve Callee
        auto call = spawnStmt->call.get();
        std::string funcName = "unknown";
        if (auto v = dynamic_cast<const VariableExpr*>(call->callee.get())) {
            funcName = v->name;
        }
        
        // 2. Prepare Wrapper Name
        std::string wrapperName = "spawn_wrapper_" + std::to_string(wrapperCounter++);
        
        // 3. Generate Wrapper (Assume single i32 arg for M1)
        std::stringstream w;
        w << "  func.func @" << wrapperName << "(%arg: !llvm.ptr) {\n";
        w << "    %val = llvm.load %arg : !llvm.ptr -> i32\n"; // Simplified load for M1
        w << "    func.call @" << funcName << "(%val) : (i32) -> i32\n"; // Assuming returns i32 for now
        w << "    func.call @cs_free(%arg) : (!llvm.ptr) -> ()\n";
        w << "    func.return\n";
        w << "  }\n";
        wrappers << w.str();
        
        // 4. Emit Call Site
        
        // Evaluate Argument
        std::string valSSA = visitExpr(*call->args[0]->expr); // Assume 1 arg
        
        std::string sizeSSA = nextSSA();
        emit(sizeSSA + " = arith.constant 4 : i64"); // 4 bytes for i32
        
        std::string memSSA = nextSSA();
        emit(memSSA + " = func.call @cs_alloc(" + sizeSSA + ") : (i64) -> !llvm.ptr");
        
        emit("llvm.store " + valSSA + ", " + memSSA + " : i32, !llvm.ptr");
        
        std::string funcSSA = nextSSA();
        emit(funcSSA + " = func.constant @" + wrapperName + " : (!llvm.ptr) -> ()");
        
        emit("func.call @cs_spawn(" + funcSSA + ", " + memSSA + ") : ((!llvm.ptr)->(), !llvm.ptr) -> ()");
        
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
    } else if (auto bin = dynamic_cast<const BinaryExpr*>(&expr)) {
        std::string lhs = visitExpr(*bin->left);
        std::string rhs = visitExpr(*bin->right);
        std::string res = nextSSA();
        
        std::string opCode = "arith.addi";
        bool isCmp = false;
        
        if (bin->op == "+") opCode = "arith.addi";
        else if (bin->op == "-") opCode = "arith.subi";
        else if (bin->op == "*") opCode = "arith.muli";
        else if (bin->op == "/") opCode = "arith.divsi";
        else if (bin->op == "==") { opCode = "arith.cmpi eq,"; isCmp = true; }
        else if (bin->op == "!=") { opCode = "arith.cmpi ne,"; isCmp = true; }
        else if (bin->op == "<")  { opCode = "arith.cmpi slt,"; isCmp = true; }
        else if (bin->op == "<=") { opCode = "arith.cmpi sle,"; isCmp = true; }
        else if (bin->op == ">")  { opCode = "arith.cmpi sgt,"; isCmp = true; }
        else if (bin->op == ">=") { opCode = "arith.cmpi sge,"; isCmp = true; }
        
        if (isCmp) {
            emit(res + " = " + opCode + " " + lhs + ", " + rhs + " : i32");
        } else {
            emit(res + " = " + opCode + " " + lhs + ", " + rhs + " : i32");
        }
        return res;
    } else if (auto call = dynamic_cast<const CallExpr*>(&expr)) {
        // 1. Channel Constructor: Channel[T](capacity)
        if (auto idx = dynamic_cast<const IndexExpr*>(call->callee.get())) {
            // Assume it's Channel constructor if Semantics passed
            // Verify? Semantics checked it.
            // Argument is capacity.
            std::string capSSA = visitExpr(*call->args[0]->expr);
            // Convert i32 to i64 for size_t
            std::string cap64 = nextSSA();
            emit(cap64 + " = arith.extsi " + capSSA + " : i32 to i64");
            
            std::string res = nextSSA();
            emit(res + " = func.call @cs_chan_create(" + cap64 + ") : (i64) -> !llvm.ptr");
            return res;
        }

        // 2. Method Calls: ch.send(val), ch.receive()
        if (auto mem = dynamic_cast<const MemberAccessExpr*>(call->callee.get())) {
            if (auto type = call->callee->resolvedType) { // Semantic analyzer sets this? No, Semantics sets it on CallExpr, not MemberAccess usually.
                // Wait, SemanticAnalyzer visits MemberAccess too.
                // But resolvedType is on the Expr node.
                // Let's check call->resolvedType?
                // Semantics sets resolvedType on the CallExpr to Void (for send) or T (for receive).
                // We need to know if it's a Channel.
                // Check mem->object->resolvedType.
            }
            
            // Re-resolve or trust semantics?
            // We can trust semantics pass has verified it's a Channel if member is send/receive.
            // But we need the object SSA.
            std::string objSSA = visitExpr(*mem->object);
            
            if (mem->member == "send") {
                std::string valSSA = visitExpr(*call->args[0]->expr);
                // Box it (i32 -> ptr)
                std::string sizeSSA = nextSSA();
                emit(sizeSSA + " = arith.constant 4 : i64");
                std::string boxSSA = nextSSA();
                emit(boxSSA + " = func.call @cs_alloc(" + sizeSSA + ") : (i64) -> !llvm.ptr");
                emit("llvm.store " + valSSA + ", " + boxSSA + " : i32, !llvm.ptr");
                
                emit("func.call @cs_chan_send(" + objSSA + ", " + boxSSA + ") : (!llvm.ptr, !llvm.ptr) -> ()");
                return nextSSA(); // Void
            } else if (mem->member == "receive") {
                std::string boxSSA = nextSSA();
                emit(boxSSA + " = func.call @cs_chan_receive(" + objSSA + ") : (!llvm.ptr) -> !llvm.ptr");
                
                // Unbox
                std::string valSSA = nextSSA();
                emit(valSSA + " = llvm.load " + boxSSA + " : !llvm.ptr -> i32");
                emit("func.call @cs_free(" + boxSSA + ") : (!llvm.ptr) -> ()");
                return valSSA;
            }
        }

        std::string funcName = "unknown";
        if (auto v = dynamic_cast<const VariableExpr*>(call->callee.get())) {
            funcName = v->name;
        }
        
        if (funcName == "print") {
             // Handle builtin print
             if (call->args.size() != 1) {
                 // Should have been caught by semantics but we are weak there
             }
             std::string val = visitExpr(*call->args[0]->expr);
             emit("func.call @cs_print_int(" + val + ") : (i32) -> ()");
             return nextSSA(); // Print returns void/undef?
        }
        
        if (funcName == "sleep") {
             std::string val = visitExpr(*call->args[0]->expr);
             emit("func.call @cs_sleep(" + val + ") : (i32) -> ()");
             return nextSSA();
        }
        
        std::stringstream args;
        for (size_t i = 0; i < call->args.size(); ++i) {
            if (i > 0) args << ", ";
            std::string val = visitExpr(*call->args[i]->expr);
            
            // Handle move/view annotations
            if (call->args[i]->mode == Argument::Mode::Move) {
                // For now, just alias.
                // std::string moveSSA = nextSSA();
                // emit(moveSSA + " = cool.move " + val + " : i32");
                // val = moveSSA;
            } else if (call->args[i]->mode == Argument::Mode::View) {
                // For now, just alias.
                // std::string viewSSA = nextSSA();
                // emit(viewSSA + " = cool.borrow " + val + " : i32");
                // val = viewSSA;
            }
            
            args << val;
        }
        std::string res = nextSSA();
        std::stringstream argTypes;
        argTypes << "(";
        for (size_t i = 0; i < call->args.size(); ++i) {
            if (i > 0) argTypes << ", ";
            argTypes << "i32"; // hardcoded for Milestone 1 basic test
        }
        argTypes << ")";
        
        emit(res + " = func.call @" + funcName + "(" + args.str() + ") : " + argTypes.str() + " -> i32");
        return res;
    }
    return "%undef";
}

} // namespace cool
