#include "SemanticAnalyzer.h"
#include <iostream>
#include <stdexcept>

namespace cool {

bool SemanticAnalyzer::analyze(const Program& program) {
    try {
        visitProgram(program);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Semantic Error: " << e.what() << std::endl;
        return false;
    }
}

void SemanticAnalyzer::visitProgram(const Program& prog) {
    for (const auto& decl : prog.decls) {
        if (auto func = dynamic_cast<const FunctionDecl*>(decl.get())) {
            symbolTable.define(func->name, TypeRegistry::Void());
        }
    }
    
    for (const auto& decl : prog.decls) {
        if (auto func = dynamic_cast<const FunctionDecl*>(decl.get())) {
            visitFunction(*func);
        }
    }
}

void SemanticAnalyzer::visitFunction(const FunctionDecl& func) {
    symbolTable.enterScope();
    for (const auto& param : func.params) {
        // TODO: Resolve real type from param.typeName. Using Int32 placeholder.
        auto type = TypeRegistry::Int32(); 
        symbolTable.define(param.name, type);
    }
    visitBlock(func.body);
    symbolTable.exitScope();
}

void SemanticAnalyzer::visitBlock(const std::vector<std::unique_ptr<Stmt>>& stmts) {
    for (const auto& stmt : stmts) {
        visitStmt(*stmt);
    }
}

void SemanticAnalyzer::visitStmt(const Stmt& stmt) {
    if (auto ret = dynamic_cast<const ReturnStmt*>(&stmt)) {
        if (ret->value) visitExpr(*ret->value);
    } else if (auto exprStmt = dynamic_cast<const ExprStmt*>(&stmt)) {
        visitExpr(*exprStmt->expr);
    }
}

void SemanticAnalyzer::visitExpr(const Expr& expr) {
    if (auto var = dynamic_cast<const VariableExpr*>(&expr)) {
        Symbol* sym = symbolTable.resolve(var->name);
        if (!sym) {
             throw std::runtime_error("Undefined variable: " + var->name);
        }
        
        if (sym->state == OwnershipState::Burned) {
            throw std::runtime_error("Use of moved value: " + var->name);
        }
    }
}

} // namespace cool
