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
        if (!symbolTable.resolve(var->name)) {
            // Check if it's a function? 
            // In Coolscript functions are values? 
            // If main calls foo(), foo is a symbol.
            // Our symbol table has functions.
            if (!symbolTable.resolve(var->name)) {
                 throw std::runtime_error("Undefined variable: " + var->name);
            }
        }
    }
}

} // namespace cool
