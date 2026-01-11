#pragma once
#include "../parser/AST.h"
#include "SymbolTable.h"

namespace cool {

class SemanticAnalyzer {
public:
    SemanticAnalyzer() = default;
    
    bool analyze(const Program& program);

private:
    SymbolTable symbolTable;
    std::shared_ptr<Type> currentReturnType;
    
    void visitProgram(const Program& prog);
    void visitFunction(const FunctionDecl& func);
    void visitBlock(const std::vector<std::unique_ptr<Stmt>>& stmts);
    void visitStmt(const Stmt& stmt);
    std::shared_ptr<Type> visitExpr(const Expr& expr);
};

} // namespace cool
