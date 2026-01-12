#pragma once
#include "../parser/AST.h"
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace cool {

class MLIRGenerator {
public:
    MLIRGenerator() = default;
    
    std::string generate(const Program& program);

private:
    std::stringstream output;
    int ssaCounter = 0;
    int indentLevel = 0;
    
    // Maps variable names to their current SSA value definition
    std::vector<std::unordered_map<std::string, std::string>> symbolStack;
    
    void emit(const std::string& line);
    std::string nextSSA();
    
    void enterScope();
    void exitScope();
    std::string getSSA(const std::string& name);
    void setSSA(const std::string& name, const std::string& ssa);

    void visitProgram(const Program& prog);
    void visitFunction(const FunctionDecl& func);
    void visitBlock(const std::vector<std::unique_ptr<Stmt>>& stmts);
    void visitStmt(const Stmt& stmt);
    std::string visitExpr(const Expr& expr);
};

} // namespace cool
