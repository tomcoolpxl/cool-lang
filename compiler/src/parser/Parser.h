#pragma once
#include "../lexer/Lexer.h"
#include "AST.h"

namespace cool {

class Parser {
public:
    Parser(Lexer& lexer);
    std::unique_ptr<Program> parseProgram();

private:
    Lexer& lexer;
    Token current;
    Token peekToken;

    void advance(); 
    bool match(TokenType type); 
    bool check(TokenType type); 
    Token consume(TokenType type, std::string message); 

    std::unique_ptr<FunctionDecl> parseFunction();
    std::unique_ptr<StructDecl> parseStruct();
    std::unique_ptr<Stmt> parseStatement();
    std::unique_ptr<Stmt> parseLetStmt();
    std::unique_ptr<Stmt> parseIfStmt();
    std::unique_ptr<Stmt> parseWhileStmt();
    std::unique_ptr<Expr> parseExpression();
    std::unique_ptr<Expr> parseLogicalOr();
    std::unique_ptr<Expr> parseLogicalAnd();
    std::unique_ptr<Expr> parseEquality();
    std::unique_ptr<Expr> parseComparison();
    std::unique_ptr<Expr> parseTerm();   // + -
    std::unique_ptr<Expr> parseFactor(); // * /
    std::unique_ptr<Expr> parsePostfix();
    std::unique_ptr<Expr> parsePrimary();
    std::unique_ptr<Expr> parseCall(std::unique_ptr<Expr> callee);
    
    std::vector<std::unique_ptr<Stmt>> parseBlock();
};

} // namespace cool
