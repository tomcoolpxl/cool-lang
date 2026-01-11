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
    Token next;

    void advance(); // Consumes current, moves next to current, fetches new next
    bool match(TokenType type); // If current is type, advance and return true
    bool check(TokenType type); // True if current is type
    Token consume(TokenType type, std::string message); // Expects type or throws/errors

    std::unique_ptr<FunctionDecl> parseFunction();
    std::unique_ptr<Stmt> parseStatement();
    std::unique_ptr<Expr> parseExpression();
    std::unique_ptr<Expr> parsePrimary();
    
    std::vector<std::unique_ptr<Stmt>> parseBlock();
};

} // namespace cool
