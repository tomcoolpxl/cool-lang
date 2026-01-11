#include "Parser.h"
#include <iostream>

namespace cool {

Parser::Parser(Lexer& lexer) : lexer(lexer) {
    current = lexer.nextToken(); // Prime the pump? 
    // Actually, if we use current/next pattern:
    // current is the one we are processing.
    // next is lookahead.
    // My Lexer::nextToken() advances.
    // So init:
    // current = lexer.nextToken();
}

void Parser::advance() {
    current = lexer.nextToken();
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::check(TokenType type) {
    return current.type == type;
}

Token Parser::consume(TokenType type, std::string message) {
    if (check(type)) {
        Token t = current;
        advance();
        return t;
    }
    std::cerr << "Parser Error: " << message << " at line " << current.line << ", got " << tokenTypeToString(current.type) << std::endl;
    exit(1); // Simple panic for now
}

std::unique_ptr<Program> Parser::parseProgram() {
    auto prog = std::make_unique<Program>();
    while (!check(TokenType::EndOfFile)) {
        if (check(TokenType::NewLine)) {
            advance(); // Skip empty lines at top level
            continue;
        }
        
        if (check(TokenType::Fn)) {
            prog->decls.push_back(parseFunction());
        } else {
            std::cerr << "Unexpected token at top level: " << tokenTypeToString(current.type) << std::endl;
            advance();
        }
    }
    return prog;
}

std::unique_ptr<FunctionDecl> Parser::parseFunction() {
    consume(TokenType::Fn, "Expected 'fn'");
    Token name = consume(TokenType::Identifier, "Expected function name");
    consume(TokenType::LParen, "Expected '('");
    // TODO: Params
    consume(TokenType::RParen, "Expected ')'");
    
    // Return type optional -> skipping for now
    
    consume(TokenType::Colon, "Expected ':'");
    consume(TokenType::NewLine, "Expected newline after function header");
    
    auto func = std::make_unique<FunctionDecl>(name.text);
    func->body = parseBlock();
    return func;
}

std::vector<std::unique_ptr<Stmt>> Parser::parseBlock() {
    consume(TokenType::Indent, "Expected block indentation");
    std::vector<std::unique_ptr<Stmt>> stmts;
    
    while (!check(TokenType::Dedent) && !check(TokenType::EndOfFile)) {
        if (check(TokenType::NewLine)) {
            advance();
            continue;
        }
        stmts.push_back(parseStatement());
    }
    
    consume(TokenType::Dedent, "Expected block dedentation");
    return stmts;
}

std::unique_ptr<Stmt> Parser::parseStatement() {
    if (match(TokenType::Return)) {
        std::unique_ptr<Expr> value = nullptr;
        if (!check(TokenType::NewLine)) {
            value = parseExpression();
        }
        consume(TokenType::NewLine, "Expected newline after return");
        return std::make_unique<ReturnStmt>(std::move(value));
    }
    
    // Expression statement
    auto expr = parseExpression();
    consume(TokenType::NewLine, "Expected newline after expression");
    return std::make_unique<ExprStmt>(std::move(expr));
}

std::unique_ptr<Expr> Parser::parseExpression() {
    return parsePrimary(); // Only primaries for now
}

std::unique_ptr<Expr> Parser::parsePrimary() {
    if (check(TokenType::IntegerLiteral)) {
        Token t = current;
        advance();
        return std::make_unique<LiteralExpr>(t.text);
    }
    if (check(TokenType::Identifier)) {
        Token t = current;
        advance();
        return std::make_unique<VariableExpr>(t.text);
    }
    std::cerr << "Unexpected token in expression: " << tokenTypeToString(current.type) << std::endl;
    exit(1);
}

} // namespace cool
