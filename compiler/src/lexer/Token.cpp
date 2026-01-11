#include "Token.h"

namespace cool {

std::string tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::EndOfFile: return "EOF";
        case TokenType::Identifier: return "Identifier";
        case TokenType::StringLiteral: return "StringLiteral";
        case TokenType::IntegerLiteral: return "IntegerLiteral";
        case TokenType::FloatLiteral: return "FloatLiteral";
        case TokenType::Fn: return "fn";
        case TokenType::Struct: return "struct";
        case TokenType::Let: return "let";
        case TokenType::If: return "if";
        case TokenType::Else: return "else";
        case TokenType::Indent: return "INDENT";
        case TokenType::Dedent: return "DEDENT";
        case TokenType::NewLine: return "NEWLINE";
        default: return "Token";
    }
}

} // namespace cool
