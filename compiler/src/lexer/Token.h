#pragma once
#include <string>
#include <string_view>

namespace cool {

enum class TokenType {
    // End of File
    EndOfFile,
    
    // Identifiers & Literals
    Identifier,
    StringLiteral,
    IntegerLiteral,
    FloatLiteral,
    
    // Keywords
    Fn, Struct, Protocol, Let, If, Else, Elif, While, For, Match, Return,
    Move, View, Inout, Copy, Shared, Unsafe,
    Try, Const, Import, As, Class, Spawn,
    
    // Operators
    Plus, Minus, Star, Slash, 
    Equal, EqualEqual, BangEqual,
    Less, LessEqual, Greater, GreaterEqual,
    Arrow, // ->
    Dot, Comma, Colon, Semicolon, Question, // ?
    LParen, RParen, LBracket, RBracket, LBrace, RBrace,
    
    // Indentation tokens
    Indent, Dedent, NewLine,
    
    // Error
    Error
};

struct Token {
    TokenType type;
    std::string text;
    int line;
    int column;
    
    bool is(TokenType t) const { return type == t; }
};

std::string tokenTypeToString(TokenType type);

} // namespace cool
