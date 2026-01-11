#pragma once
#include <string>
#include <vector>
#include <stack>
#include <deque>
#include "Token.h"

namespace cool {

class Lexer {
public:
    Lexer(std::string_view source);
    
    Token nextToken();

private:
    std::string source; // Owns the source or view? string_view is safer if source outlives lexer. 
    // Let's assume we copy source for now to be safe with main.cpp
    
    int position = 0;
    int line = 1;
    int column = 1;
    
    std::stack<int> indentStack;
    std::deque<Token> tokenQueue; // For buffering DEDENTs
    
    bool atStartOfLine = true;

    char current() const;
    char peek() const;
    void advance();
    bool match(char expected);
    
    void skipWhitespace();
    void handleIndentation();
    
    Token scanToken();
    Token scanIdentifier();
    Token scanNumber();
    Token scanString();
    
    Token makeToken(TokenType type, std::string text);
    Token errorToken(std::string message);
};

} // namespace cool
