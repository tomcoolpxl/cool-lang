#include "Lexer.h"
#include <cctype>

namespace cool {

Lexer::Lexer(std::string_view source) : source(source) {
    indentStack.push(0);
}

Token Lexer::nextToken() {
    if (!tokenQueue.empty()) {
        Token t = tokenQueue.front();
        tokenQueue.pop_front();
        return t;
    }
    return scanToken();
}

Token Lexer::scanToken() {
    if (atStartOfLine) {
        atStartOfLine = false;
        handleIndentation();
        // If indentation generated tokens, return the first one
        if (!tokenQueue.empty()) {
            return nextToken();
        }
    }

    skipWhitespace();

    if (position >= source.length()) {
        if (indentStack.size() > 1) {
            indentStack.pop();
            return makeToken(TokenType::Dedent, "");
        }
        return makeToken(TokenType::EndOfFile, "");
    }

    char c = peek();
    advance();

    if (std::isalpha(c)) return scanIdentifier();
    if (std::isdigit(c)) return scanNumber();

    switch (c) {
        case '(': return makeToken(TokenType::LParen, "(");
        case ')': return makeToken(TokenType::RParen, ")");
        case '{': return makeToken(TokenType::LBrace, "{");
        case '}': return makeToken(TokenType::RBrace, "}");
        case '[': return makeToken(TokenType::LBracket, "[");
        case ']': return makeToken(TokenType::RBracket, "]");
        case ',': return makeToken(TokenType::Comma, ",");
        case '.': return makeToken(TokenType::Dot, ".");
        case ':': return makeToken(TokenType::Colon, ":");
        case ';': return makeToken(TokenType::Semicolon, ";");
        case '+': return makeToken(TokenType::Plus, "+");
        case '-': 
            if (match('>')) return makeToken(TokenType::Arrow, "->");
            return makeToken(TokenType::Minus, "-");
        case '*': return makeToken(TokenType::Star, "*");
        case '/': return makeToken(TokenType::Slash, "/");
        case '=': 
            if (match('=')) return makeToken(TokenType::EqualEqual, "==");
            return makeToken(TokenType::Equal, "=");
        case '!':
            if (match('=')) return makeToken(TokenType::BangEqual, "!=");
            break;
        case '<':
            if (match('=')) return makeToken(TokenType::LessEqual, "<=");
            return makeToken(TokenType::Less, "<");
        case '>':
            if (match('=')) return makeToken(TokenType::GreaterEqual, ">=");
            return makeToken(TokenType::Greater, ">");
        case '"': return scanString();
        case '\n': 
            atStartOfLine = true;
            return makeToken(TokenType::NewLine, "\n");
    }

    return errorToken("Unexpected character");
}

void Lexer::skipWhitespace() {
    while (true) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r') {
            advance();
        } else if (c == '#') {
            while (peek() != '\n' && position < source.length()) advance();
        } else {
            break;
        }
    }
}

// Minimal implementation for now
char Lexer::current() const {
    if (position > 0) return source[position - 1];
    return '\0';
}

char Lexer::peek() const {
    if (position >= source.length()) return '\0';
    return source[position];
}

void Lexer::advance() {
    position++;
    column++;
}

bool Lexer::match(char expected) {
    if (peek() == expected) {
        advance();
        return true;
    }
    return false;
}

Token Lexer::makeToken(TokenType type, std::string text) {
    return {type, text, line, column};
}

Token Lexer::errorToken(std::string message) {
    return {TokenType::Error, message, line, column};
}

Token Lexer::scanIdentifier() {
    int start = position - 1;
    while (std::isalnum(peek()) || peek() == '_') advance();
    std::string text = source.substr(start, position - start);
    
    // Keyword check
    if (text == "fn") return makeToken(TokenType::Fn, text);
    if (text == "struct") return makeToken(TokenType::Struct, text);
    if (text == "protocol") return makeToken(TokenType::Protocol, text);
    if (text == "let") return makeToken(TokenType::Let, text);
    if (text == "if") return makeToken(TokenType::If, text);
    if (text == "else") return makeToken(TokenType::Else, text);
    if (text == "elif") return makeToken(TokenType::Elif, text);
    if (text == "while") return makeToken(TokenType::While, text);
    if (text == "for") return makeToken(TokenType::For, text);
    if (text == "match") return makeToken(TokenType::Match, text);
    if (text == "return") return makeToken(TokenType::Return, text);
    if (text == "move") return makeToken(TokenType::Move, text);
    if (text == "view") return makeToken(TokenType::View, text);
    if (text == "inout") return makeToken(TokenType::Inout, text);
    if (text == "copy") return makeToken(TokenType::Copy, text);
    if (text == "shared") return makeToken(TokenType::Shared, text);
    if (text == "unsafe") return makeToken(TokenType::Unsafe, text);
    if (text == "try") return makeToken(TokenType::Try, text);
    if (text == "const") return makeToken(TokenType::Const, text);
    if (text == "import") return makeToken(TokenType::Import, text);
    if (text == "as") return makeToken(TokenType::As, text);
    
    return makeToken(TokenType::Identifier, text);
}

Token Lexer::scanNumber() {
    int start = position - 1;
    while (std::isdigit(peek())) advance();
    return makeToken(TokenType::IntegerLiteral, source.substr(start, position - start));
}

Token Lexer::scanString() {
    int start = position - 1; // Quote
    while (peek() != '"' && position < source.length()) {
        advance();
    }
    if (position >= source.length()) return errorToken("Unterminated string");
    advance(); // Closing quote
    return makeToken(TokenType::StringLiteral, source.substr(start, position - start));
}

void Lexer::handleIndentation() {
    int indent = 0;
    while (peek() == ' ' || peek() == '\t') {
        if (peek() == '\t') indent += 4; // Simple tab handling
        else indent++;
        advance();
    }
    
    // Ignore empty lines (lines containing only whitespace/comments)
    if (peek() == '\n' || peek() == '\r' || peek() == '#') {
        // If we hit a newline/comment immediately after indentation, 
        // it's a blank line. We don't change indentation.
        // We DO set atStartOfLine back to true? 
        // No, the scanToken loop will hit \n or # and loop.
        // If scanToken hits \n, it sets atStartOfLine=true.
        // So we just return here. 
        return; 
    }
    
    int currentIndent = indentStack.top();
    
    if (indent > currentIndent) {
        indentStack.push(indent);
        tokenQueue.push_back(makeToken(TokenType::Indent, "INDENT"));
    } else if (indent < currentIndent) {
        while (indent < indentStack.top()) {
            indentStack.pop();
            tokenQueue.push_back(makeToken(TokenType::Dedent, "DEDENT"));
        }
        if (indentStack.top() != indent) {
            tokenQueue.push_back(errorToken("Inconsistent indentation"));
        }
    }
}

} // namespace cool