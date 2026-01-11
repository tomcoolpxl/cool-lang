#include <iostream>
#include <vector>
#include <string>
#include "../src/lexer/Lexer.h"

// Simple Test Framework
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "Assertion failed: " << #a << " != " << #b << "\n" \
                  << "  Expected: " << (b) << "\n" \
                  << "  Actual:   " << (a) << "\n" \
                  << "  File:     " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

void test_basic_tokens() {
    std::cout << "Running test_basic_tokens..." << std::endl;
    cool::Lexer lexer("fn main 123");
    
    cool::Token t1 = lexer.nextToken();
    ASSERT_EQ((int)t1.type, (int)cool::TokenType::Fn);
    
    cool::Token t2 = lexer.nextToken();
    ASSERT_EQ((int)t2.type, (int)cool::TokenType::Identifier);
    ASSERT_EQ(t2.text, "main");
    
    cool::Token t3 = lexer.nextToken();
    ASSERT_EQ((int)t3.type, (int)cool::TokenType::IntegerLiteral);
    ASSERT_EQ(t3.text, "123");
    
    std::cout << "PASS" << std::endl;
}

void test_indentation() {
    std::cout << "Running test_indentation..." << std::endl;
    // fn main:
    //     return
    std::string source = "fn main:\n    return\n"; 
    cool::Lexer lexer(source);
    
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Fn);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Identifier); // main
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Colon);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::NewLine);
    
    // Here we expect INDENT
    cool::Token t = lexer.nextToken();
    if (t.type != cool::TokenType::Indent) {
        std::cout << "Expected INDENT, got " << (int)t.type << " (" << t.text << ")" << std::endl;
    }
    ASSERT_EQ((int)t.type, (int)cool::TokenType::Indent);
    
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Return);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::NewLine);
    
    // End of file should trigger DEDENT (auto-close blocks)
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Dedent);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::EndOfFile);
    
    std::cout << "PASS" << std::endl;
}

int main() {
    test_basic_tokens();
    test_indentation();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

