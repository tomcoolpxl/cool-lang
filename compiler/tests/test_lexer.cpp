#include "TestFramework.h"
#include "../src/lexer/Lexer.h"

TEST(test_basic_tokens) {
    cool::Lexer lexer("fn main 123");
    
    cool::Token t1 = lexer.nextToken();
    ASSERT_EQ((int)t1.type, (int)cool::TokenType::Fn);
    
    cool::Token t2 = lexer.nextToken();
    ASSERT_EQ((int)t2.type, (int)cool::TokenType::Identifier);
    ASSERT_EQ(t2.text, "main");
    
    cool::Token t3 = lexer.nextToken();
    ASSERT_EQ((int)t3.type, (int)cool::TokenType::IntegerLiteral);
    ASSERT_EQ(t3.text, "123");
}

TEST(test_indentation) {
    std::string source = "fn main:\n    return\n"; 
    cool::Lexer lexer(source);
    
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Fn);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Identifier);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Colon);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::NewLine);
    
    // INDENT
    cool::Token t = lexer.nextToken();
    if (t.type != cool::TokenType::Indent) {
        throw std::runtime_error("Expected INDENT");
    }
    
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Return);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::NewLine);
    
    // DEDENT
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Dedent);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::EndOfFile);
}

TEST(test_operators) {
    cool::Lexer lexer("+ - * / = == != < <= > >=");
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Plus);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Minus);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Star);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Slash);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Equal);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::EqualEqual);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::BangEqual);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Less);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::LessEqual);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Greater);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::GreaterEqual);
}

TEST(test_strings) {
    cool::Lexer lexer("\"hello\" \"world\"");
    
    cool::Token t1 = lexer.nextToken();
    ASSERT_EQ((int)t1.type, (int)cool::TokenType::StringLiteral);
    ASSERT_EQ(t1.text, "\"hello\"");
    
    cool::Token t2 = lexer.nextToken();
    ASSERT_EQ((int)t2.type, (int)cool::TokenType::StringLiteral);
    ASSERT_EQ(t2.text, "\"world\"");
}

TEST(test_comments) {
    cool::Lexer lexer("# comment\nfn");
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::NewLine);
    ASSERT_EQ((int)lexer.nextToken().type, (int)cool::TokenType::Fn);
}

TEST(test_token_strings) {
    ASSERT_EQ(cool::tokenTypeToString(cool::TokenType::Fn), "fn");
    ASSERT_EQ(cool::tokenTypeToString(cool::TokenType::Identifier), "Identifier");
    ASSERT_EQ(cool::tokenTypeToString(cool::TokenType::NewLine), "NEWLINE");
    ASSERT_EQ(cool::tokenTypeToString(cool::TokenType::EndOfFile), "EOF");
}

TEST_MAIN()