#include <iostream>
#include <vector>
#include <string>
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"

#define ASSERT(cond) \
    if (!(cond)) { \
        std::cerr << "Assertion failed: " << #cond << "\n" \
                  << "  File: " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

void test_parse_simple_function() {
    std::cout << "Running test_parse_simple_function..." << std::endl;
    std::string source = "fn main():\n    return 123\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    
    auto prog = parser.parseProgram();
    ASSERT(prog != nullptr);
    ASSERT(prog->decls.size() == 1);
    
    std::cout << "PASS" << std::endl;
}

void test_parse_struct() {
    std::cout << "Running test_parse_struct..." << std::endl;
    std::string source = "struct Point:\n    x: i32\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog->decls.size() == 1);
    ASSERT(dynamic_cast<cool::StructDecl*>(prog->decls[0].get()) != nullptr);
    std::cout << "PASS" << std::endl;
}

void test_parse_let() {
    std::cout << "Running test_parse_let..." << std::endl;
    std::string source = "fn main():\n    let x = 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog->decls.size() == 1);
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    ASSERT(func != nullptr);
    ASSERT(func->body.size() == 1);
    ASSERT(dynamic_cast<cool::LetStmt*>(func->body[0].get()) != nullptr);
    std::cout << "PASS" << std::endl;
}

void test_parse_call() {
    std::cout << "Running test_parse_call..." << std::endl;
    std::string source = "fn main():\n    foo(x, move y)\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    auto stmt = dynamic_cast<cool::ExprStmt*>(func->body[0].get());
    auto call = dynamic_cast<cool::CallExpr*>(stmt->expr.get());
    ASSERT(call != nullptr);
    ASSERT(call->args.size() == 2);
    ASSERT(call->args[1]->mode == cool::Argument::Mode::Move);
    std::cout << "PASS" << std::endl;
}

int main() {
    test_parse_simple_function();
    test_parse_struct();
    test_parse_let();
    test_parse_call();
    std::cout << "All parser tests passed!" << std::endl;
    return 0;
}

