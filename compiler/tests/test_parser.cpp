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

int main() {
    test_parse_simple_function();
    std::cout << "All parser tests passed!" << std::endl;
    return 0;
}

