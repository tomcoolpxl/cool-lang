#include <iostream>
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"
#include "../src/semantics/SemanticAnalyzer.h"

#define ASSERT(cond) \
    if (!(cond)) { \
        std::cerr << "Assertion failed: " << #cond << "\n" \
                  << "  File: " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

void test_semantic_analysis_basic() {
    std::cout << "Running test_semantic_analysis_basic..." << std::endl;
    std::string source = "fn main():\n    return 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(result);
    std::cout << "PASS" << std::endl;
}

void test_semantic_analysis_undefined_var() {
    std::cout << "Running test_semantic_analysis_undefined_var..." << std::endl;
    std::string source = "fn main():\n    return x\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result); // Should fail
    std::cout << "PASS" << std::endl;
}

int main() {
    test_semantic_analysis_basic();
    test_semantic_analysis_undefined_var();
    std::cout << "All semantic tests passed!" << std::endl;
    return 0;
}
