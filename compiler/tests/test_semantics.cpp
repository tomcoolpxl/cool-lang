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
    // fn main():
    //     return x  <-- undefined
    std::string source = "fn main():\n    return x\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result); // Should fail
    std::cout << "PASS" << std::endl;
}

void test_semantic_analysis_move_error() {
    std::cout << "Running test_semantic_analysis_move_error..." << std::endl;
    // fn foo(move x: i32): return 0
    // fn main():
    //     let y = 10
    //     foo(move y)
    //     return y  <-- ERROR
    std::string source = 
        "fn foo(move x: i32):\n    return 0\n"
        "fn main():\n    let y = 10\n    foo(move y)\n    return y\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result); // Should fail due to use-after-move
    std::cout << "PASS" << std::endl;
}

void test_semantic_analysis_double_move() {
    std::cout << "Running test_semantic_analysis_double_move..." << std::endl;
    // fn main():
    //     let y = 10
    //     foo(move y)
    //     foo(move y) <-- ERROR
    std::string source = 
        "fn foo(move x: i32):\n    return 0\n"
        "fn main():\n    let y = 10\n    foo(move y)\n    foo(move y)\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result); // Should fail due to double move
    std::cout << "PASS" << std::endl;
}

int main() {
    test_semantic_analysis_basic();
    test_semantic_analysis_undefined_var();
    test_semantic_analysis_move_error();
    test_semantic_analysis_double_move();
    std::cout << "All semantic tests passed!" << std::endl;
    return 0;
}
