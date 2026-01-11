#include "TestFramework.h"
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"
#include "../src/semantics/SemanticAnalyzer.h"

TEST(test_semantic_analysis_basic) {
    std::string source = "fn main():\n    return 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(result);
}

TEST(test_semantic_analysis_undefined_var) {
    std::string source = "fn main():\n    return x\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result);
}

TEST(test_semantic_analysis_move_error) {
    std::string source = 
        "fn foo(move x: i32):\n    return 0\n"
        "fn main():\n    let y = 10\n    foo(move y)\n    return y\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result);
}

TEST(test_semantic_analysis_double_move) {
    std::string source = 
        "fn foo(move x: i32):\n    return 0\n"
        "fn main():\n    let y = 10\n    foo(move y)\n    foo(move y)\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result);
}

TEST_MAIN()