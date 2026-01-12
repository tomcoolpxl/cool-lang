#include "TestFramework.h"
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"
#include "../src/semantics/SemanticAnalyzer.h"
#include "../src/codegen/MLIRGenerator.h"

// Test 1: fn keyword is accepted
TEST(test_fn_keyword_lexer) {
    std::string source = "fn";
    cool::Lexer lexer(source);
    auto token = lexer.nextToken();
    ASSERT(token.type == cool::TokenType::Fn);
    ASSERT(token.text == "fn");
}

// Test 2: func keyword is NOT recognized (treated as identifier)
TEST(test_func_keyword_rejected) {
    std::string source = "func";
    cool::Lexer lexer(source);
    auto token = lexer.nextToken();
    ASSERT(token.type == cool::TokenType::Identifier);
    ASSERT(token.text == "func");
}

// Test 3: Function with no explicit return (void)
TEST(test_fn_void_return) {
    std::string source = "fn foo():\n    let x = 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog != nullptr);
    ASSERT(prog->decls.size() == 1);
    
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    ASSERT(func != nullptr);
    ASSERT(func->name == "foo");
    ASSERT(func->returnType.empty() || func->returnType == "void");
}

// Test 4: Function with explicit return type
TEST(test_fn_with_return_type) {
    std::string source = "fn foo() -> i32:\n    return 42\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog != nullptr);
    
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    ASSERT(func != nullptr);
    ASSERT(func->returnType == "i32");
}

// Test 5: Semantic analysis accepts function ending with return
TEST(test_semantic_fn_with_return) {
    std::string source = "fn foo() -> i32:\n    return 42\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(result == true);
}

// Test 6: Semantic analysis REJECTS function without return but with return type
TEST(test_semantic_fn_missing_return) {
    std::string source = "fn foo() -> i32:\n    let x = 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(result == false); // Should fail semantic analysis
}

// Test 7: Codegen produces valid MLIR for void function with no explicit return
TEST(test_codegen_void_function) {
    std::string source = "fn foo():\n    let x = 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool semantic_ok = analyzer.analyze(*prog);
    ASSERT(semantic_ok == true);
    
    cool::MLIRGenerator codegen;
    std::string ir = codegen.generate(*prog);
    
    // Should contain func.return and should be valid
    ASSERT(ir.find("func.return") != std::string::npos);
    ASSERT(ir.find("module {") != std::string::npos);
}

// Test 8: Codegen produces valid MLIR for function with explicit return
TEST(test_codegen_function_with_return) {
    std::string source = "fn foo() -> i32:\n    return 42\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool semantic_ok = analyzer.analyze(*prog);
    ASSERT(semantic_ok == true);
    
    cool::MLIRGenerator codegen;
    std::string ir = codegen.generate(*prog);
    
    // Should contain the function and return
    ASSERT(ir.find("@foo") != std::string::npos);
    ASSERT(ir.find("func.return") != std::string::npos);
}

// Test 9: Multiple functions with fn keyword
TEST(test_multiple_fn_functions) {
    std::string source = "fn add(a: i32, b: i32) -> i32:\n    return a + b\n\nfn main() -> i32:\n    return 0\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    ASSERT(prog != nullptr);
    ASSERT(prog->decls.size() == 2);
    
    auto add = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    auto main = dynamic_cast<cool::FunctionDecl*>(prog->decls[1].get());
    
    ASSERT(add != nullptr && add->name == "add");
    ASSERT(main != nullptr && main->name == "main");
}

TEST_MAIN()
