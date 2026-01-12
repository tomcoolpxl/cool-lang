#include "TestFramework.h"
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"
#include "../src/semantics/SemanticAnalyzer.h"
#include "../src/codegen/MLIRGenerator.h"

std::string generate_ir(std::string source) {
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    if (!analyzer.analyze(*prog)) {
        return "SEMANTIC_ERROR";
    }
    
    cool::MLIRGenerator generator;
    return generator.generate(*prog);
}

TEST(test_codegen_basic) {
    std::string ir = generate_ir("fn main() -> i32:\n    return 42\n");
    
    ASSERT(ir.find("func.func @main") != std::string::npos);
    ASSERT(ir.find("arith.constant 42") != std::string::npos);
    ASSERT(ir.find("return") != std::string::npos);
}

TEST(test_codegen_move) {
    std::string source = 
        "fn consume(move x: i32):\n    return\n"
        "fn main():\n"
        "    let y = 10\n"
        "    consume(move y)\n";
    std::string ir = generate_ir(source);
    
    ASSERT(ir.find("cool.move") != std::string::npos);
    ASSERT(ir.find("call @consume") != std::string::npos);
}

TEST(test_codegen_if) {
    std::string source = 
        "fn main():\n"
        "    if 1:\n"
        "        let x = 1\n";
    std::string ir = generate_ir(source);
    
    ASSERT(ir.find("scf.if") != std::string::npos);
    ASSERT(ir.find("arith.constant 1") != std::string::npos);
}

TEST(test_codegen_struct_access) {
    std::string source = 
        "struct Point:\n    x: i32\n    y: i32\n"
        "fn main():\n"
        "    let p = 0\n" // Mocking p as 0 since we don't have struct init yet, but type is what matters
        // Wait, SemanticAnalyzer needs 'p' to be a StructType for MemberAccess to pass.
        // We can't mock 'let p = 0' because 0 is i32.
        // We need a way to get a struct. Function param is easiest.
        "fn access(p: Point):\n"
        "    let v = p.y\n";
    
    std::string ir = generate_ir(source);
    
    // Check if semantic analysis passed
    ASSERT(ir != "SEMANTIC_ERROR");
    
    // Expected: get_field p[1] (since y is 2nd field, index 1)
    // The SSA for p might be %arg0.
    ASSERT(ir.find("cool.get_field") != std::string::npos);
    ASSERT(ir.find("[1]") != std::string::npos); // index 1 for y
}

TEST(test_codegen_while) {
    std::string source = 
        "fn main():\n"
        "    while 1:\n"
        "        let x = 1\n";
    std::string ir = generate_ir(source);
    
    ASSERT(ir.find("scf.while") != std::string::npos);
    ASSERT(ir.find("scf.condition") != std::string::npos);
    ASSERT(ir.find("scf.yield") != std::string::npos);
}

TEST_MAIN()
