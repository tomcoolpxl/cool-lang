#include "TestFramework.h"
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"
#include "../src/semantics/SemanticAnalyzer.h"
#include "../src/codegen/MLIRGenerator.h"
#include <fstream>
#include <sstream>

// Helper to get the correct path to test files
std::string getTestFilePath(const std::string& filename) {
    // First try current directory
    std::ifstream f1(filename);
    if (f1.good()) {
        return filename;
    }
    
    // Then try relative to source directory (__FILE__)
    std::string source_dir = __FILE__;
    size_t last_slash = source_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string path = source_dir.substr(0, last_slash) + "/" + filename;
        std::ifstream f2(path);
        if (f2.good()) {
            return path;
        }
    }
    
    return filename; // Return original if all else fails
}

// Helper to read test file
std::string readTestFile(const std::string& filename) {
    std::string path = getTestFilePath(filename);
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open test file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Helper to compile and validate a .cool file
bool compileCoolFile(const std::string& filename) {
    std::string source = readTestFile(filename);
    
    // Lexer
    cool::Lexer lexer(source);
    
    // Parser
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    if (!prog) {
        return false;
    }
    
    // Semantic Analysis
    cool::SemanticAnalyzer analyzer;
    if (!analyzer.analyze(*prog)) {
        return false;
    }
    
    // Code Generation (MLIR)
    cool::MLIRGenerator generator;
    std::string ir = generator.generate(*prog);
    
    // Basic validation: should generate non-empty MLIR
    return !ir.empty() && ir.find("module") != std::string::npos;
}

TEST(test_integration_basic_arithmetic) {
    // Test the basic arithmetic example: add function and main
    bool success = compileCoolFile("test.cool");
    ASSERT(success && "Failed to compile test.cool");
}

TEST(test_integration_spawn_concurrency) {
    // Test the spawn/concurrency example
    bool success = compileCoolFile("test_spawn.cool");
    ASSERT(success && "Failed to compile test_spawn.cool");
}

TEST(test_integration_test_cool_structure) {
    // Validate test.cool has expected structure
    std::string source = readTestFile("test.cool");
    
    // Check for main function
    ASSERT(source.find("fn main()") != std::string::npos && 
           "test.cool missing main function");
    
    // Check for add function
    ASSERT(source.find("fn add(a: i32, b: i32)") != std::string::npos && 
           "test.cool missing add function");
    
    // Check for return statements
    ASSERT(source.find("return") != std::string::npos && 
           "test.cool missing return statements");
}

TEST(test_integration_test_spawn_cool_structure) {
    // Validate test_spawn.cool has expected structure
    std::string source = readTestFile("test_spawn.cool");
    
    // Check for worker function
    ASSERT(source.find("fn worker(x: i32)") != std::string::npos && 
           "test_spawn.cool missing worker function");
    
    // Check for main function
    ASSERT(source.find("fn main()") != std::string::npos && 
           "test_spawn.cool missing main function");
    
    // Check for spawn keyword
    ASSERT(source.find("spawn") != std::string::npos && 
           "test_spawn.cool missing spawn keyword");
}

TEST(test_integration_full_pipeline_test_cool) {
    // Full pipeline test for test.cool
    std::string source = readTestFile("test.cool");
    
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog && "Failed to parse test.cool");
    
    cool::SemanticAnalyzer analyzer;
    bool semantic_ok = analyzer.analyze(*prog);
    ASSERT(semantic_ok && "Failed semantic analysis for test.cool");
    
    cool::MLIRGenerator generator;
    std::string ir = generator.generate(*prog);
    
    // Verify MLIR contains key elements
    ASSERT(ir.find("func.func @main") != std::string::npos && 
           "MLIR missing main function");
    ASSERT(ir.find("func.func @add") != std::string::npos && 
           "MLIR missing add function");
}

TEST(test_integration_full_pipeline_test_spawn_cool) {
    // Full pipeline test for test_spawn.cool
    std::string source = readTestFile("test_spawn.cool");
    
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog && "Failed to parse test_spawn.cool");
    
    cool::SemanticAnalyzer analyzer;
    bool semantic_ok = analyzer.analyze(*prog);
    ASSERT(semantic_ok && "Failed semantic analysis for test_spawn.cool");
    
    cool::MLIRGenerator generator;
    std::string ir = generator.generate(*prog);
    
    // Verify MLIR contains key elements
    ASSERT(ir.find("func.func @main") != std::string::npos && 
           "MLIR missing main function");
    ASSERT(ir.find("func.func @worker") != std::string::npos && 
           "MLIR missing worker function");
}

TEST(test_integration_no_syntax_errors) {
    // Ensure both test files parse without syntax errors
    try {
        compileCoolFile("test.cool");
        compileCoolFile("test_spawn.cool");
    } catch (const std::exception& e) {
        throw std::runtime_error("Integration test failed with error: " + std::string(e.what()));
    }
}

int main() {
    return TestRegistry::get().runAll();
}
