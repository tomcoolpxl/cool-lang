#include "TestFramework.h"
#include "../../src/lexer/Lexer.h"
#include "../../src/parser/Parser.h"
#include "../../src/semantics/SemanticAnalyzer.h"
#include "../../src/codegen/MLIRGenerator.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>

std::string getTestFilePath(const std::string& filename) {
    std::ifstream f1(filename);
    if (f1.good()) return filename;
    
    std::string source_dir = __FILE__;
    size_t last_slash = source_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string path = source_dir.substr(0, last_slash) + "/" + filename;
        std::ifstream f2(path);
        if (f2.good()) return path;
    }
    return filename;
}

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

// Compile Cool source and execute, capturing stdout
std::string compileAndExecute(const std::string& source, const std::string& testName) {
    // Write source to temp file
    std::string sourceFile = "/tmp/exec_test_" + testName + ".cool";
    std::ofstream src(sourceFile);
    src << source;
    src.close();
    
    // Compile to MLIR
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    if (!prog) {
        throw std::runtime_error("Failed to parse");
    }
    
    cool::SemanticAnalyzer analyzer;
    if (!analyzer.analyze(*prog)) {
        throw std::runtime_error("Semantic analysis failed");
    }
    
    cool::MLIRGenerator generator;
    std::string mlir = generator.generate(*prog);
    
    // Write MLIR to file
    std::string mlirFile = "/tmp/exec_test_" + testName + ".mlir";
    std::ofstream mlirOut(mlirFile);
    mlirOut << mlir;
    mlirOut.close();
    
    // Convert arith/cf dialects to LLVM dialect
    std::string mlirLoweredFile = "/tmp/exec_test_" + testName + "_lowered.mlir";
    std::string lowerCmd = "/opt/llvm-mlir/bin/mlir-opt " + mlirFile + 
                          " --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm" +
                          " -o " + mlirLoweredFile + " 2>&1";
    if (system(lowerCmd.c_str()) != 0) {
        throw std::runtime_error("MLIR lowering failed");
    }
    
    // Try to convert MLIR to LLVM (if tools available)
    std::string llvmFile = "/tmp/exec_test_" + testName + ".ll";
    std::string translateCmd = "/opt/llvm-mlir/bin/mlir-translate --mlir-to-llvmir " + mlirLoweredFile + " -o " + llvmFile + " 2>&1";
    int translateResult = system(translateCmd.c_str());
    
    if (translateResult != 0) {
        throw std::runtime_error("MLIR translation failed");
    }
    
    // Compile LLVM to object
    std::string objFile = "/tmp/exec_test_" + testName + ".o";
    std::string compileCmd = "/opt/llvm-mlir/bin/llc -filetype=obj " + llvmFile + " -o " + objFile + " 2>&1";
    if (system(compileCmd.c_str()) != 0) {
        throw std::runtime_error("LLVM compilation failed");
    }
    
    // Link to executable
    std::string exeFile = "/tmp/exec_test_" + testName;
    std::string buildDir = std::string(BUILD_DIR);
    std::string runtimeLib = buildDir + "/runtime/libcool_runtime.a";
    std::string linkCmd = "gcc " + objFile + " " + runtimeLib + " -o " + exeFile + " -lpthread 2>&1";
    if (system(linkCmd.c_str()) != 0) {
        throw std::runtime_error("Linking failed");
    }
    
    // Execute and capture output
    std::string execCmd = exeFile + " 2>&1";
    FILE* pipe = popen(execCmd.c_str(), "r");
    if (!pipe) {
        return "EXEC_FAILED";
    }
    
    std::string output;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    pclose(pipe);
    
    return output;
}

// ===== EXECUTION: Basic Arithmetic =====

TEST(exec_return_constant_value) {
    std::string source = readTestFile("exec_test_const42.cool");
    std::string result = compileAndExecute(source, "const42");
    ASSERT(result.find("42") != std::string::npos);
}

TEST(exec_simple_addition) {
    std::string source = readTestFile("exec_test_add53.cool");
    std::string result = compileAndExecute(source, "add53");
    ASSERT(result.find("8") != std::string::npos);
}

TEST(exec_arithmetic_chain) {
    std::string source = readTestFile("exec_test_chain.cool");
    std::string result = compileAndExecute(source, "chain");
    ASSERT(result.find("60") != std::string::npos);
}

// ===== EXECUTION: Function Calls =====

TEST(exec_nested_function_calls) {
    std::string source = readTestFile("exec_test_nested.cool");
    std::string result = compileAndExecute(source, "nested");
    ASSERT(result.find("11") != std::string::npos);
}

TEST(exec_multiple_function_calls) {
    std::string source = readTestFile("exec_test_multifunc.cool");
    std::string result = compileAndExecute(source, "multifunc");
    ASSERT(result.find("19") != std::string::npos);
}

// ===== EXECUTION: Control Flow =====

TEST(exec_if_true_branch) {
    std::string source = readTestFile("exec_test_iftrue.cool");
    std::string result = compileAndExecute(source, "iftrue");
    ASSERT(result.find("42") != std::string::npos);
}

TEST(exec_if_false_branch) {
    std::string source = readTestFile("exec_test_iffalse.cool");
    std::string result = compileAndExecute(source, "iffalse");
    ASSERT(result.find("99") != std::string::npos);
}

TEST(exec_while_loop_simple) {
    std::string source = readTestFile("exec_test_whileloop.cool");
    std::string result = compileAndExecute(source, "whileloop");
    ASSERT(result.find("99") != std::string::npos);
}

// ===== EXECUTION: Complex Programs =====

TEST(exec_struct_operations) {
    std::string source = readTestFile("exec_test_struct.cool");
    std::string result = compileAndExecute(source, "struct");
    ASSERT(result.find("0") != std::string::npos);
}

TEST(exec_combined_all_features) {
    std::string source = readTestFile("exec_test_combined.cool");
    std::string result = compileAndExecute(source, "combined");
    ASSERT(result.find("30") != std::string::npos);
}

TEST(exec_loop_with_arithmetic) {
    std::string source = readTestFile("exec_test_loopmath.cool");
    std::string result = compileAndExecute(source, "loopmath");
    ASSERT(result.find("16") != std::string::npos);
}

TEST_MAIN();
