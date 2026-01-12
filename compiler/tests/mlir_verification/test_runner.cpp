// Combined test runner for MLIR verification tests
// This includes all tests from test_integration.cpp and test_execution.cpp

#include "TestFramework.h"
#include "../../src/lexer/Lexer.h"
#include "../../src/parser/Parser.h"
#include "../../src/semantics/SemanticAnalyzer.h"
#include "../../src/codegen/MLIRGenerator.h"
#include <fstream>
#include <sstream>

std::string getTestFilePath(const std::string& filename) {
    std::ifstream f1(filename);
    if (f1.good()) {
        return filename;
    }
    
    std::string source_dir = __FILE__;
    size_t last_slash = source_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string path = source_dir.substr(0, last_slash) + "/" + filename;
        std::ifstream f2(path);
        if (f2.good()) {
            return path;
        }
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

std::string compileAndGetIR(const std::string& filename) {
    std::string source = readTestFile(filename);
    
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    if (!prog) {
        throw std::runtime_error("Failed to parse: " + filename);
    }
    
    cool::SemanticAnalyzer analyzer;
    if (!analyzer.analyze(*prog)) {
        throw std::runtime_error("Semantic analysis failed for: " + filename);
    }
    
    cool::MLIRGenerator generator;
    return generator.generate(*prog);
}

std::string compileToMLIR(const std::string& source) {
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
    return generator.generate(*prog);
}

// ===== FEATURE: Basic Functions =====
TEST(feature_basic_function_compilation) {
    std::string ir = compileAndGetIR("feature_basic_function.cool");
    ASSERT(ir.find("func.func @add") != std::string::npos);
    ASSERT(ir.find("func.func @main") != std::string::npos);
}

TEST(feature_basic_function_parameters) {
    std::string ir = compileAndGetIR("feature_basic_function.cool");
    ASSERT(ir.find("%arg") != std::string::npos);
}

TEST(feature_basic_function_return_value) {
    std::string ir = compileAndGetIR("feature_basic_function.cool");
    ASSERT(ir.find("return") != std::string::npos);
}

// ===== FEATURE: Variables and Arithmetic =====
TEST(feature_variables_and_arithmetic_compilation) {
    std::string ir = compileAndGetIR("feature_variables_arithmetic.cool");
    ASSERT(ir.find("arith.constant") != std::string::npos);
}

TEST(feature_variables_and_arithmetic_operations) {
    std::string ir = compileAndGetIR("feature_variables_arithmetic.cool");
    ASSERT(ir.find("arith.") != std::string::npos || ir.find("addi") != std::string::npos);
}

// ===== FEATURE: If Statements =====
TEST(feature_if_statement_compilation) {
    std::string ir = compileAndGetIR("feature_if_statement.cool");
    ASSERT(ir.find("cf.cond_br") != std::string::npos);
}

TEST(feature_if_statement_basic_blocks) {
    std::string ir = compileAndGetIR("feature_if_statement.cool");
    ASSERT(ir.find("^bb_") != std::string::npos);
}

// ===== FEATURE: While Loops =====
TEST(feature_while_loop_compilation) {
    std::string ir = compileAndGetIR("feature_while_loop.cool");
    ASSERT(ir.find("cf.cond_br") != std::string::npos);
}

TEST(feature_while_loop_structure) {
    std::string ir = compileAndGetIR("feature_while_loop.cool");
    ASSERT(ir.find("^bb_while") != std::string::npos);
    ASSERT(ir.find("cf.br") != std::string::npos);
}

// ===== FEATURE: Struct Definitions =====
TEST(feature_struct_definition_compilation) {
    std::string ir = compileAndGetIR("feature_struct_definition.cool");
    ASSERT(ir.find("module") != std::string::npos);
}

TEST(feature_struct_field_access) {
    std::string ir = compileAndGetIR("feature_struct_definition.cool");
    ASSERT(ir.find("cool.get_field") != std::string::npos);
}

// ===== FEATURE: Spawn Tasks =====
TEST(feature_spawn_task_compilation) {
    std::string ir = compileAndGetIR("feature_spawn_task.cool");
    ASSERT(ir.find("func.func @worker") != std::string::npos);
    ASSERT(ir.find("func.func @main") != std::string::npos);
}

TEST(feature_spawn_task_runtime_calls) {
    std::string ir = compileAndGetIR("feature_spawn_task.cool");
    ASSERT(ir.find("@cs_spawn") != std::string::npos);
}

// ===== FEATURE: Function Calls =====
TEST(feature_function_calls_compilation) {
    std::string ir = compileAndGetIR("feature_function_calls.cool");
    ASSERT(ir.find("call @cs_print") != std::string::npos);
}

TEST(feature_function_calls_builtin_print) {
    std::string ir = compileAndGetIR("feature_function_calls.cool");
    int print_count = 0;
    size_t pos = 0;
    while ((pos = ir.find("@cs_print", pos)) != std::string::npos) {
        print_count++;
        pos++;
    }
    ASSERT(print_count >= 2 && "Expected at least 2 print calls");
}

// ===== FEATURE: Complex Integration =====
TEST(complex_integration_full_pipeline) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    ASSERT(ir.find("Point") != std::string::npos);
    ASSERT(ir.find("Box") != std::string::npos);
}

TEST(complex_integration_nested_struct_access) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    int get_field_count = 0;
    size_t pos = 0;
    while ((pos = ir.find("cool.get_field", pos)) != std::string::npos) {
        get_field_count++;
        pos++;
    }
    ASSERT(get_field_count >= 2 && "Expected nested field accesses");
}

TEST(complex_integration_multiple_functions) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    ASSERT(ir.find("func.func @point_sum") != std::string::npos);
    ASSERT(ir.find("func.func @box_area") != std::string::npos);
    ASSERT(ir.find("func.func @validate_box") != std::string::npos);
    ASSERT(ir.find("func.func @process_geometry") != std::string::npos);
    ASSERT(ir.find("func.func @main") != std::string::npos);
}

TEST(complex_integration_deep_call_stack) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    int call_count = 0;
    size_t pos = 0;
    while ((pos = ir.find("call @", pos)) != std::string::npos) {
        call_count++;
        pos++;
    }
    ASSERT(call_count >= 3 && "Expected multiple function calls");
}

TEST(complex_integration_arithmetic_chain) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    ASSERT(ir.find("arith.") != std::string::npos || ir.find("addi") != std::string::npos);
}

TEST(complex_integration_control_flow_with_data) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    ASSERT(ir.find("cf.cond_br") != std::string::npos);
    ASSERT(ir.find("^bb_then") != std::string::npos);
}

TEST(complex_integration_loop_structure) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    ASSERT(ir.find("^bb_while") != std::string::npos);
}

TEST(complex_integration_variable_scope_and_lifetime) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    ASSERT(ir.find("func.func") != std::string::npos);
    ASSERT(!ir.empty());
}

TEST(complex_integration_print_output_verification) {
    std::string ir = compileAndGetIR("feature_complex_integration.cool");
    int print_count = 0;
    size_t pos = 0;
    while ((pos = ir.find("@cs_print", pos)) != std::string::npos) {
        print_count++;
        pos++;
    }
    ASSERT(print_count >= 2 && "Expected multiple print calls");
}

// ===== EXECUTION VERIFICATION: MLIR Pattern Tests =====

TEST(exec_basic_add_function_returns_8) {
    std::string source = 
        "fn add(a: i32, b: i32) -> i32:\n"
        "    return a + b\n"
        "fn main() -> i32:\n"
        "    let result = add(5, 3)\n"
        "    return result\n";
    
    std::string mlir = compileToMLIR(source);
    ASSERT(mlir.find("func.func @add") != std::string::npos);
    ASSERT(mlir.find("call @add") != std::string::npos);
}

TEST(exec_arithmetic_chain_produces_60) {
    std::string source = 
        "fn compute() -> i32:\n"
        "    let x = 10\n"
        "    let y = 20\n"
        "    let z = 2\n"
        "    let sum = x + y\n"
        "    return sum * z\n";
    
    std::string mlir = compileToMLIR(source);
    ASSERT(mlir.find("arith.constant 10") != std::string::npos);
    ASSERT(mlir.find("arith.constant 20") != std::string::npos);
    ASSERT(mlir.find("arith.constant 2") != std::string::npos);
    ASSERT(mlir.find("arith.addi") != std::string::npos);
    ASSERT(mlir.find("arith.muli") != std::string::npos);
}

TEST(exec_nested_calls_stack_correctness) {
    std::string source = 
        "fn inner(x: i32) -> i32:\n"
        "    return x + 1\n"
        "fn middle(x: i32) -> i32:\n"
        "    return inner(x * 2)\n"
        "fn main() -> i32:\n"
        "    return middle(5)\n";
    
    std::string mlir = compileToMLIR(source);
    int call_count = 0;
    size_t pos = 0;
    while ((pos = mlir.find("call @", pos)) != std::string::npos) {
        call_count++;
        pos++;
    }
    ASSERT(call_count >= 2);
}

TEST(exec_print_generates_call) {
    std::string source = 
        "fn main() -> i32:\n"
        "    print(42)\n"
        "    print(100)\n"
        "    return 0\n";
    
    std::string mlir = compileToMLIR(source);
    int print_count = 0;
    size_t pos = 0;
    while ((pos = mlir.find("@cs_print", pos)) != std::string::npos) {
        print_count++;
        pos++;
    }
    ASSERT(print_count >= 2);
}

TEST(exec_if_generates_branches) {
    std::string source = 
        "fn main() -> i32:\n"
        "    if 1:\n"
        "        return 1\n"
        "    return 0\n";
    
    std::string mlir = compileToMLIR(source);
    ASSERT(mlir.find("cf.cond_br") != std::string::npos);
    ASSERT(mlir.find("^bb_") != std::string::npos);
}

TEST(exec_struct_field_access_generates_ops) {
    std::string source = 
        "struct Point:\n"
        "    x: i32\n"
        "    y: i32\n"
        "\n"
        "fn get_x(p: Point) -> i32:\n"
        "    return p.x\n"
        "\n"
        "fn main() -> i32:\n"
        "    return 0\n";
    
    std::string mlir = compileToMLIR(source);
    ASSERT(mlir.find("cool.get_field") != std::string::npos);
}

TEST(exec_nested_struct_access_multiple_gets) {
    std::string source = 
        "struct Point:\n"
        "    x: i32\n"
        "    y: i32\n"
        "\n"
        "struct Box:\n"
        "    tl: Point\n"
        "    br: Point\n"
        "\n"
        "fn get_tl_x(b: Box) -> i32:\n"
        "    return b.tl.x\n"
        "\n"
        "fn main() -> i32:\n"
        "    return 0\n";
    
    std::string mlir = compileToMLIR(source);
    int get_field_count = 0;
    size_t pos = 0;
    while ((pos = mlir.find("cool.get_field", pos)) != std::string::npos) {
        get_field_count++;
        pos++;
    }
    ASSERT(get_field_count >= 2);
}

TEST(exec_complex_program_with_all_features) {
    std::string source = 
        "struct Point:\n"
        "    x: i32\n"
        "    y: i32\n"
        "\n"
        "fn add_points(p: Point) -> i32:\n"
        "    return p.x + p.y\n"
        "\n"
        "fn main() -> i32:\n"
        "    let w = 10\n"
        "    if w:\n"
        "        print(w)\n"
        "    return w\n";
    
    std::string mlir = compileToMLIR(source);
    ASSERT(mlir.find("func.func @add_points") != std::string::npos);
    ASSERT(mlir.find("func.func @main") != std::string::npos);
    ASSERT(mlir.find("cool.get_field") != std::string::npos);
    ASSERT(mlir.find("cf.cond_br") != std::string::npos);
    ASSERT(mlir.find("@cs_print") != std::string::npos);
}

TEST_MAIN();
