#pragma once

/**
 * Test Framework for Cool Compiler
 * 
 * Supports two distinct test categories:
 * 
 * 1. MLIR VERIFICATION TESTS (Pattern matching on IR)
 *    Location: compiler/tests/mlir_verification/
 *    - Fast verification of IR generation
 *    - Pattern match for expected MLIR operations
 *    - ~32 tests covering all features
 * 
 * 2. EXECUTION VERIFICATION TESTS (Compilation + Runtime)
 *    Location: compiler/tests/execution_verification/
 *    - Full compilation pipeline to executable
 *    - Runtime output verification
 *    - ~10 tests for comprehensive coverage
 */

#include <functional>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

class TestRegistry {
public:
    static TestRegistry& get() {
        static TestRegistry instance;
        return instance;
    }
    
    void add(const std::string& name, std::function<void()> fn) {
        tests.push_back({name, fn});
    }
    
    int runAll() {
        int passed = 0;
        int failed = 0;
        
        for (const auto& test : tests) {
            try {
                test.fn();
                std::cout << "✓ " << test.name << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cout << "✗ " << test.name << ": " << e.what() << std::endl;
                failed++;
            } catch (...) {
                std::cout << "✗ " << test.name << ": Unknown exception" << std::endl;
                failed++;
            }
        }
        
        std::cout << "\n" << passed << " passed, " << failed << " failed" << std::endl;
        return failed == 0 ? 0 : 1;
    }
    
private:
    std::vector<TestCase> tests;
};

#define TEST(name) \
    void test_##name(); \
    namespace { \
        struct Registrar_##name { \
            Registrar_##name() { \
                TestRegistry::get().add(#name, test_##name); \
            } \
        } registrar_##name; \
    } \
    void test_##name()

#define ASSERT(cond) \
    if (!(cond)) { \
        throw std::runtime_error("Assertion failed: " #cond); \
    }

#define TEST_MAIN() \
    int main() { \
        return TestRegistry::get().runAll(); \
    }
