#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <stdexcept>

// Registry for tests
class TestRegistry {
public:
    static TestRegistry& get() {
        static TestRegistry instance;
        return instance;
    }
    void registerTest(std::string name, std::function<void()> func) {
        tests.push_back({name, func});
    }
    int runAll() {
        int passed = 0;
        int failed = 0;
        std::cout << "Running " << tests.size() << " tests..." << std::endl;
        for (const auto& test : tests) {
            std::cout << "[ RUN      ] " << test.name << std::endl;
            try {
                test.func();
                std::cout << "[       OK ] " << test.name << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cout << "[  FAILED  ] " << test.name << ": " << e.what() << std::endl;
                failed++;
            } catch (...) {
                std::cout << "[  FAILED  ] " << test.name << ": Unknown error" << std::endl;
                failed++;
            }
        }
        std::cout << "\nTest Summary: " << passed << " passed, " << failed << " failed." << std::endl;
        return (failed == 0) ? 0 : 1;
    }
private:
    struct TestEntry {
        std::string name;
        std::function<void()> func;
    };
    std::vector<TestEntry> tests;
};

// Macros for users
#define TEST(name) \
    void name(); \
    struct Register##name { \
        Register##name() { TestRegistry::get().registerTest(#name, name); } \
    } register_##name; \
    void name()

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond);

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) throw std::runtime_error(std::string("Assertion failed: ") + #a + " != " + #b);

#define TEST_MAIN() \
    int main() { \
        return TestRegistry::get().runAll(); \
    }
