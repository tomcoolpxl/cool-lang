
# TESTING.md

## Overview
This document describes the testing setup for the `cool-lang` project, including the distinction between MLIR-based verification and execution-based testing, and provides instructions for running and writing tests.

---

## Test Types

### 1. MLIR Verification
- **Purpose:** Ensures that the compiler generates correct MLIR (Multi-Level Intermediate Representation) code from Cool source files.
- **Location:** `compiler/tests/mlir_verification/`
- **How it works:**
    - Source files are compiled to MLIR.
    - The output MLIR is compared against expected results (golden files).
    - Useful for verifying code generation, optimizations, and IR transformations.
- **How to run:**
    - Use the provided test runner or CMake targets (see below).
    - Check output against golden files for differences.

### 2. Execution Verification
- **Purpose:** Validates that compiled Cool programs execute correctly and produce expected results.
- **Location:** `compiler/tests/execution_verification/`
- **How it works:**
    - Source files are compiled and executed.
    - The runtime output is compared to expected output.
    - Useful for end-to-end testing of language features and runtime behavior.
- **How to run:**
    - Use the test runner or CMake targets.
    - Output is checked for correctness.

### 3. Unit and Integration Tests
- **Purpose:** Test individual components (lexer, parser, codegen, semantics) and their integration.
- **Location:** `compiler/tests/` (e.g., `test_lexer.cpp`, `test_parser.cpp`, etc.)
- **How to run:**
    - Use CMake or the test runner to execute all or specific tests.

---

## Running Tests

### Using CMake
1. **Configure the project:**
     ```bash
     cmake -S . -B build
     ```
2. **Build tests:**
     ```bash
     cmake --build build --target tests
     ```
3. **Run all tests:**
     ```bash
     cd build
     ctest
     ```
     Or run specific tests:
     ```bash
     ctest -R <test_name>
     ```

### Using Test Runner
- Some tests may have custom runners or scripts in the `tests/` directory.
- Refer to `docs/internals/TESTING.md` for advanced usage or troubleshooting.

---

## Writing New Tests

### MLIR Verification
- Add new Cool source files and expected MLIR output to `mlir_verification/`.
- Update golden files as needed.
- Ensure new features are covered.

### Execution Verification
- Add new source files and expected output to `execution_verification/`.
- Cover edge cases and runtime scenarios.

### Unit/Integration Tests
- Add or update C++ test files in `compiler/tests/`.
- Use the provided test framework (`TestFramework.h`).

---

## Directory Structure
- `compiler/tests/` — C++ unit/integration tests
- `compiler/tests/mlir_verification/` — MLIR golden tests
- `compiler/tests/execution_verification/` — Execution output tests
- `runtime/tests/` — Runtime-specific tests

---

## Troubleshooting
- If tests fail, check build logs and output files for details.
- For MLIR mismatches, compare generated and golden files.
- For execution failures, check runtime output and error messages.

---

## Additional Resources
- See `docs/internals/TESTING.md` for deeper technical details.
- Refer to `docs/guides/BUILDING.md` for build setup.
- For bug reporting, use `docs/internals/BUG_REPORT_TEMPLATE.md`.

---

## Contact
For questions or help, reach out via repository issues or contact maintainers listed in `README.md`.
---


