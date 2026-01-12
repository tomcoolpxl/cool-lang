# Coolscript (Cool-Lang) Project Context

## Project Overview

**Coolscript** is a statically typed, natively compiled, resource-oriented programming language. It aims to combine:
*   **Python's Readability:** Significant indentation, colons, and clean syntax.
*   **Go's Concurrency:** Structured concurrency via Isolates and Owned Channels.
*   **Rust's Safety:** Deterministic performance and memory safety without a Garbage Collector (GC) or borrow checker complexity, using a "Move-by-Default" ownership model.

## key Features

*   **Move-by-Default:** Linear types ensure zero leaks and no use-after-free bugs.
*   **No-Escape Rule:** Views are strictly bound to the stack.
*   **Zero-Cost Safety:** No GC, no "Stop the World" pauses.
*   **MLIR/LLVM Backend:** Compiles to static binaries.

## Getting Started

### Prerequisites
*   **LLVM 18+**
*   **Clang** (for linking the C runtime)
*   **Make**

### Building and Running

1.  **Build the Project:**
    ```bash
    make build
    ```
    This should produce binaries in the `bin/` directory (e.g., `bin/cool`).

2.  **Add to Path:**
    ```bash
    export PATH=$PATH:$(pwd)/bin
    ```

3.  **Run the Compiler:**
    ```bash
    cool build main.cool -o my_app
    ./my_app
    ```

### Testing
To run the test suite (inferred from `compiler/BUILD.md`):
```bash
./tests/run_suite.sh
```
Or potentially via `make test` if available.

## Project Structure

*   **`compiler/`**: Core compiler implementation (C++ source, MLIR dialect code, includes).
*   **`docs/`**: Central documentation hub.
    *   **`spec/`**: Official Language Specification (Core, Grammar, StdLib, Features).
    *   **`internals/`**: Compiler and Runtime implementation details.
    *   **`plans/`**: Roadmaps, Milestones, and Orchestration plans.
    *   **`tracking/`**: Progress dashboards and Golden Tests.
*   **`runtime/`**: The runtime environment (C) handling channels and task management.
*   **`stdlib/`**: The Coolscript standard library.
*   **`tests/`**: Integration and unit tests.
*   **`tools/`**: Auxiliary tools (formatters, LSP, etc.).

## Documentation Index

*   **Language Spec:** [`docs/SPEC_OVERVIEW.md`](docs/SPEC_OVERVIEW.md)
*   **Tracking:** [`docs/tracking/TRACKING.md`](docs/tracking/TRACKING.md)

## Development Conventions

### Coding Style (`spec/STYLE.md`)
*   **Indentation:** Strictly 4 spaces. No tabs.
*   **Blocks:** Use colons (`:`) and indentation. No curly braces.
*   **Naming:**
    *   `PascalCase`: Structs, Protocols, Enums.
    *   `snake_case`: Functions, Methods, Variables.
    *   `SCREAMING_SNAKE_CASE`: Constants.
*   **Formatting:** The project uses `cool fmt` as the single source of truth for formatting.

### Documentation
*   **Docstrings:** Triple-quoted strings `"""` immediately following definitions.
*   **Comments:** Use `#` for single-line comments.

## Current Status (Milestone 1)
*   **Focus:** Core Engine Implementation (Week 3).
*   **Completed:** Lexer, Parser (Full), AST, Semantic Analysis (Linear Types & No-Escape Rule), Textual MLIR Codegen.
*   **Active Tasks:** MLIR/LLVM Library Integration and C Runtime.
