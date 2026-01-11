This project template establishes the foundation for building the **Coolscript** compiler. It includes the module manifest, the main entry point for the compiler's driver, and a build system to manage the MLIR and LLVM dependencies.

---

## Coolscript Project Template

### 1. The Module Manifest (`cool.mod`)

This file defines the compiler project itself. Since the compiler will likely be written in Coolscript (bootstrapping) or a low-level language like C++/Rust for the initial version, we define the identity here.

```python
module github.com/user/coolscript-compiler
compiler 1.0.0

require:
    std.io v1.0.0
    std.fs v1.0.0
    github.com/cool-lang/peg-parser v0.1.0
    github.com/cool-lang/mlir-gen v0.1.0

# Redirecting MLIR generator to a local dev branch
replace:
    github.com/cool-lang/mlir-gen => ./internal/mlir_gen

```

---

### 2. The Entry Point (`main.cool`)

This is the driver for the `cool` CLI. It handles argument parsing and directs the flow from the PEG parser to the LLVM backend.

```python
import std.fs
import std.io
import github.com/cool-lang/peg-parser as parser

fn main():
    # 1. Parse CLI Arguments
    let args = io.get_args()
    if len(args) < 2:
        print("Usage: cool build <file.cool>")
        return

    let command = args[1]
    let target_file = args[2]

    # 2. Read Source File
    let source = fs.File.open(target_file, "r") try (err):
        print("Error opening file: " + err.msg)
        return

    let code = source.read_all() try (err):
        return

    # 3. Parsing (PEG)
    let ast = parser.parse(move code) try (err):
        print("Syntax Error: " + err.message)
        return

    # 4. Semantic Pass (Ownership & View check)
    if not ast.validate_ownership():
        print("Ownership Violation Detected")
        return

    # 5. Build Binary
    if command == "build":
        compile_to_binary(move ast)

```

---

### 3. The Build System (`Makefile`)

To handle the native compilation and linking of the MLIR/LLVM components, a standard `Makefile` is used to orchestrate the `cool build` process.

```makefile
# Coolscript Compiler Build System
BINARY_NAME=cool
SOURCE_FILES=main.cool
BUILD_DIR=./bin

all: clean build

build:
	@echo "Fetching dependencies..."
	cool tidy
	@echo "Compiling Coolscript Compiler..."
	cool build -o $(BUILD_DIR)/$(BINARY_NAME) $(SOURCE_FILES)

clean:
	rm -rf $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)

test:
	./tests/run_suite.sh

```

---

### 4. Project Directory Structure

A well-organized Coolscript project separates the parser, the IR generation, and the backend.

```text
coolscript-compiler/
├── cool.mod            # Project manifest
├── cool.sum            # Dependency checksums
├── main.cool           # CLI Driver
├── Makefile            # Build orchestration
├── src/
│   ├── parser/         # PEG Grammar and AST definitions
│   ├── semantics/      # Ownership and Region Analysis logic
│   ├── codegen/        # MLIR/LLVM lowering
│   └── linker/         # Static binary generation
├── tests/              # .cool test cases for the compiler
└── std/                # Local copy of the standard library

```

---

## The Compilation Lifecycle

To visualize how the files in this template interact, here is the lifecycle of a `.cool` file during compilation:

1. **Frontend**: `main.cool` invokes the `peg-parser` module.
2. **Analysis**: The `semantics` package verifies that the "No-Escape Rule" and "Move" rules are strictly followed in the AST.
3. **Lowering**: The AST is converted into `cool-dialect` MLIR operations.
4. **Backend**: LLVM takes the MLIR and performs machine-specific optimizations (Inlining, Loop Unrolling).
5. **Output**: A single static binary is produced, containing no external references.

---

## Next Steps for Development

To start building the language, I recommend the following sequence:

1. **Implement the PEG Parser**: Use the grammar we defined to generate the AST.
2. **The Ownership Tracker**: Build the logic that "marks" variables as burned when a `move` occurs.
3. **The MLIR Dialect**: Define the operations for `view`, `move`, and `spawn`.

