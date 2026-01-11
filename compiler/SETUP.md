To begin **Milestone 1, Week 1**, you need a solid foundation using **CMake** and **MLIR**. This structure sets up the directory hierarchy for the frontend (Parser), the middle-end (MLIR Dialect), and the backend (LLVM Lowering).

---

## Coolscript Project Directory Structure

```text
cool-lang/
├── CMakeLists.txt
├── README.md
├── bin/                    # Compiled binaries
├── runtime/                # C Runtime source (runtime.c, runtime.h)
├── stdlib/                 # Coolscript standard library (.cool files)
├── include/
│   └── cool/
│       ├── Dialect.h       # MLIR Dialect definitions
│       └── Ops.td          # TableGen definitions for ODS
├── lib/
│   ├── Dialect/            # Dialect implementation
│   ├── Lowering/           # MLIR to LLVM conversion passes
│   └── Parser/             # PEG Parser and Lexer logic
└── tools/
    └── coolc/
        └── main.cpp        # The compiler driver (cool build)

```

---

## The Master `CMakeLists.txt`

This file links the MLIR and LLVM libraries required to build a modern compiler.

```cmake
cmake_minimum_required(VERSION 3.18)
project(cool-lang LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${LLVM_LIBRARY_DIRS} ${MLIR_LIBRARY_DIRS})

# Include TableGen for ODS
include(MLIRSubdirectory)
include(AddMLIR)

# Add subdirectories for libraries and tools
add_subdirectory(lib/Dialect)
add_subdirectory(lib/Parser)
add_subdirectory(tools/coolc)

```

---

## MLIR Dialect Boilerplate (`include/cool/Ops.td`)

This is the entry point for your custom MLIR dialect. It defines the "Cool" namespace.

```tablegen
#ifndef COOL_OPS
#define COOL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cool_Dialect : Dialect {
    let name = "cool";
    let summary = "The Coolscript dialect.";
    let description = [{
        This dialect represents the high-level operations of the Coolscript
        language, including ownership tracking and isolate spawning.
    }];
    let cppNamespace = "::mlir::cool";
}

// Base class for Cool operations
class Cool_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cool_Dialect, mnemonic, traits>;

#endif // COOL_OPS

```

---

## Milestone 1, Week 1: First Technical Task

Your first objective is to build the **Indentation-Aware Lexer**. In a PEG parser for Coolscript, the lexer must maintain a stack of integers representing indentation depths.

### The Lexer Logic (C++ snippet for `lib/Parser/Lexer.cpp`)

```cpp
class Lexer {
    std::vector<int> indentStack = {0};
    
    Token getNextToken() {
        if (atStartOfLine) {
            int currentIndent = countLeadingSpaces();
            if (currentIndent > indentStack.back()) {
                indentStack.push_back(currentIndent);
                return Token::INDENT;
            } else if (currentIndent < indentStack.back()) {
                indentStack.pop_back();
                return Token::DEDENT;
            }
        }
        // Normal tokenization logic...
    }
};

```

---

## Summary of the Week 1 Setup

1. **Environment**: You now have a CMake project that links against MLIR/LLVM.
2. **Definition**: You have a `cool` dialect namespace ready for your custom IR.
3. **Frontend**: You have the logic required to handle Python-style blocks.

To move from the Lexer to actual IR generation, you need a set of **Operations** that represent the fundamental actions of Coolscript: allocating memory, freeing it, and transferring ownership.

The following **TableGen (ODS)** file defines the low-level memory operations that your parser will emit.

---

## Core Memory Operations (`include/cool/Ops.td`)

Append these definitions to your `Ops.td` file. These operations are designed to be "Linearity-Aware," meaning the compiler can easily track which pointer is being consumed.

```tablegen
// --- Allocation ---
def AllocOp : Cool_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Allocates memory for a Coolscript object";
    let description = [{
        Allocates a block of memory for the given type on the heap.
        Returns an 'Owned' reference to the memory.
    }];

    let arguments = (ins TypeAttr:$type);
    let results = (outs AnyType:$res);

    let assemblyFormat = "$type attr-dict `:` type($res)";
}

// --- Deallocation ---
def FreeOp : Cool_Op<"free", [MemoryEffects<[MemFree]>]> {
    let summary = "Injected deallocation of memory";
    let description = [{
        Calls the runtime's cs_free on the pointer. Usually injected
        by the Auto-Destruction pass at the end of a scope.
    }];

    let arguments = (ins AnyType:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

// --- The Move Operation ---
def MoveOp : Cool_Op<"move", [RecursiveMemoryEffects]> {
    let summary = "Explicit ownership transfer";
    let description = [{
        Marks a variable as burned in the current scope. The result
        represents the same pointer but signifies that ownership 
        has been transferred to a new location.
    }];

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);

    let assemblyFormat = "$input attr-dict `:` functional-type($input, $res)";
}

```

---

## Basic Arithmetic Operations

Coolscript needs standard math operations that work on its primitive types (`i32`, `i64`, `f64`).

```tablegen
class BinaryOp<string mnemonic> : Cool_Op<mnemonic, [Pure]> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def AddOp : BinaryOp<"add">;
def SubOp : BinaryOp<"sub">;
def MulOp : BinaryOp<"mul">;
def DivOp : BinaryOp<"div">;

```

---

## How the Parser uses these Ops

When your PEG parser encounters a line like `let x = 10 + 20`, it will use the `mlir::OpBuilder` to generate the following IR:

```mlir
%c10 = cool.constant 10 : i32
%c20 = cool.constant 20 : i32
%x = cool.add %c10, %c20 : (i32, i32) -> i32

```

When it encounters `process(move x)`, it emits:

```mlir
// Marks %x as consumed and produces %x_moved for the function call
%x_moved = cool.move %x : (i32) -> i32
cool.call @process(%x_moved) : (i32) -> ()

```

---

## Next Step: The `cool.mod` Manager

With the Dialect and Parser started, you need to manage dependencies. In Coolscript, we don't have a central "NPM" or "Cargo" registry; we use URLs directly in the `cool.mod` file.

### Proposed `cool.mod` Example:

```python
module github.com/username/my_project

require:
    github.com/cool-lang/std v0.1.0
    github.com/cool-lang/net v1.2.0

```
