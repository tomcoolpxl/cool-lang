To complete the **Coolscript** architecture, the compiler driver must orchestrate the transformation from high-level source code to a statically linked binary. This process involves invoking the **MLIR-to-LLVM** pipeline and then calling the system linker to merge the user code with the **C Runtime** (`runtime.o`).

---

## The `cool build` Orchestration Logic

When the user runs `cool build main.cool`, the compiler driver executes the following internal sequence:

### 1. The Compilation Phase (Object Generation)

The driver lowers the Coolscript source through the PEG parser and the Ownership Pass into LLVM IR. It then invokes the LLVM backend (often via `llc`) to produce a platform-specific object file.

```bash
# Internal step: Generate user object code
cool-compiler --emit-obj main.cool -o main.o

```

### 2. The Linking Phase (Binary Creation)

The driver must combine the generated `main.o` with the pre-compiled Coolscript runtime. It uses the system linker (e.g., `ld` on Linux or `link.exe` on Windows).

```bash
# Internal step: Final Link
ld main.o runtime.o -lpthread -lc -o main

```

---

## Final Compiler Driver Specification (`driver.cool`)

This is the pseudo-implementation of the `build` command within the compiler's own source code.

```python
fn compile_to_binary(move ast: AST, output_name: str):
    # 1. Transform AST to MLIR
    let mlir_module = mlir.generate(move ast)
    
    # 2. Run Ownership and Lifetime Injection Passes
    mlir_module.run_pass("cool-linear-types")
    mlir_module.run_pass("cool-auto-destruction")
    
    # 3. Lower to LLVM IR
    let llvm_ir = mlir.lower_to_llvm(move mlir_module)
    
    # 4. Generate Machine Code (.o)
    let object_file = backend.emit_object(move llvm_ir)
    
    # 5. Link with Runtime
    # The driver finds the runtime path via $COOLPATH/lib/runtime.o
    let runtime_path = get_runtime_path()
    
    linker.link(
        target=output_name,
        input_files=[move object_file, view runtime_path],
        flags=["-lpthread", "-lc", "-static"]
    )
    
    print("Build successful: " + output_name)

```

---

## The Static Binary Advantage

By linking the runtime and the standard library statically, Coolscript achieves your goal of **C++ performance** and **Go-style deployment simplicity**:

* **Zero Dependencies**: The resulting binary does not require a local Python interpreter, a Go runtime, or even specific shared libraries (glibc is linked statically).
* **Instant Startup**: Since there is no VM to initialize or JIT to warm up, the binary starts in microseconds.
* **Predictable Execution**: Every `move` and `free` is determined at compile-time, so the execution speed is consistent from start to finish.

---

## Complete Project Roadmap (Milestone 1)

With the logic for the parser, semantic analyzer, MLIR dialect, and C runtime finalized, the implementation path is now clear:

| Step | Technical Task | Goal |
| --- | --- | --- |
| **1** | **Lexer/Parser** | Handle indentation and colons. |
| **2** | **Ownership Pass** | Enforce `move` and `view` semantics. |
| **3** | **Dialect Conversion** | Map Coolscript ops to LLVM-compatible IR. |
| **4** | **Runtime Link** | Integrate the `cs_chan` and `cs_spawn` C code. |

---

## Final Documentation Index

You now have a complete set of specifications for:

1. **Language Grammar** (PEG-based indentation logic).
2. **Memory Model** (Move/View and the No-Escape Rule).
3. **Concurrency Model** (Isolated Tasks and Owned Channels).
4. **Module System** (`cool.mod` and decentralized imports).
5. **Compiler Architecture** (MLIR lowering and C Runtime integration).
