This **30-Day Implementation Roadmap** breaks down the development of the **Coolscript** compiler into four distinct phases. Each phase builds the technical infrastructure required to support the unique memory model and syntax defined in the specification.

---

## Phase 1: The Parser and AST (Days 1–7)

**Goal:** Convert `.cool` source code into an Abstract Syntax Tree (AST) that understands indentation.

* **Day 1–3: Lexer and Indentation Tracker.** Implement the logic to convert physical whitespace into `INDENT` and `DEDENT` tokens. This is the foundation of the Pythonic aesthetic.
* **Day 4–5: PEG Parser Implementation.** Use the provided PEG grammar to build the parser. Focus on top-level declarations: `struct`, `protocol`, and `fn`.
* **Day 6–7: Expression and Statement Parsing.** Handle control flow (`if`, `while`, `for`) and the explicit `move`/`view`/`inout`/`copy` keywords in function calls.

---

## Phase 2: Semantic Analysis and Ownership (Days 8–15)

**Goal:** Implement the "Static Safety" engine. This is the most critical phase for Coolscript.

* **Day 8–10: Symbol Table and Type Resolution.** Resolve types and ensure that `opt[T]` and `Result[T, E]` are treated as distinct wrappers that require unwrapping.
* **Day 11–13: The Ownership Tracker (Linearity Pass).**
* Implement a "state" for every variable: `Available`, `Borrowed (Viewed)`, `Mutably Borrowed (Inout)`, or `Burned`.
* Verify that any use of the `move` keyword transitions a variable to the `Burned` state.
* Generate the "Ownership Traceback" error messages for invalid accesses.


* **Day 14–15: Transient Analysis (No-Escape).** Implement the logic to mark structs containing views as "Transient" and enforce that they cannot outlive their referents or be moved to long-lived scopes.

---

## Phase 3: MLIR and Code Generation (Days 16–23)

**Goal:** Lower the validated AST into a machine-readable intermediate representation.

* **Day 16–18: Dialect Definition.** Define the `cool` MLIR dialect. Create operations for `cool.move`, `cool.borrow`, `cool.inout`, and `cool.box` (for Protocols).
* **Day 19–21: AST-to-MLIR Lowering.** Translate the semantic-checked AST into the MLIR dialect.
* **Day 22–23: Memory Lifetime Injection.** Since there is no GC, the compiler must inject `llvm.free` calls (and `Drop.drop` invocations) at the exact point a variable is "Burned" or its owner goes out of scope.

---

## Phase 4: Concurrency and Runtime (Days 24–30)

**Goal:** Support the `spawn` keyword and generate a final binary.

* **Day 24–26: Structured Concurrency Implementation.**
* Implement the `spawn` logic: lowering it to a thread-pool call.
* Ensure that the "Isolation" check is enforced (only `move` or `copy` arguments allowed in `spawn`).


* **Day 27–28: Standard Library (Prelude).** Build the minimal C-based runtime for basic I/O (`print`, `str`) and the `Channel` implementation.
* **Day 29: Static Linking.** Integrate the linker to combine the LLVM-generated object files with the standard library into one `.cool` binary.
* **Day 30: Self-Correction.** Run the `ownership_tests.cool` suite. If the compiler successfully rejects all invalid code and compiles all valid code, the milestone is achieved.

---

## Success Criteria for Month 1

1. **Safety**: The compiler must catch a "Use-After-Move" error 100% of the time.
2. **Deployment**: A "Hello World" written in `.cool` must compile to a single binary that runs on a machine without the compiler installed.
3. **Performance**: Compilation of a 1,000-line file should take less than 1 second.
