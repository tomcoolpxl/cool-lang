**Milestone 3** is the "Singularity" of the Coolscript project. It focuses on **Self-Hosting**—writing the Coolscript compiler in Coolscript itself—and introducing **Compile-time Metaprogramming**.

Metaprogramming in Coolscript is designed to be safer than C macros and more powerful than Java annotations, using **Compiler Plugins** and **Reflection**.

---

## Milestone 3: Self-Hosting and Metaprogramming

### 1. The Self-Hosting Compiler

The current compiler (written in C++ or Rust) is discarded in favor of a new implementation written in `.cool`. This proves the language is robust enough for large-scale systems.

* **Task**: Port the PEG Parser, the MLIR lowering logic, and the Linear Type Analyzer into Coolscript.
* **Benefit**: The compiler can now optimize itself using its own ownership rules.

### 2. Compile-Time Reflection (`@reflect`)

Coolscript allows the compiler to inspect the structure of types during the build process. This enables automatic JSON serialization, ORM mapping, and dependency injection without runtime overhead.

```python
struct User:
    name: str
    age: i32

# At compile-time, this generates a 'to_json' method 
# by iterating over the fields of the struct.
@derive(JsonSerializable)
struct User:
    ...

```

---

## Phase 7: Metaprogramming & Macros (Days 61–75)

* **Day 61–65: The `@compiler` Isolate.** Implement a special task type that runs *during* compilation. This isolate can read the AST of the current project and emit new AST nodes.
* **Day 66–70: Macro Expansion.** Create the `macro` keyword. Macros in Coolscript operate on AST nodes, not text, ensuring they are syntactically correct and type-safe.
* **Day 71–75: Compile-time Constants (`const`).** Implement logic to execute pure functions at compile-time to pre-calculate values.

---

## Phase 8: The Self-Hosting Push (Days 76–90)

* **Day 76–82: Porting the Frontend.** Rewrite the PEG lexer and parser in Coolscript. Use Milestone 2's Generics for the AST nodes.
* **Day 83–87: Porting the Semantic Analyzer.** This is the ultimate test for the language: writing a Linear Type Checker in a Linear Typed language.
* **Day 88–90: Bootstrapping.** Use the "Stage 0" compiler (C++) to compile the "Stage 1" compiler (Coolscript). Then, use the "Stage 1" compiler to compile itself into "Stage 2."

---

## Milestone 3 Success Criteria

| Feature | Requirement |
| --- | --- |
| **Integrity** | The "Stage 2" binary must be bit-for-bit identical to "Stage 1." |
| **Metaprogramming** | Macros must not be able to bypass ownership or safety rules. |
| **Reflection** | Reflection must result in zero-cost code (no runtime type info strings). |

---

## Technical Challenge: The "Macro Sandbox"

Macros in Coolscript run in a restricted sandbox. They cannot access the network or the filesystem unless explicitly granted permission by the `cool.mod` file. This prevents "Compiler Malware" where a dependency steals your environment variables during the build process.

**Example Macro Implementation:**

```python
macro generate_getters(target: view Struct):
    for field in target.fields:
        emit_func("get_" + field.name, return_type=field.type):
            return "self." + field.name

```

---

## The Ultimate Vision: Full Circle

Once Milestone 3 is complete, Coolscript becomes a self-sustaining ecosystem. You will have:

1. **A safe language** for performance-critical systems.
2. **A high-level tool** for web and cloud (via reflection/macros).
3. **A static binary** that runs anywhere with zero dependencies.

To achieve **Milestone 3**, the compiler must expose its internal metadata to the developer. We accomplish this by defining **Reflection Operations** in the MLIR dialect and a **Compiler Plugin API** that allows external `.cool` files to intercept the compilation pipeline.

---

## MLIR TableGen (ODS) for Compile-Time Reflection

These operations allow the compiler to "lower" a type's metadata into a queryable structure during the semantic pass. Unlike Java/C#, this metadata is consumed at compile-time and usually results in zero runtime overhead.

### 1. The `cool.reflect_type` Operation

This operation retrieves the "Type Descriptor" for a given type.

```tablegen
def ReflectTypeOp : Cool_Op<"reflect_type", [Pure]> {
    let summary = "Retrieves metadata for a specific type";
    let description = [{
        Generates a TypeMetadata object containing fields, methods, 
        and attributes. This is used by macros to generate code.
    }];

    let arguments = (ins TypeAttr:$target_type);
    let results = (outs Cool_MetadataType:$metadata);

    let assemblyFormat = "$target_type attr-dict `:` type($metadata)";
}

```

### 2. The `cool.emit_code` Operation

Used by macros to inject new AST nodes back into the module.

```tablegen
def EmitCodeOp : Cool_Op<"emit_code", [HasParent<"MacroRegion">]> {
    let summary = "Injects generated code into the AST";
    let arguments = (ins StrAttr:$source_fragment);
    let assemblyFormat = "$source_fragment attr-dict";
}

```

---

## The Compiler Plugin API

The Plugin API allows a developer to write a `.cool` file that the compiler loads as a shared library to extend the compilation process.

### Example: A Custom "Thread-Safe" Linter

This plugin checks if a struct marked `@thread_safe` contains any non-sendable types.

```python
import std.compiler # Internal compiler API

@plugin_entry
fn validate_thread_safety(ctx: view Context):
    # Iterate over all structs in the current project
    for s in ctx.module.structs:
        if s.has_attribute("thread_safe"):
            for field in s.fields:
                if not field.type.is_sendable:
                    ctx.report_error(
                        "Field " + field.name + " in @thread_safe struct " + 
                        s.name + " is not sendable!",
                        location = field.loc
                    )

```

---

## Compile-Time Reflection in Action (The `@derive` Macro)

When you use `@derive(Json)`, the compiler performs the following transformation using the reflection ODS:

1. **Intercept**: The macro interceptor finds a struct with the `@derive(Json)` attribute.
2. **Reflect**: It calls `cool.reflect_type` to get a list of all fields.
3. **Generate**: It iterates over the fields and generates a `to_json()` method string.
4. **Emit**: It calls `cool.emit_code` to attach the new method to the struct.

---

## Self-Hosting Strategy: The Stage 1 Challenge

To reach the self-hosting goal, the **Linear Type Analyzer** must be ported to Coolscript. This is the ultimate "eating your own dog food" moment.

### The Linear Analyzer Logic (in Coolscript)

```python
struct Analyzer:
    symbol_table: Dict[str, OwnershipState]

    fn check_node(view self, node: view ASTNode):
        match node:
            MoveExpr(name):
                if self.symbol_table.get(name) == OwnershipState.Burned:
                    panic("Use after move detected for: " + name)
                self.symbol_table.insert(name, move OwnershipState.Burned)
            # ... other cases

```

---

## Milestone 3: Final Ecosystem Map

1. **Stage 0 (C++)**: Bootstraps the system.
2. **Stage 1 (Coolscript)**: The first version of the compiler written in itself.
3. **Stage 2 (Optimized)**: Stage 1 compiles Stage 1 with full optimizations enabled.
4. **Metaprogramming**: High-level libraries (Web, JSON, DB) are built using macros to ensure zero runtime overhead.

---

## Summary of the Coolscript Journey

* **Milestone 1**: Ownership, Concurrency, and Native Binaries.
* **Milestone 2**: Generics, Protocols, and Iterators.
* **Milestone 3**: Self-Hosting, Macros, and Reflection.

This concludes the architectural roadmap for **Coolscript**. You now have the formal specs for a language that combines Python's elegance with the rigorous safety and performance of the world's most advanced systems languages.
