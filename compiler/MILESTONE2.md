Milestone 1 established the foundation: a working compiler that handles ownership, structured concurrency, and linear analysis for concrete types. **Milestone 2** elevates Coolscript from a systems tool to a scalable language by introducing **Compile-Time Generics** and **Trait-Based Polymorphism**.

This phase focuses on "Monomorphization" (making generics as fast as concrete types) and "Protocol Bounds."

---

## Milestone 2: Generics and Protocol Bounds

### 1. Zero-Cost Generics (Monomorphization)

Coolscript generics do not use "Type Erasure" (like Java) or "Interface Objects" (like Go). Instead, it uses the C++/Rust approach: the compiler generates a unique, optimized version of the function for every type used.

* **Syntax**: `List[T]`, `Map[K, V]`, `fn process[T](item: T)`.
* **Requirement**: The compiler must track which versions of a generic function are needed across all `.cool` files and generate them during the LLVM pass.

### 2. Protocol Bounds (Constraints)

To perform operations on a generic type, you must prove it implements a specific Protocol.

```python
# T must implement the 'Comparable' protocol
fn find_max[T: Comparable](a: T, b: T) -> T:
    if a > b:
        return a
    return b

```

---

## Phase 5: Generics Implementation (Days 31–45)

* **Day 31–35: AST & Type System Updates.** Update the parser to support `[T]` brackets and the symbol table to store "Type Templates" instead of just concrete types.
* **Day 36–40: Substitution Engine.** Implement the logic that replaces `T` with `i32` or `str` and verifies that the resulting code is valid.
* **Day 41–45: Generic Ownership Analysis.** This is the hardest part. The compiler must verify ownership rules for `move T` where `T` is unknown.
* *Rule:* If a generic type is moved, the compiler must ensure that the specific type used later is indeed "Moveable" (which all Coolscript types are by default).



---

## Phase 6: Advanced Protocols & Associated Types (Days 46–60)

* **Day 46–52: Associated Types.** Support protocols that define an internal type, like an `Iterator` defining its `Item`.
* **Day 53–60: The "VTable" Optimization.** For cases where you want a `List[Protocol]` (Dynamic Dispatch), implement the "Fat Pointer" logic in MLIR.

```python
protocol Iterator:
    type Item
    fn next(view self) -> opt[Item]

```

---

## Milestone 2 Success Criteria

| Feature | Requirement |
| --- | --- |
| **Performance** | Generic code must be exactly as fast as hard-coded concrete code. |
| **Safety** | Generics must respect the No-Escape Rule (e.g., a `List[view str]` is forbidden). |
| **Usability** | Error messages must clearly point to the specific line in the generic template that failed. |

---

## Technical Challenge: The "Generic Leak"

A common bug in generic compilers is failing to call the destructor of a generic type `T`.

**Example Challenge:**

```python
fn drop_it[T](item: move T):
    # The compiler doesn't know what T is.
    # It MUST inject a call to the destructor of whatever T ends up being.
    pass 

```

Your Milestone 2 task is to ensure that the **MLIR Auto-Destruction Pass** can handle these "Opaque Destructors" by looking up the correct `free` function at the moment of monomorphization.

