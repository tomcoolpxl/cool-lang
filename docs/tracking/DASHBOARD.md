This **Coolscript Implementation Dashboard** is your master roadmap. It breaks down the 90-day development cycle into granular, actionable tasks based on the technical specifications we have established.

---

## Coolscript Implementation Dashboard (90-Day Sprint)

### Milestone 1: The Core Engine (Days 1–30)

| Week | Focus | Key Deliverables | Status |
| --- | --- | --- | --- |
| **1** | **Frontend** | Indentation Stack & Lexer | [x] |
| **1** | **Frontend** | PEG Parser & AST Nodes | [ ] |
| **2** | **Ownership** | Linear Type Pass, Move/View Validation | [ ] |
| **3** | **MLIR/LLVM** | `cool` Dialect, Lowering to LLVM IR, `cs_alloc` | [ ] |
| **4** | **Runtime** | C-based Channels, Task Spawning, Static Linker | [ ] |

---

### Milestone 2: Scaling & Ergonomics (Days 31–60)

| Week | Focus | Key Deliverables | Status |
| --- | --- | --- | --- |
| **5** | **Generics** | Monomorphizer, Type Parameter Substitution | [ ] |
| **6** | **Protocols** | Protocol Implementation Map (PIM), Constraints | [ ] |
| **7** | **Polymorphism** | VTable Generation, Protocol Boxing (`cool.box`) | [ ] |
| **8** | **StdLib** | Iterators, `std.fs`, Result-based Error Handling | [ ] |

---

### Milestone 3: The Self-Hosting Singularity (Days 61–90)

| Week | Focus | Key Deliverables | Status |
| --- | --- | --- | --- |
| **9** | **Macros** | `@compiler` Isolate, AST Manipulation API | [ ] |
| **10** | **Reflection** | Compile-time Metadata Registry, `@derive` Logic | [ ] |
| **11** | **Self-Host** | Porting Parser and Linear Analyzer to `.cool` | [ ] |
| **12** | **Bootstrap** | Stage 0 -> Stage 1 -> Stage 2 Compilation | [ ] |

---

## Critical Path Checklist

### Safety Guardrails

* [ ] **Double-Move Check**: Ensure the compiler catches `move x` followed by `move x`.
* [ ] **Transient Struct Check**: Ensure structs containing views ("Transient") cannot be stored in long-lived structures or globals.
* [ ] **Isolation Check**: Ensure only `move` types can cross the `spawn` boundary.

### Performance Benchmarks

* [ ] **Static Binary Size**: Verify "Hello World" is under 1.5MB.
* [ ] **Latency Consistency**: Confirm zero GC pauses during high-load channel tests.
* [ ] **Zero-Copy Verification**: Ensure `move List` only passes a 64-bit pointer.

### Tooling & DX

* [ ] **`cool build`**: Single-command static compilation.
* [ ] **`cool mod`**: Decentralized dependency resolution.
* [ ] **Error Messages**: Implementation of "Branch Traceback" for ownership errors.

---

## Project Status Tracking

> **Current Phase**: Implementation (Milestone 1)
> **Next Immediate Task**: Implement the PEG Parser and AST generation (Milestone 1, Week 1).

