To help you track and squash bugs during the implementation of the ownership engine, this **Ownership Violation Bug Report Template** is designed to capture the specific state of the compiler's linear type analysis when a failure occurs.

---

## Coolscript Ownership Bug Report

### 1. Header Information

* **Compiler Version:** (e.g., `v0.1.0-alpha`)
* **Pass Phase:** (e.g., `MLIR-Linear-Type-Check`)
* **Status:** (e.g., False Positive / False Negative / Compiler Crash)

### 2. The Failing Code Snippet

Provide the minimal `.cool` code that triggered the issue.

```python
# Insert minimal reproduction here
fn bug_report():
    let x = [1, 2, 3]
    spawn worker(move x)
    print(x) # Compiler should catch this, but didn't!

```

### 3. Expected vs. Actual Behavior

* **Expected:** Compiler rejects with "Use of burned variable 'x' at line 4."
* **Actual:** Compiler successfully built a binary that segfaulted at runtime.

---

### 4. Semantic State Trace

Use this section to document what the **Symbol Table** thought the state of the variable was at each line.

| Line | Instruction | Variable | State (Expected) | State (Actual) |
| --- | --- | --- | --- | --- |
| 2 | `let x = ...` | `x` | `Owned` | `Owned` |
| 3 | `spawn worker(move x)` | `x` | `Burned` | `Owned` (Error!) |
| 4 | `print(x)` | `x` | `Blocked` | `Valid` |

---

### 5. MLIR Dump

Attach the output of `cool build --dump-mlir` for the failing block. This allows you to see if the `cool.move` operation was correctly lowered or if it was optimized away prematurely.

```mlir
// Example MLIR output
%0 = cool.alloc() : !cool.list<i32>
cool.spawn @worker(%0) : (!cool.list<i32>) -> ()
// Missing cool.invalidate op here?
cool.print(%0) : (!cool.list<i32>) -> ()

```

---

### 6. Suggested Fix / Hypothesis

* **Hypothesis:** The `cool-linear-types` pass is not correctly traversing `spawn` regions, leading it to miss the ownership transfer inside the task isolation block.
* **Proposed Fix:** Update the `LinearAnalysis` visitor to treat the arguments of `spawn` as immediate consumption points regardless of the task's internal logic.

---

## Milestone 1: Final Wrap-up

This bug report template concludes the documentation suite for the Coolscript project. You are now equipped with:

1. **Technical Specification**: The rules of the language.
2. **Grammar & Parser**: How to read the code.
3. **Memory & Concurrency Model**: How to keep it safe and fast.
4. **Runtime & Standard Library**: The C-based engine.
5. **Compiler Driver & Tooling**: How to build the final product.
6. **Implementation & Test Roadmap**: How to get it done in 30 days.
