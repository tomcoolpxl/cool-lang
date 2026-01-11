To implement **Phase 3** of the Coolscript compiler, you must define how the ownership semantics of the high-level AST map to low-level machine operations. Using **MLIR (Multi-Level Intermediate Representation)** allows us to keep the "Move/View" logic intact during optimization before lowering to LLVM IR.

Below are the operational definitions for the `cool` dialect.

---

## MLIR Dialect Operations for Coolscript

In MLIR, we define a custom dialect `cool` that tracks the lifecycle of resources.

### 1. The `cool.move` Operation

This operation marks the point where ownership is transferred. In the MLIR dataflow, it consumes an SSA value and produces a new one, effectively "killing" the original register.

```mlir
// MLIR Syntax
%new_owner = cool.move %original_value : !cool.string

// Lowering Logic:
// 1. Mark %original_value as 'invalid' for subsequent ops.
// 2. Transfer the pointer without copying the underlying heap data.
// 3. If %original_value is not used by %new_owner, inject a destructor call.

```

### 2. The `cool.borrow` (View) Operation

This operation creates a scoped reference. It utilizes MLIR's **Region** system to ensure the "view" cannot outlive the owner.

```mlir
// MLIR Syntax
%view = cool.borrow %owner { scope = @current_block } : (!cool.user) -> !cool.view<user>

// Lowering Logic:
// 1. Creates a raw pointer to the data.
// 2. The 'scope' attribute tells the optimizer that if %owner is burned
//    within this scope, it is a compilation error.

```

---

## The MLIR Lowering Table

The following table defines how high-level Coolscript code is lowered into MLIR and eventually to LLVM.

| Coolscript Syntax | MLIR Operation | LLVM (Final Backend) |
| --- | --- | --- |
| `let x = User(...)` | `cool.alloc` | `call i8* @malloc(...)` |
| `process(move x)` | `cool.move` | No-op (Pointer pass) |
| `process(x)` | `cool.borrow` | No-op (Pointer pass) |
| `process(copy x)` | `cool.copy` | `call i8* @memcpy(...)` |
| `end of scope` | `cool.dealloc` | `call void @free(i8*)` |

---

## Memory Safety Injection Pass

During the transformation from `cool` dialect to LLVM, the compiler performs an **Automatic Destruction Pass**. This is where Coolscript avoids the need for a Garbage Collector.

### Scenario: Move inside a function

If a variable is moved into a function, the caller's stack frame no longer owns it. The *callee* is now responsible for freeing it.

```python
# Coolscript
fn main():
    let s = "Resource"
    send_to_worker(move s) 
    # No 'free' injected here; ownership was moved.

```

### Scenario: Variable goes out of scope

If a variable is never moved and reaches the end of its block, the compiler injects the destructor.

```python
# Coolscript
fn main():
    let s = "Resource"
    print(s) # Passed as view
    # MLIR Pass detects 's' is still owned and scope is ending.
    # Injects: cool.dealloc %s

```

---

## Handling the "Boxed" Protocol Logic

When a struct is treated as a Protocol (e.g., `List[Speaker]`), the MLIR must handle the **Fat Pointer** generation.

```mlir
// MLIR Representation of a boxed Protocol
%fat_ptr = cool.box %human_instance as !cool.protocol<Speaker>

// This creates a struct: { ptr_to_data, ptr_to_vtable }

```

---

## Implementation Task: The Linear Type Pass

To start Phase 3, you must write an **MLIR Pass** that iterates through the blocks. Its logic should be:

1. **Initialize**: Every `cool.alloc` or `alloca` starts in the `Owned` state.
2. **Consume**: Every `cool.move` checks if the input value is in the `Owned` state. If it is, transition it to `Burned`. If it is already `Burned`, throw a **Use-After-Move** error.
3. **Validate Borrow**: Every `cool.borrow` checks if the input value is `Owned`. If it is `Burned`, throw an error.
4. **Finalize**: At the end of every block, any value still in the `Owned` state must have a `cool.dealloc` operation appended.

This pass ensures that memory is handled with C++ speed but without the risk of manual `free()` mistakes.
