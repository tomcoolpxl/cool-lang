This initial test suite, `tests/ownership_tests.cool`, is designed to verify that the compiler's semantic analyzer correctly enforces the **Move/View** rules and the **No-Escape Rule**. Each test case represents a fundamental memory safety requirement.

---

## Ownership and Memory Safety Test Suite

### 1. Basic Ownership Violations

These tests verify that the compiler correctly "burns" variables and prevents "Use-After-Move" errors.

```python
# test_invalid_use_after_move.cool
fn consume_string(s: move str):
    print(s)

fn main():
    let name = "Coolscript"
    consume_string(move name)
    
    # EXPECTED ERROR: Variable 'name' is burned. 
    # Attempted access at line 9.
    print(name) 

```

### 2. The No-Escape Rule

This test ensures that a `view` (borrow) cannot be stored in a long-lived structure or returned from a scope where its owner died.

```python
# test_view_escape.cool
struct DataBox:
    # EXPECTED ERROR: Structs cannot contain 'view' types.
    # DataBox must own its data.
    content: view str 

fn get_leaked_view() -> view str:
    let local_data = "Temporary"
    let v: view str = local_data
    
    # EXPECTED ERROR: View of 'local_data' escapes its scope.
    return v 

```

### 3. Loop Ownership Logic

This test verifies that the compiler understands ownership transitions within loops.

```python
# test_loop_move.cool
fn main():
    let items: List[str] = ["A", "B", "C"]
    let single_item = "D"

    for i in 0..3:
        # EXPECTED ERROR: 'single_item' is moved in the first iteration.
        # It is unavailable for subsequent iterations.
        process(move single_item) 

    # CORRECT PATTERN:
    for item in items:
        # This is safe because 'item' is a view provided by the iterator.
        process_view(item) 

```

### 4. Structured Concurrency (Spawn)

This test ensures that data is moved, not shared, when spawning background tasks.

```python
# test_spawn_safety.cool
fn worker(data: move List[i32]):
    pass

fn main():
    let shared_data = [1, 2, 3]
    
    # Pass 1: Correct Move
    spawn worker(move shared_data)
    
    # Pass 2: Attempting to access data after spawning the task
    # EXPECTED ERROR: 'shared_data' was moved into background task.
    print(len(shared_data)) 

```

### 5. Optional Unwrapping

This test verifies that the compiler forces the developer to check `opt[T]` types before access.

```python
# test_optional_access.cool
fn main():
    let name: opt[str] = None
    
    # EXPECTED ERROR: Type 'opt[str]' has no attribute 'len'.
    # print(name.len()) 

    # CORRECT PATTERN:
    if let n = name:
        print(n.len()) # Safe
    else:
        print("Empty")

```

---

## Testing CLI Integration

The compiler toolchain will use these files to verify its own correctness during development.

```bash
# Running the test suite
cool check tests/ownership_tests.cool

```

### Traceback Logic

When the compiler finds an error (e.g., in `test_invalid_use_after_move.cool`), it should produce a "Lifetime Traceback" to help the developer:

> **Ownership Error: Use of burned variable 'name'**
> * **Line 7**: `let name = "Coolscript"` (Initialized as owner)
> * **Line 8**: `consume_string(move name)` (Ownership moved here)
> * **Line 11**: `print(name)` (Invalid access of burned variable)
> 
> 

---

## Summary of Compiler Requirements for the Test Suite

| Test Category | Requirement | Implementation Strategy |
| --- | --- | --- |
| **Linearity** | No use-after-move | Track "Burned" state in the Symbol Table. |
| **Regions** | No view escape | Forbid `view` in structs; restrict return types. |
| **Exhaustiveness** | Match/Opt handling | Verify all paths in the AST for `Result`/`opt`. |
| **Isolation** | Spawn moves | Ensure all arguments to `spawn` are `move` or `copy`. |

This concludes the initial design phase of Coolscript. You now have a full specification, a module system, a standard library draft, a formal grammar, and a test suite to validate your implementation.

