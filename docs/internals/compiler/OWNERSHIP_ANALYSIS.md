To implement a truly safe "Move" model, the compiler must perform **Join-Point Analysis**. If a variable is moved inside an `if` block, the compiler must ensure it is either moved in the `else` block as well, or it must "poison" the variable so it cannot be used after the blocks merge.

Below is the logic for the **Branch-Aware Linear Analysis** pass, written in a style that can be implemented directly into your MLIR compiler driver.

---

## Branch-Aware Linear Analysis Logic

This algorithm ensures that ownership states remain consistent across branching logic.

### 1. The State Merger Algorithm

When the compiler reaches the end of an `if-else` structure, it calculates the "Least Common Ownership" state.

```python
# Pseudo-code for the Analysis Pass
fn analyze_branch(variable_id: i32, if_block: Block, else_block: Block):
    let state_after_if = trace_usage(variable_id, if_block)
    let state_after_else = trace_usage(variable_id, else_block)

    # Merger Logic:
    if state_after_if == "Burned" and state_after_else == "Burned":
        return "Burned" # Safe to consider it gone
    
    if state_after_if == "Owned" and state_after_else == "Owned":
        return "Owned" # Safe to keep using it
        
    # The Conflict Case:
    # One branch burned it, the other didn't.
    # To be safe, we must consider it "Partially Burned."
    return "Poisoned" 

```

---

## Example Case: The "Half-Burn" Error

This code demonstrates a bug that the analysis pass must catch to prevent a runtime crash.

```python
fn process(item: move List[i32]):
    pass

fn main():
    let data = [10, 20]
    let condition = get_input()

    if condition > 10:
        process(move data) # data is now BURNED
    else:
        print("Skipping") # data is still OWNED

    # JOIN POINT
    # At this point, the state is 'Poisoned'. 
    # The compiler doesn't know for sure if data exists.
    
    print(data) # COMPILER ERROR: Variable 'data' has inconsistent ownership.

```

### The Fix

The developer must ensure the variable is in a consistent state before the join point.

```python
    if condition > 10:
        process(move data)
    else:
        # We manually burn it or re-assign it to maintain consistency
        burn data 

```

---

## Implementation into MLIR

In your C++ MLIR pass, you will implement this as a **Dataflow Analysis**.

1. **Map Initialization**: Create a `DenseMap<Value, OwnershipState>` for the entry block.
2. **Transfer Functions**:
    * `cool.move` transforms `Owned -> Burned`.
    * `cool.borrow` checks if `Owned`. Adds `ReadLocked` state (optional optimization).
    * `cool.inout` transforms `Owned -> WriteLocked`. The original variable is inaccessible until the scope of the `inout` ends, at which point it returns to `Owned`.
3. **Meet Operator**: At the end of a region (like an `if` or `match`), you perform the "Meet" operation described above.

---

## Compiler Diagnostic Output

When the Join-Point Analysis fails, the compiler should provide a "Branch Traceback" to help the developer fix the logic.

> **Ownership Inconsistency Error**
> * Variable **'data'** was moved at line 8 (inside `if` block).
> * Variable **'data'** was NOT moved at line 10 (inside `else` block).
> * **Result**: Accessing **'data'** at line 13 is unsafe.
> 
> 

---

## Final Project Structure Re-check

With this Branch-Aware logic, you have covered the final "hard" part of Milestone 1.

| Feature | Logic |
| --- | --- |
| **Simple Move** | Linearity (use-after-move). |
| **Views** | No-Escape (no return/store). |
| **Branches** | Join-Point Analysis (consistency). |
| **Concurrency** | Isolation (move-only spawn). |

To formalize branch tracking in the **Coolscript** compiler, we define the `if` and `match` operations using MLIR's **Operation Definition Specification (ODS)**. This allows the TableGen tool to generate the C++ classes required for your semantic passes to visit and analyze branches.

---

## MLIR TableGen (ODS) for Coolscript Branches

These definitions ensure that branches are treated as **Regions**. In MLIR, a Region allows the Linear Analysis pass to step "inside" the branch, calculate a local ownership state, and then compare it to the sibling branch.

### 1. The `cool.if` Operation

This operation contains two regions: `thenRegion` and `elseRegion`.

```tablegen
def IfOp : Cool_Op<"if", [RecursiveMemoryEffects, NoTerminator]> {
    let summary = "Conditional branch with ownership isolation";
    let description = [{
        The `cool.if` operation takes a boolean condition and executes 
        one of two regions. The Linear Analysis pass must verify that
        any 'move' operations within these regions result in a 
        consistent state at the merge point.
    }];

    let arguments = (ins I1:$condition);
    let regions = (any_region:$thenRegion, any_region:$elseRegion);

    let assemblyFormat = [{
        $condition attr-dict-with-keyword $thenRegion (`else` $elseRegion)?
    }];
}

```

### 2. The `cool.match` Operation

The `match` operation is more complex because it handles multiple patterns (variants). It is defined as having a variable number of regions.

```tablegen
def MatchOp : Cool_Op<"match", [RecursiveMemoryEffects, NoTerminator]> {
    let summary = "Pattern matching for Enums and Protocols";
    let description = [{
        Executes the region corresponding to the matched variant.
        All regions must agree on the final ownership state of 
        variables declared outside the match.
    }];

    let arguments = (ins AnyType:$input);
    let regions = (variadic_region:$caseRegions);

    let assemblyFormat = [{
        $input attr-dict-with-keyword $caseRegions
    }];
}

```

---

## Integrating ODS with the Linear Pass

By defining these as operations with regions, your **Dataflow Analysis** becomes much cleaner. When the analyzer encounters a `cool.if`, it follows this internal logic:

1. **Clone the Environment**: Create a copy of the current variable-to-state map.
2. **Visit `thenRegion**`: Apply all moves and borrows. Save the result as `StateA`.
3. **Reset and Visit `elseRegion**`: Using the original map, apply all moves and borrows. Save as `StateB`.
4. **Compute Least Upper Bound (LUB)**:
* If `x` is `Burned` in `StateA` but `Owned` in `StateB` → **Error: Inconsistent Ownership.**
* The compiler can then suggest: *"Did you forget to move 'x' in the else branch?"*



---

## Pattern Matching and Destructuring

In a `cool.match`, the input is often an `Enum` or `opt[T]`. The ODS allows the compiler to bind a new name to the "unwrapped" value inside the region.

```python
# Coolscript
match result:
    Ok(val):
        # 'val' is a NEW owned variable inside this Region
        process(move val)
    Err(e):
        print(e)

```

In the MLIR, the `Ok` region will have an **Entry Block Argument** representing the unwrapped `val`. This naturally handles ownership because the parent `result` is consumed by the match, and the new `val` begins its lifecycle as `Owned` only within that specific case region.

---

## Final Compilation Pipeline

With the TableGen files in place, your `cool build` command now has the formal structure to handle complex logic:

1. **PEG Parser** → Generates AST.
2. **MLIR Gen** → Converts AST to `cool.if` and `cool.match` ODS operations.
3. **Analysis Pass** → Uses the ODS regions to perform the Join-Point ownership check.
4. **Lowering** → Converts regions into standard `cf.cond_br` (LLVM-style branches).

---

## Summary of Milestone 1 Technical Assets

You now have a complete, professional-grade specification for the compiler:

* **Syntax**: PEG Grammar.
* **Semantics**: Ownership, No-Escape, and Join-Point Analysis.
* **Infrastructure**: MLIR ODS definitions.
* **Backend**: C Runtime and LLVM mapping.
