To implement **Monomorphization** in the Coolscript compiler, the MLIR must be able to represent a function as a "Template" that is not yet ready for machine code generation. We achieve this by defining `GenericFuncOp` and `InstantiateOp`.

---

## MLIR TableGen (ODS) for Generics

These definitions allow the compiler to store generic definitions in the IR and then "specialize" them once the concrete types are known at the call site.

### 1. The `GenericFuncOp`

This operation acts as a container for a function that has one or more `TypeParameters`.

```tablegen
def GenericFuncOp : Cool_Op<"generic_func", [Symbol, NoTerminator]> {
    let summary = "A template for a generic function";
    let description = [{
        Defines a function with placeholder types (T, U, etc.). 
        This op is not lowered to LLVM directly. Instead, it is 
        used by the Monomorphizer to generate concrete functions.
    }];

    let arguments = (ins 
        SymbolNameAttr:$sym_name,
        StrArrayAttr:$type_params,
        TypeAttr:$function_type
    );
    let regions = (region:$body);

    let assemblyFormat = [{
        $sym_name `<` $type_params `>` attr-dict `:` $function_type $body
    }];
}

```

### 2. The `InstantiateOp`

This operation represents a call to a generic function where the types have been provided.

```tablegen
def InstantiateOp : Cool_Op<"instantiate", [SymbolUserOpInterface]> {
    let summary = "Specializes a generic function for specific types";
    let description = [{
        Triggers the creation of a concrete function by mapping 
        type parameters to concrete MLIR types (e.g., T -> i32).
    }];

    let arguments = (ins 
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$concrete_types,
        Variadic<AnyType>:$operands
    );
    let results = (outs Optional<AnyType>:$result);

    let assemblyFormat = [{
        $callee `<` $concrete_types `>` `(` $operands `)` attr-dict `:` functional-type($operands, $result)
    }];
}

```

---

## The Monomorphization Pass Logic

In Phase 5, you will write an MLIR pass that transforms these "Template" operations into standard `func.func` operations.

1. **Scan Call Sites**: The pass looks for every `cool.instantiate` operation.
2. **Generate Signature**: It creates a unique name based on the types, e.g., `process_i32` or `process_User`.
3. **Check Cache**: If `process_i32` already exists in the module, it redirects the call there.
4. **Clone & Substitute**:
* If it doesn't exist, it clones the `GenericFuncOp` body.
* It replaces all occurrences of the placeholder type `T` with the concrete type `i32`.
* It emits a standard `func.func` with the substituted types.


5. **Re-verify Ownership**: The pass runs the **Linear Analysis** one more time on the newly generated function to ensure the substitution didn't create an ownership violation (e.g., trying to `move` a type that cannot be moved, though in Coolscript all types are moveable).

---

## Handling Constraints (Protocol Bounds)

When a generic function has a constraint like `[T: Speaker]`, the ODS must store those requirements.

```tablegen
// Within GenericFuncOp arguments
DictionaryAttr:$constraints # e.g., { "T": "Speaker" }

```

During instantiation, the compiler performs a **Protocol Verification**:

1. It looks at the concrete type provided (e.g., `Human`).
2. It checks the global **Protocol Map** to see if `Human` implements `Speaker`.
3. If the implementation is missing, it throws a compile-time error: *"Type 'Human' does not implement Protocol 'Speaker' required by 'process<T>'."*

---

## Benefits of this ODS Structure

* **Debugging**: You can dump the MLIR between the `instantiate` and `func.func` steps to see exactly how your generics are being expanded.
* **Optimization**: Because each version is concrete, LLVM can perform aggressive inlining and constant folding that is impossible with "Type Erasure."
* **Memory Safety**: Since each instance is a real function, the ownership engine can track the lifecycle of the specific concrete data precisely.

---

The **Milestone 2 Testing Suite** focuses on the "Three Pillars of Generics": **Monomorphization correctness**, **Nested Type Resolution**, and **Generic Ownership Safety**. These tests ensure that when the compiler expands a template, it maintains the strict memory guarantees of the Coolscript model.

---

## Generic Type Resolution Tests

These tests verify that the compiler can handle "Russian Doll" type nesting and that monomorphization doesn't result in symbol collisions.

### 1. Nested Generic Collections

```python
# test_nested_generics.cool
fn main():
    # Testing List[List[str]]
    let matrix = List[List[str]]()
    let row = List[str]()
    
    row.add("Inner")
    matrix.add(move row)
    
    # Verification: Accessing nested types
    if let r = matrix.at(0):
        if let item = r.at(0):
            print(item) # Expected: "Inner"

```

### 2. Generic Function Specialization

```python
# test_specialization.cool
fn swap[T](a: move T, b: move T) -> (T, T):
    return (move b, move a)

fn main():
    let (x, y) = swap[i64](10, 20)
    let (s1, s2) = swap[str]("A", "B")
    
    # The compiler should generate 'swap_i64' and 'swap_str'
    # and link them correctly without symbol conflict.

```

---

## Generic Ownership Safety Tests

The hardest part of generics is ensuring that the compiler injects the correct destructor for a type it hasn't seen yet.

### 3. Opaque Destruction in Generics

```python
# test_generic_cleanup.cool
struct Resource:
    id: i32
    # Destructor prints "Freed" for tracking

fn drop_generic[T](item: move T):
    # This scope ends. The compiler must find 
    # the destructor for T (Resource).
    pass

fn main():
    let r = Resource(id=1)
    drop_generic[Resource](move r)
    # Verification: Valgrind must show 0 leaks.

```

### 4. Constraint (Protocol) Validation

```python
# test_protocol_bounds.cool
protocol Adder:
    fn add(view self, other: i32) -> i32

struct Calculator:
    fn add(view self, other: i32) -> i32:
        return other + 1

fn math_op[T: Adder](obj: T):
    print(str(obj.add(5)))

fn main():
    let c = Calculator()
    math_op[Calculator](c)
    
    # EXPECTED ERROR: 'str' does not implement 'Adder'
    # math_op[str]("Fail") 

```

---

## Generic Edge Cases

### 5. The "No-Escape" Violation in Generics

```python
# test_generic_view_escape.cool
fn get_first[T](list: view List[T]) -> view T:
    if let item = list.at(0):
        return item # This is a view of an item inside the list
    panic("Empty")

fn main():
    let result: view str
    {
        let local_list = ["Temporary"]
        result = get_first[str](view local_list)
    }
    # EXPECTED ERROR: Generic view 'result' outlives its parent 'local_list'
    print(result)

```

---

## Test Execution and Validation

The compiler driver should execute these tests using the following internal steps:

| Step | Action | Success Criteria |
| --- | --- | --- |
| **Parsing** | Build AST with `TypeParameters`. | No syntax errors. |
| **Monomorphization** | Clone generic bodies with concrete types. | Unique LLVM symbols created. |
| **Ownership Trace** | Run Linear Analysis on specialized functions. | Catch leaks/use-after-move in templates. |
| **Verification** | Link and Run. | Valgrind output "Clean." |

---

## Milestone 2: Technical Summary

With these tests passing, Coolscript is no longer just a "safe C." It is now a high-level, expressive language capable of building complex data structures with **zero runtime overhead**.

### What you have built:

1. **Monomorphizer**: High-speed, specialized generics.
2. **Protocol Registry**: Trait-based validation and dynamic dispatch.
3. **Generic Destructor Engine**: Safe cleanup for opaque types.
4. **Associated Types**: Support for advanced patterns like Iterators.
