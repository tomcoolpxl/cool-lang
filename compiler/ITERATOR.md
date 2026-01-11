In Milestone 2, the **Iterator Protocol** leverages **Associated Types** and **Generics** to provide the ergonomic feel of a Python `for` loop while maintaining the performance of a C pointer increment. Because Coolscript uses **Monomorphization**, the overhead of the iterator is optimized away by LLVM during compilation.

---

## The Iterator Protocol Definition

The `Iterator` protocol uses an associated type, `Item`, to define what it produces. This allows a single loop structure to handle anything from integers to complex owned structs.

```python
protocol Iterator:
    # The type of value returned by the iterator
    type Item
    
    # Returns Some(Item) if available, or None to terminate the loop
    fn next(view self) -> opt[Item]

```

---

## Implementing an Iterator: The Range Example

To make `for i in 0..10` work, the standard library defines a `Range` struct that implements the `Iterator` protocol.

```python
struct Range:
    start: i64
    end: i64
    current: i64

# Implementation map for Range as an Iterator
protocol impl Iterator for Range:
    type Item = i64
    
    fn next(view self) -> opt[i64]:
        if self.current < self.end:
            let val = self.current
            self.current += 1
            return Some(val)
        return None

```

---

## Lowering the `for` loop to MLIR

When the compiler sees a `for` loop, it transforms it into a `while` loop that interacts with the `Iterator` protocol.

**Coolscript Source:**

```python
for x in 0..10:
    print(x)

```

**Lowered MLIR Logic:**

```mlir
// 1. Initialize the iterator
%range = cool.call @Range_init(0, 10)

// 2. Loop Header
cool.loop_header {
    // 3. Call the protocol method
    %result = cool.call @Range_next(%range)
    
    // 4. Pattern match the option
    cool.match %result {
        Some(%x):
            cool.print(%x)
            cool.continue
        None:
            cool.break
    }
}

```

---

## Zero-Cost Performance through Monomorphization

Because `Range` is a concrete struct and `Iterator` is monomorphized, the "VTable" logic we discussed for dynamic dispatch is **not used** here.

1. The compiler sees `Iterator[Item=i64]`.
2. It replaces the generic protocol calls with direct calls to `Range_next`.
3. LLVM inlines `Range_next` directly into the loop body.
4. The final machine code is simply a register increment and a comparison—**identical to a C `for(int i=0; i<10; i++)` loop.**

---

## Iterator Ownership and Memory Safety

Iterators in Coolscript must respect the **No-Escape Rule**. There are two primary ways to iterate over a collection:

### 1. Iterating by View (`view Iterator`)

The iterator holds a `view` of the collection. It does not own the data, so the collection cannot be modified or moved while the iterator exists.

```python
let my_list = [1, 2, 3]
for x in my_list: # x is a view i32
    print(x)
# my_list is still valid here

```

### 2. Iterating by Move (`move Iterator`)

The iterator consumes the collection and yields ownership of each element.

```python
let my_list = ["Owned", "Strings"]
for s in my_list.consume(): # s is an owned str
    process(move s)
# my_list is burned here

```

---

## Milestone 2: Technical Summary

By combining **Generics**, **Associated Types**, and **Protocols**, you have built a system that is:

* **Safe**: Ownership is tracked through every iteration.
* **Ergonomic**: Users write high-level loops.
* **Fast**: No runtime overhead or dynamic dispatch for the majority of use cases.

This completes the technical architecture for the Coolscript Standard Library's core iteration engine.
