# Coolscript Language Specification

**Coolscript** (`.cool`) is a statically typed, natively compiled, resource-oriented programming language. It is designed to provide the ergonomic experience of Python, the deployment simplicity and structural typing of Go, and the memory safety guarantees of Rust without the cognitive load of a borrow checker or the performance penalties of a garbage collector.

---

## Core Architectural Philosophy

Coolscript follows several strict design principles to achieve high performance and developer productivity.

* **Explicit Ownership**: Every resource has exactly one owner at any given time.
* **Zero-Cost Safety**: Memory safety is guaranteed at compile-time through a combination of linear types and region analysis.
* **No Garbage Collection**: There is no background runtime for memory management. Memory is reclaimed immediately when an owner is burned or goes out of scope.
* **No Global Virtual Machine**: Coolscript compiles directly to machine code via an **MLIR** (Multi-Level Intermediate Representation) backend.
* **Explicit Cost**: Significant operations like moving large data structures or copying memory are made visible at the call-site using specific keywords.
* **Static Linking**: All dependencies are compiled into a single, optimized machine code binary with no external runtime dependencies.

---

## Memory Management and Ownership

Coolscript replaces the traditional borrow checker with a model based on **Linear Types** and **Region Analysis**. This provides Rust-level safety with a significantly lower learning curve.

### The Move-by-Default Model

By default, passing a variable to a function or assigning it to another variable transfers ownership. This is known as a **move**. Once a variable is moved, it is considered **burned**. The compiler treats the original variable name as a tombstone. Any attempt to access a burned variable results in a compile-time error.

### Call-Site Keywords

To ensure readability and prevent "where did my variable go" confusion, moves and copies must be explicit at the call-site.

* **View (Default)**: If no keyword is used, the compiler attempts to create a read-only view. The original variable remains valid and owned by the caller.
* **inout**: The `inout` keyword allows a function to modify a variable owned by the caller. Semantically, this is "Copy-In, Copy-Out": the value is moved in, modified, and moved back. The compiler optimizes this to a pointer reference for large types or register manipulation for small types. While an `inout` access is active, the caller cannot access the original variable (Exclusivity Rule).
* **Move**: The `move` keyword explicitly transfers ownership. The caller can no longer use the variable.
* **Copy**: The `copy` keyword creates a duplicate of the value.
    *   If the type implements the **`Copy`** protocol (e.g., primitives like `i32`, `bool`), it performs a bitwise copy.
    *   If the type implements the **`Clone`** protocol (e.g., `List`, `str`), it invokes the `.clone()` method for a deep copy.
    *   If the type implements neither (e.g., a `FileHandle`), using `copy` is a **Compile-Time Error**. This prevents accidental resource duplication and double-free bugs.

**Example:**

```python
fn increment(val: inout i32):
    val += 1

fn process_user(u: move User):
    # This function now owns 'u'
    print(u.name)
    # 'u' is destroyed here

fn main():
    let count = 10
    increment(inout count) # count is now 11
    
    let admin = User(name="Alice", age=30)
    
    # Explicit move: readability tells us 'admin' is gone after this
    process_user(move admin)
```

---

## Views and the No-Escape Rule

A **View** is a temporary, read-only permission to access data. It is the Coolscript alternative to borrowing.

### Interior Mutability Exception
While views are strictly read-only for standard types (`i32`, `List`, `User`), certain synchronization primitives in the standard library (such as `Atomic[T]` and `Mutex[T]`) allow **Interior Mutability**. This enables thread-safe state updates through a shared view. This is achieved via verified low-level intrinsics and is not available for standard user-defined structs.

### Transient Structs (Option C)

Coolscript allows structs to contain `view` fields. However, any struct that contains a `view` is implicitly marked as **transient**.

**The Transience Rule:**
1.  A transient struct cannot outlive the data it views.
2.  A transient struct cannot be stored in a non-transient struct or a global variable.
3.  A transient struct cannot be moved into an Isolate (`spawn`).

This pattern is ideal for "Context Objects," "Iterators," or "Slices" that package multiple references to be passed down the stack during a single operation.

```python
struct StringCursor:
    buffer: view str
    pos: i32

fn process(c: view StringCursor):
    print(c.buffer[c.pos])

fn main():
    let s = "Hello"
    let cursor = StringCursor(buffer=view s, pos=0)
    process(view cursor)
```

### The No-Escape Rule

To keep the language simpler than Rust, Coolscript enforces the **No-Escape Rule**: Views and Transient Structs cannot be stored in long-lived heaps or globals. They are strictly bound to the stack frame.

If a piece of data needs to persist inside a permanent struct (e.g., adding a User to a List), it must be moved into the struct.

---

## Data Structures and Graphs (Option A)

Because Coolscript enforces a tree-like ownership model, creating cycles (like Parent-Child pointers or Graphs) using raw references is not allowed for permanent data.

**Idiomatic Graph Implementation:**
For long-lived graphs, Coolscript recommends the **Managed Index** pattern. Store nodes in a `List[Node]` and refer to them using `i32` indices. This is 100% memory-safe, has zero GC overhead, and avoids ownership cycles.

```python
struct Node:
    id: i32
    parent_index: i32 # Reference via Index

struct Forest:
    nodes: List[Node]
```

---

## Type System and Syntax

Coolscript uses a Python-like syntax with significant indentation and mandatory type annotations for function signatures.

### Variable Declaration and Inference

Local variables are declared with `let`. While the compiler can infer types, explicit annotations are allowed.

```python
let name = "Coolscript"      # Inferred as str
let version: f64 = 1.0       # Explicit f64
let is_active = True         # Inferred as bool

```

### Bracket Notation for Generics

Coolscript uses bracket notation for generic types, ensuring clarity when dealing with complex collections.

```python
let scores: List[i32] = [10, 20, 30]
let registry: Dict[str, User] = {}

```

### String Interpolation (F-Strings)

Coolscript supports Python-style string interpolation using the `f` prefix. This is syntactic sugar that the compiler lowers into efficient `StringBuilder` operations.

```python
let name = "Cool"
let version = 1.0
print(f"Welcome to {name} v{version}!")
```

### The Drop Protocol

Coolscript provides a deterministic way to clean up resources through the `Drop` protocol. 

```python
protocol Drop:
    """
    Called automatically by the compiler before an object is 
    deallocated or burned.
    """
    fn drop(inout self)
```

If a struct implements `Drop`, the compiler ensures that the `drop` method is invoked exactly once when the object's lifecycle ends (at the end of a block or when moved into a "sink").

**The No Resurrection Rule:**
To prevent "zombie objects," the `drop` method cannot move `self` or any of its fields to a new owner. While `inout self` allows mutation (e.g., zeroing memory or closing handles), the ownership of the resource must remain with the compiler-managed cleanup process. Attempting to `move self` or `move self.field` inside a `drop` implementation results in a **Compile-Time Error**.

---

## Anonymous Functions (Lambdas)

Coolscript supports inline anonymous functions for ergonomic functional programming (e.g., `map`, `filter`).

```python
let squared = list.map(|x| x * 2)
```

### Milestone 1: Non-Capturing
In the initial release, lambdas are treated as **pure function pointers**. They cannot capture variables from the surrounding scope. This limitation ensures zero allocation and prevents hidden lifetime issues during the compiler's bootstrapping phase.

### Future Vision: Full Closures
In future versions, Coolscript will introduce **Capturing Closures**.
*   **View Captures**: Will create a transient struct, valid only within the current stack frame (safe for `map`/`filter`).
*   **Move Captures**: Will take ownership of captured variables, allowing the closure to be passed to `spawn` (e.g., `move |x| ...`).

---

## Control Flow and Logic

Coolscript adopts a minimalist approach to control flow, favoring "one way to do it" and strict boolean logic.

### Conditionals: `if`, `elif`, `else`

Only boolean expressions are allowed in conditions. Coolscript does not support "truthy" or "falsy" values for integers, strings, or lists.

```python
let x = 10

if x > 5:
    print("Greater")
elif x == 5:
    print("Equal")
else:
    print("Lesser")

```

### Iteration: `while` and `for`

Coolscript supports `while` and `for` loops. It excludes `until` and `do-while` to keep the syntax lean.

* **While**: A standard condition-based loop.
* **For**: Iterates over collections. By default, it creates a **View** of each element.

```python
# Range-based loop (0 to 9) using 0..10 syntax
for i in 0..10:
    print(f"{i}")

# Collection loop (View-based)
let items = ["apple", "banana", "cherry"]
for item in items:
    # 'item' is a view. 'items' remains valid.
    print(item) 

# Consumption loop (Move-based)
for item in items.consume():
    # Each 'item' is now owned; ownership is moved out of the list.
    save_to_database(move item) 
# 'items' is empty and burned after this loop.

```

### Logical Operators

Coolscript uses word-based logical operators for maximum readability.

| Operator | Action |
| --- | --- |
| `and` | Logical AND |
| `or` | Logical OR |
| `not` | Logical Negation |
| `is` | Identity check (pointer equality) |
| `==` | Value equality |

### Pattern Matching: `match`

The `match` statement provides exhaustive pattern matching. The compiler ensures all possible cases are covered.

```python
enum Status:
    Pending
    Active
    Closed(reason: str)

fn check_status(s: view Status):
    match s:
        Status.Pending:
            print("Action is pending")
        Status.Active:
            print("Action is active")
        Status.Closed(reason):
            print(f"Closed because: {reason}")
        ```

---

## Structural Typing: Protocols

Coolscript does not support classes or classical inheritance. It uses **Structs** for data and **Protocols** for behavior (Static Duck Typing).

### Static Duck Typing

A struct satisfies a protocol if it implements the required methods. No explicit `implements` keyword is required.

```python
protocol Speaker:
    fn say_hello() -> str

struct Human:
    name: str

    fn say_hello() -> str:
        return "Hi, I am " + self.name

struct Robot:
    model: str

    fn say_hello() -> str:
        return "BEEP. MODEL " + self.model

fn greet(s: view Speaker):
    print(s.say_hello())

```

### Dynamic Dispatch and Fat Pointers

When different types are stored in a `List[Speaker]`, Coolscript uses **Fat Pointers**. A fat pointer contains:

1. A pointer to the instance data on the heap.
2. A pointer to a **Virtual Table (vtable)** that maps the protocol methods to the specific implementation for that type.

Heterogeneous lists automatically "box" elements on the heap. Because the list owns these boxes, it handles the destruction of different types correctly when the list itself is burned.

---

## Collections and Resource Management

Collections like `List` and `Dict` own their elements.

* **Adding an element**: Requires the `move` keyword. The element is now owned by the collection.
* **Indexing**: Returns a **view**. You can read the data, but you cannot move it out or destroy it through the indexer.
* **Popping**: Returns an **owned** object. The element is removed from the collection and ownership is returned to the caller.

```python
let my_list: List[User] = []
let u = User(name="Dev")

my_list.add(move u) # 'u' is burned locally

# Read access via view
let first = my_list[0]
print(first.name)

# Take ownership back
let recovered_user = my_list.pop(0)
# 'recovered_user' is now owned by the local scope again

```

---

## Concurrency: Tasks and Channels

Coolscript implements **Structured Concurrency** to avoid the "hidden" costs of Goroutines while maintaining high efficiency.

### Tasks (Isolates)

Tasks are spawned using the `spawn` keyword. Unlike Goroutines, Tasks are **isolated**. They do not share memory with the parent thread. To pass data to a task, you must `move` it or `copy` it. This prevents data races by design. Coolscript tasks use fixed-size stacks determined at compile-time, avoiding Go's stack-check overhead.

```python
fn background_task(data: move LargeBuffer):
    # This runs in a separate thread pool
    # It has exclusive ownership of 'data'
    pass

fn main():
    let buf = load_buffer()
    spawn background_task(move buf)
    # 'buf' cannot be accessed here, so no race condition is possible

```

### Channels



Channels facilitate communication between tasks. Sending a value through a channel moves the ownership from the sender to the receiver.



```python

fn producer(ch: move Channel[i32]):

    let val = 42

    ch.send(move val)



fn consumer(ch: view Channel[i32]):

    # .receive() returns an opt[T]

    if let val = ch.receive():

        print(f"Received: {val}")
```



fn main():

    let ch = Channel[i32](capacity=5)

    spawn producer(move ch.clone_sender())

    consumer(view ch)

```



### Shared Ownership (shared[T])



To efficiently share large, read-only data (like configuration or game assets) across hundreds of tasks without copying, Coolscript provides the `shared[T]` wrapper.



*   **Atomic Reference Counting**: `shared[T]` uses an atomic counter. Cloning the handle is cheap (increment integer).

*   **Immutable**: Data inside a `shared` handle is strictly read-only.

*   **Spawn-Safe**: Unlike standard views, `shared` handles can be safely moved or copied into `spawn` tasks.



```python

struct Config:

    theme: str



fn worker(cfg: move shared[Config]):

    # Read-only access to underlying data

    print(cfg.theme)



fn main():

    # Wrap the data. 'shared' takes ownership.

    let config = shared(Config(theme="Dark"))



    for i in 0..10:

        # 'copy' creates a new handle (increments ref count)

        spawn worker(copy config)

    

    # original 'config' is burned here when scope ends

```

---

## Error Handling



Coolscript avoids exceptions. Instead, it uses a `Result[T, E]` type and a `try` keyword for ergonomic error propagation.



### The `try` Block



Used when you want to handle the error locally.



```python

fn open_file(path: str) -> Result[File, IOError]:

    # Returns the File (Ok) or an IOError (Err)

    pass



fn main():

        # The 'try' block triggers on the 'Err' case

        let f = open_file("data.txt") try (err):

            print(f"Error: Could not open file: {err.message}")

            return

    # 'f' is now a valid File object if 'try' didn't trigger the block

```



### The Propagation Operator (`?`)



Used when you want to bubble the error up to the caller. This is syntactic sugar for `try (e): return Err(e)`.



```python

fn read_config() -> Result[str, IOError]:

    # If open_file fails, the error is immediately returned from read_config

    let f = open_file("config.txt")?

    return Ok("Success")

```

---

## C-Interop



Support for C is a first-class citizen. Using `import "C"` allows direct access to C functions without wrappers. However, because C functions cannot guarantee Coolscript's memory safety, calling them requires an `unsafe` block.



### Unsafe Pointers (`UnsafePtr[T]`)



For implementing low-level primitives (like `List`, `str`, or custom allocators), Coolscript provides `UnsafePtr[T]`. This is a raw pointer that is **not** tracked by the ownership system. It can be null, dangling, or aliased. Dereferencing it is only allowed inside an `unsafe` block.



```python

import "C"



extern "libc.so.6":

    fn printf(format: str, ...) -> i32

    fn malloc(size: i64) -> UnsafePtr[u8]

    fn floor(x: f64) -> f64



fn main():

    unsafe:

        let result = C.floor(10.5)

        C.printf("Result: %f\n", result)

```

---

## Program Entry Point

Every executable Coolscript program must have a `main` function.

*   **Signature**: `fn main() -> Result[(), Error]` or `fn main()`.
*   **Error Handling**: If `main` returns a `Result`, the runtime will automatically print the error message and exit with a non-zero status code if an `Err` occurs.

```python
fn main() -> Result[(), IOError]:
    let content = fs.read_file("config.txt")?
    print(content)
    return Ok(())
```

---

## Project Structure and Compilation

* **File Extension**: `.cool`
* **Project Manifest**: `cool.mod` (defines project metadata and decentralized dependencies).
* **Dependency Management**: Go-style imports (URLs or local paths). No centralized package manager like pip.
* **Compilation**: All code is statically linked into a single binary.
* **Tooling**: The compiler uses a **PEG** (Parsing Expression Grammar) for fast, predictable parsing of Python-style indentation.

---
