# Coolscript Standard Library Specification

The Coolscript Standard Library (`std`) is built on the principles of **Ownership**, **Null Safety**, and **Result-based Error Handling**. It is designed to be lean, providing only the essential primitives needed for high-performance systems programming while delegating complex higher-level logic to the decentralized module system.

---

## Core Principles

* **Resource Ownership**: Objects representing external resources (Files, Sockets, Database Handles) are owned by the scope that creates them. They are automatically closed when burned or when they go out of scope.
* **No Exceptions**: All I/O and system operations return a `Result[T, E]`.
* **Thread Safety**: Standard library types that handle shared state (like Channels) enforce move semantics to prevent data races.
* **Zero-Allocation Defaults**: Where possible, the library provides APIs that work with `view` buffers to avoid unnecessary heap allocations.

---

## I/O Abstractions: The Stream Protocols

Coolscript abstracts input and output through two primary protocols, allowing functions to work interchangeably with files, network sockets, or memory buffers.

```python
protocol Reader:
    # Reads data into the provided buffer. 
    # Returns the number of bytes read or an IOError.
    fn read(buffer: inout List[u8]) -> Result[i64, IOError]

protocol Writer:
    # Writes data from the provided view buffer. 
    # Returns the number of bytes written or an IOError.
    fn write(data: view List[u8]) -> Result[i64, IOError]

```

---

## Filesystem (`std.fs`)

The `File` struct is the primary interface for filesystem operations.

### File Struct Definition

```python
struct File:
    _handle: i32
    path: str
    is_open: bool

    # Static constructor
    fn open(path: str, mode: str) -> Result[File, IOError]:
        # Implementation maps to C.open
        pass

```

### File Methods

* **read_all(inout self) -> Result[str, IOError]**: Reads the entire file into an immutable string.
* **write_str(inout self, data: str) -> Result[i64, IOError]**: Writes a string to the file.
* **close(move self)**: Explicitly closes the file handle and burns the resource.

---

## Networking (`std.net`)

Coolscript networking is built around **Structured Concurrency**. Sockets must be moved into background tasks to ensure exclusive access.

### TCP Socket Operations

```python
struct TcpStream:
    _fd: i32
    remote_addr: str

    fn dial(address: str) -> Result[TcpStream, NetError]:
        pass

    fn send(inout self, data: view List[u8]) -> Result[i64, NetError]:
        pass

    fn receive(inout self, buffer: inout List[u8]) -> Result[i64, NetError]:
        pass

struct TcpListener:
    _fd: i32

    fn listen(address: str) -> Result[TcpListener, NetError]:
        pass

    fn accept(inout self) -> Result[TcpStream, NetError]:
        pass

```

---

## Concurrency Primitives (`std.sync`)

While `spawn` and `Channel` are language-level keywords/types, the standard library provides the underlying implementations.

### Channel[T]

Channels are the primary synchronization primitive.

* **send(view self, item: move T)**: Ownership of the item is moved into the channel's internal buffer.
* **receive(view self) -> opt[T]**: If data is available, ownership is moved out and returned as `Some(T)`.

### Atomic[T]

For the rare cases where true shared state is required (e.g., a global configuration), `Atomic` provides thread-safe access to small, copyable types.

```python
let counter = Atomic[i32](0)
counter.add(1) # Thread-safe increment

```

---

## Basic Collections (`std.collections`)

Coolscript's built-in collections are optimized for the move-by-default model.

### List[T]

A dynamic array that owns its elements.

* **add(inout self, item: move T)**: Consumes the item.
* **pop(inout self, index: i64) -> opt[T]**: Removes an item and returns ownership to the caller.
* **at(view self, index: i64) -> opt[view T]**: Returns a temporary read-only view.

### Dict[K, V]

A hash map implementation.

* **insert(inout self, key: move K, value: move V)**: Both key and value ownership are transferred to the map.
* **get(view self, key: view K) -> opt[view V]**: Provides a view of the value associated with the key.

---

## Core Wrapper Methods

The `opt[T]` and `Result[T, E]` types provide helper methods for ergonomic unwrapping.

* **unwrap(move self) -> T**: Returns the value or panics if None/Err.
* **expect(move self, msg: str) -> T**: Returns the value or panics with the provided message if None/Err.
* **is_some(view self) -> bool**: Returns true if opt contains a value.
* **is_ok(view self) -> bool**: Returns true if Result is Ok.

---

## Built-in Functions and Constants

The standard library includes a "prelude" that is imported into every `.cool` file automatically:

* **print(data: view Any)**: Outputs a representation of the data to stdout.
* **str(data: view Any) -> str**: Converts basic types to their string representation.
* **len(data: view Collection) -> i64**: Returns the size of a list, dict, or string.
* **panic(msg: str)**: Terminates the program immediately (used only for unrecoverable logic errors).

