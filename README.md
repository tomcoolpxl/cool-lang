# Coolscript: Python Aesthetics, C++ Performance, Go Simplicity, Rust Safety

Coolscript (`.cool`) is a statically typed, natively compiled, resource-oriented programming language. It is built for developers who want the **readability of Python** and the **concurrency of Go**, but require the **deterministic performance and memory safety of Rust** without the complexity of a borrow checker or the overhead of a Garbage Collector (GC).

---

## Key Features

* **Move-by-Default Ownership**: Memory is managed via linear types. The compiler tracks the lifecycle of every object, ensuring zero leaks and no use-after-free bugs.
* **Structured Concurrency**: No green threads or hidden schedulers. Coolscript uses **Isolates** and **Owned Channels** to prevent data races at compile-time.
* **Zero-Cost Safety**: No Garbage Collector, no "Stop the World" pauses, and no reference counting overhead.
* **Static Binaries**: Compiles directly to machine code (via MLIR/LLVM). Deployment is a single file.
* **Pythonic Syntax**: Significant indentation and colons for clean, readable code.

---

## At a Glance

```python
import std.fs
import std.net

fn worker(ch: move Channel[str]):
    if let msg = ch.receive():
        print("Received: " + msg)

fn main():
    let ch = Channel[str](capacity=10)
    
    # Ownership of the sender is moved into the task
    spawn worker(move ch.clone_sender())
    
    let message = "Hello from Coolscript"
    ch.send(move message)
    # 'message' is burned here; cannot be used again.

```

---

## Memory Model: The No-Escape Rule

Coolscript utilizes a **Move/View** duality. You either own a resource (`move`) or you have a temporary, read-only window into it (`view`).

To maintain simplicity, Coolscript enforces the **No-Escape Rule**: Views are bound to the stack. They cannot be stored in structs or globals. If you need data to persist, you must move it. This eliminates the need for complex lifetime annotations (`<'a>`).

---

## Getting Started

### Prerequisites

* LLVM 18+
* Clang (for linking the C runtime)
* Make

### Installation

```bash
git clone https://github.com/user/coolscript
cd coolscript
make build
export PATH=$PATH:$(pwd)/bin

```

### Your First Build

```bash
# Create a new module
cool mod init my_project

# Build a static binary
cool build main.cool -o my_app

# Run it
./my_app

```

---

## Project Status: Milestone 1

We are currently in the **Initial Implementation Phase**. The following components are defined:

1. **PEG Grammar**: Indentation-aware parser.
2. **MLIR Dialect**: Ownership and Linearity verification passes.
3. **C Runtime**: High-performance channel and task management.

---

## Contributing

We are looking for contributors interested in:

* **Compiler Engineering**: Implementing the MLIR Linear Type pass.
* **Language Design**: Refining the Standard Library protocols.
* **Tooling**: Building the `cool fmt` and LSP integration.

Please see the [Technical Manual](https://www.google.com/search?q=./TECHNICAL_MANUAL.md) for full architectural details.

---

## License

Coolscript is released under the MIT License.
