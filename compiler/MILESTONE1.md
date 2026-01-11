This **Milestone 1 Release Checklist** defines the objective benchmarks and performance requirements the **Coolscript** compiler and runtime must achieve. Since the goal is to compete with C++ and Go, these metrics ensure that the "Zero-Cost Safety" and "Structured Concurrency" claims are backed by data.

---

## 1. Compiler Performance Benchmarks

The compiler itself must be fast to maintain a productive developer loop.

* **Null Build Speed**: Running `cool build` on a project with no changes must return in **< 100ms** (checking file hashes only).
* **Cold Build Throughput**: The compiler must process at least **50,000 lines of code per second** on a modern quad-core machine.
* **Memory Usage**: The compiler must not consume more than **512MB of RAM** when compiling the "Gold Standard" program.

---

## 2. Runtime and Memory Efficiency

Coolscript must prove it handles memory as efficiently as C++.

* **Zero Leak Verification**: 100% of tests must pass `valgrind --leak-check=full` with "0 bytes in use at exit."
* **Peak Memory Overhead**: The overhead of the runtime (excluding user data) must be **< 2MB** for a basic static binary.
* **Allocation Latency**: `cs_alloc` and `cs_free` calls should have no more than **5% overhead** compared to standard `libc` malloc/free.

---

## 3. Concurrency and Latency

Benchmarks for the `spawn` and `Channel` primitives compared to Go and C++ threads.

| Metric | Target Requirement | Context |
| --- | --- | --- |
| **Task Creation** | < 500 nanoseconds | Time to `spawn` a minimal isolate. |
| **Channel Throughput** | > 10M msgs/sec | Moving 8-byte pointers through a buffered channel. |
| **Context Switch** | < 2 microseconds | Time for the thread pool to switch tasks. |
| **Max Isolates** | > 100,000 | Number of suspended tasks before memory exhaustion. |

---

## 4. Safety and Correctness Tests

The "Golden Suite" that must pass before the compiler is considered stable.

* **The Burn Test**: Successfully reject 50 unique variations of "Use-After-Move" bugs.
* **The Escape Test**: Successfully reject 20 unique variations of "Returning a view of a local variable."
* **The Isolate Test**: Successfully reject any attempt to pass a `view` or a non-sendable type into `spawn`.
* **FFI Safety**: Verify that `import "C"` calls do not corrupt the Coolscript stack via buffer overflows (tested with AddressSanitizer).

---

## 5. Deployment and Portability

The "Go-style" deployment guarantee.

* **Static Analysis**: `ldd` (on Linux) or `Dependency Walker` (on Windows) must show **zero** non-system library dependencies.
* **Binary Size**: A "Hello World" binary must be **under 1.5MB** (after stripping symbols).
* **Cross-Compilation**: The compiler must be able to target `linux-amd64` from a `macos-arm64` host using the `--target` flag.

---

## Final Verification Steps

1. **Compile the Compiler**: The ultimate test is when the Coolscript compiler can compile its own source code (self-hosting).
2. **Gold Standard Run**: Execute the `log_processor` example on a 1GB log file. It must complete within 10% of a hand-written C++ implementation.
3. **Documentation Audit**: Ensure all `view` and `move` behaviors match the PEG Grammar and Specification documents exactly.

---

This checklist completes the roadmap for Milestone 1. You have the specification, the grammar, the runtime implementation, and the performance targets.
