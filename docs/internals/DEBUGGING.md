This **Debugging and Inspection Guide** explains how to use standard system tools to verify the memory safety and concurrency behavior of Coolscript binaries. Because Coolscript compiles to native machine code and links against a C runtime, you can use `gdb` or `lldb` to step through owned pointer transitions and inspect channel buffers.

---

## Debugging Coolscript with GDB/LLDB

Since Coolscript uses an MLIR/LLVM backend, the compiler can emit standard DWARF debug symbols. When you build with debug flags (`cool build -g`), the mapping between Coolscript source lines and the generated C-runtime calls is preserved.

### 1. Inspecting Owned Pointers

To verify that a variable is correctly "burned" or "freed," you can set a breakpoint on the runtime deallocator.

```bash
# Start the debugger
gdb ./log_processor

# Break at the point where Coolscript returns memory to the system
break cs_free

# Run the program
run

```

When the breakpoint hits, use `backtrace` to see which Coolscript variable just went out of scope.

### 2. Inspecting Channel State

Channels are structs in the C runtime. You can inspect the number of items currently waiting in a channel to diagnose deadlocks or performance bottlenecks.

```gdb
# Set a breakpoint in the worker task
break worker

# Inspect the channel structure
# (Assuming 'ch' is the variable name in the C runtime)
print *ch

```

You will see the internal state:

* `chan->size`: How many items are currently buffered.
* `chan->capacity`: The maximum limit before `send` blocks.
* `chan->closed`: Whether the channel has been shut down by the owner.

---

## Verifying Memory Safety with Valgrind

Since Coolscript guarantees no leaks at compile-time, **Valgrind** should always return a "clean" report. If it doesn't, there is a bug in the compiler's `cool-auto-destruction` pass.

```bash
valgrind --leak-check=full ./log_processor

```

**What to look for:**

* **Zero Leaks**: All `cs_alloc` calls must be matched by `cs_free` calls injected by the compiler.
* **No Invalid Reads**: This confirms that the "Burned" logic is working; the program never touches memory after it has been moved or freed.

---

## Visualizing Task Execution

Because Coolscript uses structured concurrency (Isolates), you can use system tracing tools to see how tasks are distributed across CPU cores.

### Using `htop` or `top`

While your program is running, press `H` in `htop` to see individual threads. You should see a number of threads equal to your core count, all executing tasks from the global work-stealing queue.

### Using `strace` (Linux)

To see how your program interacts with the OS (opening files, sending network packets), use `strace`:

```bash
strace -e trace=open,read,write,connect ./log_processor

```

This will show the underlying `cs_file_open` and `cs_chan_send` operations as they hit the kernel.

---

## Troubleshooting Common Milestone 1 Issues

| Symptom | Probable Cause | Fix |
| --- | --- | --- |
| **Segfault on `receive**` | Runtime returned `NULL` (None) but code tried to access it. | Check your `if let` unwrapping logic in the compiler. |
| **Deadlock** | All workers are waiting on an empty channel, and the owner didn't close it. | Ensure the `cs_free` pass correctly burns the channel owner at the end of `main`. |
| **Memory Leak** | A `move` occurred, but the compiler still injected a `free` in the parent. | Fix the Linear Type Pass; it should only inject `free` for variables in the `Owned` state. |

---

## Summary of the Development Environment

1. **Compiler**: `cool build` (The driver we designed).
2. **Runtime**: `runtime.c` (The C bridge for channels/tasks).
3. **Debugger**: `gdb` (To watch owned pointers).
4. **Verification**: `valgrind` (To prove the "Zero-Leak" claim).

With this guide, you have the full "Developer Loop" required to implement and verify the Coolscript compiler.
