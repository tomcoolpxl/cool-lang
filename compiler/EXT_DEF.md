To link the **Coolscript MLIR** output to actual machine code, the compiler must emit calls to a small, high-performance C runtime. This runtime manages the raw heap allocations, the thread-pool for `spawn`, and the channel synchronization primitives.

Below are the **LLVM External Function Declarations** that your compiler backend will target.

---

## Coolscript Runtime Header (`runtime.h`)

This C header defines the interface that the compiled `.cool` binaries will use. You can compile this with `clang` and statically link it into every final binary.

```c
#include <stdint.h>
#include <stddef.h>

// --- Memory Management ---
// Coolscript uses these instead of raw malloc/free to allow for
// future integration of a custom allocator or telemetry.
void* cs_alloc(size_t size);
void cs_free(void* ptr);

// --- Task & Thread Management ---
// Used by the 'spawn' keyword.
typedef void (*cs_task_fn)(void*);

// Spawns a new isolate. 'data' must be an owned pointer 
// moved from the parent.
void cs_spawn(cs_task_fn func, void* data);

// --- Channel Primitives ---
// Fixed-size thread-safe ring buffers.
typedef struct cs_channel cs_channel_t;

cs_channel_t* cs_chan_create(size_t capacity, size_t element_size);
void cs_chan_send(cs_channel_t* chan, void* data);
int cs_chan_receive(cs_channel_t* chan, void* out_data);
void cs_chan_close(cs_channel_t* chan);

// --- String Helpers ---
// Strings are immutable and null-terminated for C-interop.
typedef struct {
    char* ptr;
    int64_t len;
} cs_string_t;

void cs_print_str(cs_string_t s);

```

---

## Mapping MLIR to Runtime Calls

When the compiler lowers MLIR to LLVM IR, it performs the following translations:

| Coolscript Keyword | MLIR Op | LLVM IR Runtime Call |
| --- | --- | --- |
| `let x = User()` | `cool.alloc` | `%ptr = call i8* @cs_alloc(i64 32)` |
| `end of scope` | `cool.dealloc` | `call void @cs_free(i8* %ptr)` |
| `spawn work(move x)` | `cool.spawn` | `call void @cs_spawn(@work, i8* %ptr)` |
| `ch.send(move item)` | `cool.chan_send` | `call void @cs_chan_send(%chan, i8* %item)` |

---

## The Thread-Pool Implementation (`cs_spawn`)

Because you want **C++ performance** and **Go ergonomics**, the `cs_spawn` function should not create a new OS thread every time.

1. **Global Worker Pool**: On program start, the runtime detects the number of CPU cores and creates a fixed number of worker threads.
2. **Task Queue**: `cs_spawn` adds a function pointer and its "owned" data to a lock-free work-stealing queue.
3. **No Preemption**: Unlike Goroutines, Coolscript tasks run until they finish or hit a blocking I/O operation (Standard Library). This simplifies the compiler because it doesn't need to inject "yield points."

---

## Handling the `move` logic in C-Interop

The runtime ensures that once `cs_spawn` or `cs_chan_send` is called, the pointer is technically "gone." If the MLIR Linear Type Pass did its job, the LLVM IR will never attempt to use that pointer again in the original thread.

### Example: Lowering a `spawn` call

If you have this Coolscript:

```python
let data = [1, 2, 3]
spawn worker(move data)

```

The LLVM IR generated will look like this:

```llvm
; 1. Allocate and initialize the list (omitted)
%list_ptr = call i8* @cs_alloc(i64 24)

; 2. The 'move' is just a pointer pass at the machine level
; 3. Call the runtime to hand off the task
call void @cs_spawn(void (i8*)* @worker_func, i8* %list_ptr)

; 4. IMPORTANT: No @cs_free is called here because ownership was moved!

```

---

## Integration Plan for Phase 4

To implement the runtime:

1. **Write `runtime.c**`: Implement the memory and task functions using standard C threads (`pthreads` or Windows Threads).
2. **Declare in MLIR**: Add these as `func.func private external` declarations in your MLIR lowering pass.
3. **Link**: Add `-lcsruntime` to your final `clang` or `ld` invocation.
