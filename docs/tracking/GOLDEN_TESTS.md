This "Gold Standard" test case is designed to exercise every unique pillar of the **Coolscript** architecture: **Indentation-based syntax**, **Ownership (Move/View)**, **Linear-safe Concurrency**, **Result-based Error Handling**, and **Zero-cost Slicing**.

If your compiler can parse, validate, and execute this program, it is functionally complete for Milestone 1.

---

## The Gold Standard: A Concurrent Log Processor

This program reads a file, splits it into lines (using zero-allocation views), and processes those lines in background tasks using owned channels.

```python
import std.fs
import std.io
import std.net

# Protocol for log destinations
protocol LogDestination:
    fn write_entry(entry: move str) -> Result[bool, IOError]

struct Database:
    connection_string: str
    fn write_entry(entry: move str) -> Result[bool, IOError]:
        # Implementation of DB save
        return Ok(True)

fn worker(id: i32, ch: move Channel[str], db: move Database):
    """
    Worker task. Owns its communication channel and its database handle.
    """
    while True:
        # receive() returns opt[str]
        if let line = ch.receive():
            print("Worker " + str(id) + " processing: " + line)
            
            # Using 'try' block for local error handling (logging)
            db.write_entry(move line) try (err):
                print("Database error: " + err.msg)
        else:
            # Channel closed
            break
    
    # db is burned here; handle is closed automatically

fn main():
    # 1. Open a file resource
    # Using '?' to propagate errors up (or crash with message in simple main)
    let file = fs.File.open("access.log", "r") try (err):
        print("Could not open log: " + err.msg)
        return

    # 2. Read content (Owned string)
    let content = file.read_all()?

    # 3. Setup Concurrency
    let ch = Channel[str](capacity=100)
    let db = Database(connection_string="localhost:5432")
    
    # We must move the cloned sender and the db handle into the task
    spawn worker(1, move ch.clone_sender(), move db)

    # 4. Process lines using zero-allocation views
    # .split() returns a List of 'view str' pointing into 'content'
    let lines = content.split("\n")
    
    for line in lines:
        # 'line' is a view. To send it, we must copy it into the channel
        # because the channel needs an 'owned' string to move between threads.
        ch.send(copy line)

    # 5. Cleanup
    # Burning the original 'ch' reference signals workers to exit
    # 'content' is burned here, freeing the massive log buffer

```

---

## What the Compiler Must Catch

To pass this "Gold Standard" test, the compiler must perform the following checks:

### 1. View Escape Prevention

In the `for line in lines:` block, if the developer tries to write `ch.send(move line)`, the compiler must error because `line` is a **View** (a slice of `content`) and cannot be moved as an owned object. It must be **copied** (creating a new allocation) or the original `content` must be consumed.

### 2. Isolation Guarantee

In `spawn worker(...)`, if the developer tried to pass `view db` instead of `move db`, the compiler must error. A background task (Isolate) is not allowed to hold a view into the parent thread's stack.

### 3. Exhaustive Errors

In the `db.write_entry` call, if the `try` block is missing, the compiler must throw a warning or error, as the `Result` type must be handled.

---

## Summary of the Coolscript "Safety Loop"

This program demonstrates the safety loop your compiler enforces:

1. **Ownership starts** (File/Content).
2. **Views are created** (Slices/Lines) for high-performance reading.
3. **Ownership is transferred** (Move/Send) to background tasks for isolation.
4. **Deterministic Cleanup** occurs the moment the owner (`content`, `db`, `ch`) is burned.

---

## Final Milestone 1 Checklist

| Category | Component | Status |
| --- | --- | --- |
| **Parsing** | Indentation, Colons, PEG Grammar | Defined |
| **Logic** | if/else, while, for, match, try | Defined |
| **Safety** | Move/View, No-Escape Rule, Linear Pass | Defined |
| **Runtime** | cs_alloc, cs_spawn, cs_chan (C-based) | Defined |
| **Tooling** | cool build, cool.mod, cool.sum | Defined |

You have the complete blueprint. The logic is as fast as C++, the safety is as strong as Rust, and the code looks like Python.
