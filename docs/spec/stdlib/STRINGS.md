### Standard Library: String Specification (`std.str`)

In Coolscript, strings are **immutable byte sequences** (UTF-8 encoded). By making strings immutable and utilizing the `view` model, we can pass strings across the entire program without ever copying the underlying data, achieving C++ performance with a Pythonic interface.

---

## The String Structure

At the machine level, a `str` is a small descriptor.

```python
struct str:
    _ptr: UnsafePtr[u8] # Pointer to the raw bytes
    _len: i64           # Number of bytes
    _is_owned: bool     # Flag to determine if the bytes need deallocation

```

---

## String Ownership vs. Views

Because strings are immutable, the difference between an **Owned String** and a **View String** is simply about who is responsible for the memory.

### 1. The Owned String (`str`)

An owned string "holds" the memory. When an owned string is moved, the pointer is passed, and the original variable is burned. When the owner goes out of scope, the memory is immediately freed.

### 2. The View String (`view str`)

A view string is a "window" into an existing string. It does not own the memory and cannot be stored in a struct. This allows for zero-allocation slicing.

```python
fn main():
    let original = "Hello Coolscript" # Owned str
    
    # Slicing creates a view, not a copy
    let sub: view str = original[0..5] 
    
    print(sub) # Prints "Hello"
    # No memory was allocated for 'sub'; it points into 'original'

```

---

## The String Protocol and Methods

Strings implement the `Collection` and `Reader` protocols.

```python
protocol String:
    fn len(view self) -> i64
    fn chars(view self) -> Iterator[char]
    fn to_upper(view self) -> str # Returns a NEW owned string
    fn split(view self, sep: str) -> List[view str] # List of views

```

### Concatenation

Since strings are immutable, concatenation always produces a new owned string. To prevent "allocation storms" (like in Python), Coolscript encourages the use of `StringBuilder`.

```python
# Inefficient (creates 3 intermediate strings)
let s = "a" + "b" + "c"

# Efficient (single allocation)
let sb = StringBuilder.new()
sb.append("a")
sb.append("b")
sb.append("c")
let s = sb.build() # Returns owned str

```

---

## Memory Safety and Strings

The "No-Escape Rule" is vital here. If you have a `view str` pointing into a local variable, the compiler prevents you from returning that view or storing it in a global.

```python
fn get_prefix(full: view str) -> view str:
    return full[0..3] # OK: The view is derived from the input view

fn leak_attempt():
    let local = "Hidden"
    # let v: view str = local[0..2]
    # return v # COMPILER ERROR: View of 'local' escapes its scope

```

---

## UTF-8 Handling

Coolscript treats `str` as UTF-8 by default. Indexing a string returns a **view** of the bytes, but iterating over a string yields `char` (32-bit Unicode points).

```python
let s = "👋 Hello"

# Length in bytes
print(len(s)) # Prints 10 (4 for emoji + 6 for " Hello")

# Iterating over characters
for c in s:
    print(c) # Correctly prints the emoji as one unit

```

---

## C-Interop with Strings

Since C expects null-terminated strings (`char*`), Coolscript provides an explicit conversion. This is the only place where a "View" might be converted into a temporary "Owned" pointer to ensure a null terminator exists.

```python
import "C"

fn main():
    let s = "Cool"
    # s.c_str() provides a view to a null-terminated buffer
    C.printf("%s\n", s.c_str())

```

---

## Summary of String Performance

| Action | Coolscript | Python Equivalent |
| --- | --- | --- |
| **Slicing** (`s[0..5]`) | **O(1)** (View) | O(N) (New string) |
| **Passing to function** | **O(1)** (View/Move) | O(1) (Reference count) |
| **Iteration** | **O(N)** (No GC overhead) | O(N) (Heavy VM overhead) |
| **Deallocation** | **Deterministic** | Non-deterministic (GC) |

