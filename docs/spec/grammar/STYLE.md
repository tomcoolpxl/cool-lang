# Coolscript Style and Formatting Specification

This document defines the official formatting rules and indentation standards for **Coolscript** (`.cool`). These rules are strictly enforced by the `cool fmt` tool to ensure a unified, "one-way-to-do-it" aesthetic across the entire ecosystem, similar to `gofmt` but with Pythonic readability.

---

## Indentation and Whitespace

Coolscript uses **Significant Indentation** to define block scope. This eliminates the need for curly braces `{}` or `begin/end` keywords.

### Indentation Rules

* **Strictly 4 Spaces**: The compiler and `cool fmt` will reject or convert tabs to spaces. Mixing tabs and spaces results in a compile-time error.
* **The Colon (`:` )**: All blocks (functions, conditionals, loops, protocols, and structs) must be preceded by a colon.
* **Blank Lines**:
* Two blank lines between top-level declarations (structs, protocols, functions).
* One blank line between methods inside a struct or protocol.
* No blank lines at the beginning or end of a block.



```python
struct User:
    name: str
    age: i32


fn main():
    let u = User(name="Dev", age=25)
    
    if u.age > 18:
        print("Authorized")

```

---

## Naming Conventions

Coolscript enforces a hybrid naming convention to distinguish between types and instances clearly.

| Entity | Style | Example |
| --- | --- | --- |
| **Structs / Protocols / Enums** | PascalCase | `TcpStream`, `Logger` |
| **Functions / Methods** | snake_case | `calculate_total`, `read_all` |
| **Variables / Parameters** | snake_case | `user_id`, `buffer_size` |
| **Constants** | SCREAMING_SNAKE_CASE | `MAX_RETRY_COUNT` |
| **Private Symbols** | Leading Underscore | `_internal_cache` |

---

## Function Formatting

Function signatures should stay on one line if possible. If a signature exceeds **80 characters**, parameters must be broken into multiple lines, indented once.

### Standard Function

```python
fn process_and_validate_data(id: i32, payload: view List[u8], retry: bool) -> Result[bool, str]:
    pass

```

### Wrapped Function

```python
fn long_function_name(
    user_id: i32,
    session_token: str,
    payload: move LargeDataStructure
) -> Result[bool, NetworkError]:
    pass

```

---

## Control Flow Formatting

### Conditionals

Parentheses are never used around the condition unless required for operator precedence.

```python
# Correct
if x > 10 and not is_valid:
    do_something()

# Incorrect
if (x > 10):
    do_something()

```

### Loops

The `0..10` range operator and the `consume()` method are formatted without spaces around the dots to indicate tight binding.

```python
for i in 0..10:
    pass

for item in collection.consume():
    pass

```

---

## The `cool fmt` Tool

The `cool fmt` command is the "Source of Truth" for formatting. It does not take configuration files; it enforces these rules globally to prevent "style wars."

### Automatic Corrections

1. **Alignment**: It aligns struct fields for better readability.
2. **Move/View Visibility**: It ensures exactly one space exists before the `move` and `copy` keywords in function calls.
3. **Import Sorting**: Imports are sorted alphabetically, grouped by Standard Library first, then Local Packages, then External URLs.

---

## Comments and Documentation

### Single-line Comments

Use the `#` symbol. A single space must follow the `#`.

```python
# This is a valid comment
let x = 10 # This is an inline comment

```

### Docstrings

Documentation for public symbols is placed inside a triple-quoted string immediately following the definition line. `cool doc` uses these to generate project documentation.

```python
protocol Reader:
    """
    Reader defines the standard interface for objects that
    can read raw bytes into a provided buffer.
    """
    fn read(buffer: view List[u8]) -> Result[i64, IOError]

```
