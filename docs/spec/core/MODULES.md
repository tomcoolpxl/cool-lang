# Coolscript Module System and Dependency Specification

This document details the organization, resolution, and management of code within the **Coolscript** ecosystem. Following the Go philosophy of decentralized management and the Python philosophy of clean syntax, Coolscript uses a single manifest file and path-based resolution to manage dependencies.

---

## Module Definition

A module is the highest level of code organization in Coolscript. It is defined as any directory containing a `cool.mod` file. A module contains one or more packages, which are subdirectories containing `.cool` source files.

### The cool.mod File

The `cool.mod` file uses a Pythonic, indentation-based syntax. It defines the module's identity, the minimum compiler version required, and its external dependencies.

**Example cool.mod:**

```python
module github.com/user/project
compiler 1.0.0

require:
    github.com/cool-lang/http v1.2.0
    github.com/cool-lang/json v0.5.4

exclude:
    github.com/bad-actor/broken-lib v1.0.1

replace:
    github.com/cool-lang/http => ../local_http_lib

```

### Key Manifest Keywords

| Keyword | Purpose |
| --- | --- |
| **module** | Defines the unique path/identity of the current module. |
| **compiler** | Specifies the minimum Coolscript compiler version required. |
| **require** | Lists external dependencies and their specific versions. |
| **exclude** | Prevents specific versions of a dependency from being used. |
| **replace** | Redirects a module path to a different location (usually a local path for development). |

---

## Package Structure and Visibility

Within a module, every directory is treated as a package. A package name is derived from its directory name.

### File Organization

```text
my_project/
    cool.mod
    main.cool           # Belongs to package 'main'
    net/
        client.cool     # Belongs to package 'net'
        server.cool     # Belongs to package 'net'
    utils/
        helpers.cool    # Belongs to package 'utils'

```

### Visibility Rules

Coolscript adopts a simple naming convention for visibility, similar to Python but enforced by the compiler.

* **Public Symbols**: Any function, struct, protocol, or variable starting with a letter (a-z, A-Z) is public and accessible to other packages that import it.
* **Private Symbols**: Any symbol starting with an underscore `_` is private to the package it is defined in.

**Example of visibility in `utils/helpers.cool`:**

```python
# Public function
fn ValidateInput(data: str) -> bool:
    return _check_length(data)

# Private function
fn _check_length(data: str) -> bool:
    return data.len() > 0

```

---

## Import Resolution

Imports are explicit and use the module path followed by the package path.

### Import Syntax

```python
import github.com/user/project/net
import github.com/user/project/utils as u

fn main():
    let client = net.NewClient()
    if u.ValidateInput("test"):
        client.Connect()

```

### Resolution Logic

1. **Standard Library**: The compiler first looks in the built-in `std` library.
2. **Local Packages**: If the path starts with the current module name (defined in `cool.mod`), it looks within the current project directory.
3. **External Dependencies**: The compiler looks in the local cache (usually `$COOLPATH/pkg/mod`). if not found, it fetches the source from the provided URL (e.g., GitHub).

---

## Dependency Management and Versioning

Coolscript uses **Semantic Versioning (SemVer)**. It does not allow conflicting versions of the same library in a single binary.

### Version Selection

If two dependencies require different versions of the same library, Coolscript employs the **Minimal Version Selection (MVS)** algorithm:

* If dependency A requires `lib v1.1.0` and dependency B requires `lib v1.2.0`, the compiler will select `v1.2.0` (the highest minimum version) as long as it is backwards compatible within the same major version.

### Checksums

To ensure security and reproducibility, the compiler generates a `cool.sum` file alongside `cool.mod`. This file contains cryptographic hashes of every dependency. If a dependency's source code changes on the remote server without a version bump, the compiler will error due to a checksum mismatch.

---

## Local Development and Overrides

The `replace` keyword in `cool.mod` is used to facilitate development across multiple local modules.

### Using the replace Keyword

When developing a library and an application simultaneously, you can point the application to the local source of the library instead of the remote URL.

```python
# Inside cool.mod
replace:
    github.com/org/library => ../local-library-dir

```

This tells the compiler to ignore the version in the `require` block and use the files in `../local-library-dir`. This is essential for debugging and feature development before pushing code to a remote repository.

---

## The Build Process

Coolscript prioritizes speed and static safety during the build process.

### Step 1: Parsing and Collection

The compiler parses `cool.mod` and retrieves all necessary source files. It verifies that all imported packages exist.

### Step 2: Type Checking and Ownership Analysis

The compiler performs static analysis across the entire dependency graph. It ensures that `move` and `view` rules are respected across package boundaries.

### Step 3: MLIR Generation and Optimization

Each package is lowered to MLIR. Cross-package optimizations (like inlining) are performed here.

### Step 4: Static Linking

The final stage uses a linker to combine all compiled packages into a single, standalone binary. This binary contains all necessary code to run, requiring no external `.so` or `.dll` files (except for system-level C libraries defined in `extern` blocks).

> **Note**: Because Coolscript produces static binaries, the deployment process is as simple as copying one file to the target server.

---
