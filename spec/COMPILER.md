## Coolscript Compiler CLI and Tooling Specification

The Coolscript toolchain is a unified binary (`cool`) that handles everything from dependency management to optimized compilation. It follows the Go philosophy of having a small set of powerful, orthogonal commands.

---

## Core Commands

| Command | Action |
| --- | --- |
| **cool build** | Compiles the current module and its dependencies into a static binary. |
| **cool run** | Compiles and immediately executes the main package. |
| **cool get** | Fetches an external dependency and adds it to `cool.mod`. |
| **cool tidy** | Removes unused dependencies from `cool.mod` and updates `cool.sum`. |
| **cool check** | Performs static analysis and ownership verification without compiling. |
| **cool fmt** | Re-indents and formats code according to the official style guide. |

---

## Dependency Management via CLI

Coolscript manages dependencies through a local cache, typically located at `$COOLPATH/pkg/mod`. This prevents redundant downloads and allows for offline builds.

### Fetching Dependencies: `cool get`

When you run `cool get <url>`, the CLI performs the following steps:

1. **Discovery**: Resolves the URL (e.g., GitHub, GitLab) and determines the version (defaults to `@latest` or the newest SemVer tag).
2. **Download**: Clones the source into the local cache.
3. **Validation**: Computes a cryptographic hash of the source code.
4. **Manifest Update**: Adds the dependency to the `require` block in `cool.mod`.
5. **Checksum Update**: Appends the hash to `cool.sum`.

```bash
# Fetch a specific version
cool get github.com/cool-lang/http@v1.2.0

```

### Cleaning the Manifest: `cool tidy`

Over time, `cool.mod` may contain dependencies that are no longer imported in the source code. `cool tidy` scans the entire module, identifies imported packages, and prunes any `require` entries that are not strictly necessary. It also ensures that the `cool.sum` file matches the current state of the code.

---

## The Build Pipeline: `cool build`

The build command is the primary entry point for creating production-ready binaries.

### 1. Verification Phase

Before any code is generated, the compiler runs a full **Ownership Pass**.

* It checks for any use of "burned" variables.
* It ensures no `view` escapes its stack frame.
* It verifies that every `spawn` call correctly moves ownership to the background task.

### 2. Incremental Compilation

Coolscript uses a "Package Hash" system. The compiler hashes the contents of each package. If a package and its dependencies have not changed since the last build, the compiler reuses the cached object file. This ensures that even large projects with hundreds of dependencies compile in seconds.

### 3. Static Linking and Binary Stripping

By default, `cool build` produces a statically linked binary. It also performs **Dead Code Elimination** (Tree Shaking), removing any function, struct, or protocol implementation that is not reachable from the `main` function. This results in very small binaries, often only a few megabytes in size.

---

## Reproducible Builds and `cool.sum`

The `cool.sum` file is critical for security and reproducibility. It acts as a lockfile, ensuring that every developer on a team (and the CI/CD server) uses the exact same source code for dependencies.

**Example cool.sum:**

```text
github.com/cool-lang/http v1.2.0 h1:8pY...hash...
github.com/cool-lang/json v0.5.4 h1:2qX...hash...

```

If a developer attempts to build the project and a dependency in the local cache has a different hash than what is recorded in `cool.sum`, the compiler will abort with a **Security Violation Error**. This prevents "supply chain attacks" where a dependency's code is changed silently on a remote server.

---

## Environment Variables

Coolscript behavior can be tuned via a few key environment variables:

* **COOLPATH**: The directory where the compiler stores the dependency cache and downloaded packages.
* **COOLBIN**: The destination for binaries installed via `cool install`.
* **COOLCACHE**: Directs the compiler to a specific location for build artifacts.

---

## Development Tooling

### Language Server (CoolLS)

Coolscript comes with a built-in Language Server (LSP) that provides:

* **Burn-site Highlighting**: Visually dims variables that have been moved/burned.
* **View Tracking**: Shows the scope and validity of a reference.
* **Protocol Auto-complete**: Suggests missing methods when a struct is intended to satisfy a protocol.
