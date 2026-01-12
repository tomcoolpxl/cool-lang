# Building Coolscript Applications

This guide explains how to build and run applications written in Coolscript using the `cool` CLI tool.

## Prerequisites

Ensure you have built the Coolscript compiler and runtime:

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
```

## The `cool` CLI

The project includes a wrapper script `bin/cool` that automates the compilation and linking process.

### Basic Usage

To build a Coolscript file into an executable:

```bash
./bin/cool build <file.cool> -o <output_binary>
```

### Example

1.  Create a file named `hello.cool`:

    ```python
    func main() -> i32:
        print(42)
        return 0
    ```

2.  Build it:

    ```bash
    ./bin/cool build hello.cool -o hello
    ```

3.  Run it:

    ```bash
    ./hello
    # Output: 42
    ```

## How it Works

The `cool build` command performs the following steps:

1.  **Compilation**: Invokes the internal compiler (`coolc`) to parse the `.cool` file and generate a native object file (`.o`) containing the machine code.
2.  **Linking**: Invokes `clang` to link the object file with the static runtime library (`libcool_runtime.a`) and system threads (`pthread`).
3.  **Cleanup**: Removes the intermediate object file.

## Troubleshooting

*   **Compiler not found**: Ensure you have run `cmake --build build` to generate the `coolc` executable and `libcool_runtime.a`.
*   **Linker errors**: If `clang` fails, check that `build/runtime/libcool_runtime.a` exists.
