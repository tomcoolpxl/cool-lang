To complete the linking process for the **Gold Standard** program, the compiler needs a concrete C implementation of the `std.io` and `std.fs` modules. These functions bridge the high-level Coolscript `str` and `File` types to the underlying OS system calls.

Below is the implementation of `std_io.c` and `std_fs.c`.

---

## Standard I/O Implementation (`std_io.c`)

This handles the basic `print` and `str` conversions. It uses the `cs_string_t` struct defined in the runtime.

```c
#include <stdio.h>
#include <stdlib.h>
#include "runtime.h"

// The internal representation of a Coolscript string
typedef struct {
    char* ptr;
    int64_t len;
} cs_string_t;

// Standard print function
void std_io_print(cs_string_t s) {
    // We use fwrite to handle potential null bytes or non-terminated views
    fwrite(s.ptr, 1, s.len, stdout);
    putchar('\n');
}

// Convert i32 to an owned string
cs_string_t std_io_int_to_str(int32_t val) {
    char* buffer = (char*)cs_alloc(12); // Max i32 is 10 digits + sign
    int len = sprintf(buffer, "%d", val);
    
    cs_string_t result = { .ptr = buffer, .len = (int64_t)len };
    return result;
}

```

---

## Filesystem Implementation (`std_fs.c`)

This implementation enforces the **Result** pattern by returning a struct that contains either the resource or an error code.

```c
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "runtime.h"

typedef struct {
    int32_t handle;
    int32_t error_code;
    const char* error_msg;
} cs_file_result_t;

// Maps to: fn File.open(path: str, mode: str) -> Result[File, IOError]
cs_file_result_t std_fs_file_open(cs_string_t path, cs_string_t mode) {
    cs_file_result_t res = {0};
    
    // Convert Coolscript string to null-terminated C string for open()
    char* c_path = (char*)malloc(path.len + 1);
    memcpy(c_path, path.ptr, path.len);
    c_path[path.len] = '\0';

    int flags = O_RDONLY; // Simplified for this example
    int fd = open(c_path, flags);
    free(c_path);

    if (fd < 0) {
        res.handle = -1;
        res.error_code = 1;
        res.error_msg = "Could not open file";
    } else {
        res.handle = fd;
    }
    return res;
}

// Reads entire file into a buffer
cs_string_t std_fs_file_read_all(int32_t handle) {
    struct stat st;
    fstat(handle, &st);
    
    char* buffer = (char*)cs_alloc(st.st_size);
    read(handle, buffer, st.st_size);

    cs_string_t res = { .ptr = buffer, .len = (int64_t)st.st_size };
    return res;
}

void std_fs_file_close(int32_t handle) {
    if (handle >= 0) {
        close(handle);
    }
}

```

---

## Linking Strategy for `cool build`

When your compiler processes the `Gold Standard` program, it will generate an object file `main.o`. To create the final binary, you will link it like this:

```bash
# Combine all artifacts into a single static executable
clang -static main.o runtime.o std_io.o std_fs.o -lpthread -o log_processor

```

### Compiler Mapping Table (FFI)

The compiler's code generation phase needs to know the exact names of these C functions.

| Coolscript Call | C Function Name | Argument Passing |
| --- | --- | --- |
| `print(s)` | `std_io_print` | Pass `cs_string_t` by value (small struct). |
| `file.open(...)` | `std_fs_file_open` | Returns `cs_file_result_t` (Result type). |
| `file.read_all()` | `std_fs_file_read_all` | Returns `cs_string_t` (Owned memory). |
| `file.close()` | `std_fs_file_close` | Injected during ownership burn. |

---

## Ownership Trace in Action

In the **Gold Standard** example:

1. **`file.read_all()`** calls `std_fs_file_read_all`. This C function calls `cs_alloc`.
2. The resulting `cs_string_t` is handed to the `content` variable in Coolscript.
3. Coolscript **owns** this pointer.
4. At the end of `main()`, the compiler sees that `content` is still owned and hasn't been moved.
5. The compiler injects a call to `cs_free(content.ptr)`.

This ensures **Zero Leaks** without a Garbage Collector.
