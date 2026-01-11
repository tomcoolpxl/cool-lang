To support Milestone 2's trait-based polymorphism, the compiler needs a centralized way to verify that a `Struct` satisfies the requirements of a `Protocol`. This is handled by the **Protocol Implementation Map (PIM)**.

In Coolscript, this map is used during two distinct phases:

1. **Static Verification**: During monomorphization, to ensure a concrete type `T` satisfies a constraint `[T: Protocol]`.
2. **Dynamic Dispatch**: To generate "Fat Pointers" (VTable + Data) when a protocol is used as an object (e.g., `List[Speaker]`).

---

## The Protocol Map Structure

The PIM is essentially a nested dictionary maintained by the Symbol Table during the semantic analysis pass.

```python
# Conceptual structure of the Implementation Map
{
    "Speaker": {
        "Human": {
            "methods": {
                "speak": "@Human_speak_impl",
                "shout": "@Human_shout_impl"
            },
            "associated_types": {
                "Language": "English"
            }
        },
        "Robot": {
            "methods": {
                "speak": "@Robot_speak_impl",
                "shout": "@Default_shout_impl" # Inherited from Protocol default
            }
        }
    }
}

```

---

## C++ Implementation (Compiler Internals)

In your MLIR-based compiler, this is implemented as a global registry.

```cpp
struct ProtocolImpl {
    // Maps method names to their concrete Function symbols
    llvm::StringMap<mlir::SymbolRefAttr> methodMap;
    // Maps associated type names to actual MLIR Types
    llvm::StringMap<mlir::Type> typeMap;
};

class ProtocolRegistry {
private:
    // Outer Key: Protocol Name, Inner Key: Struct Name
    llvm::StringMap<llvm::StringMap<ProtocolImpl>> registry;

public:
    void registerImpl(StringRef proto, StringRef strct, ProtocolImpl impl) {
        registry[proto][strct] = std::move(impl);
    }

    bool satisfies(StringRef strct, StringRef proto) {
        return registry.count(proto) && registry[proto].count(strct);
    }

    ProtocolImpl* getImpl(StringRef proto, StringRef strct) {
        if (!satisfies(strct, proto)) return nullptr;
        return &registry[proto][strct];
    }
};

```

---

## VTable Generation for Dynamic Dispatch

When the compiler encounters a Protocol used as a type (e.g., `let s: Speaker = move my_human`), it must lower this into a **Boxed Object**.

The ODS logic for this involves creating a constant VTable for every unique Struct-Protocol pair.

```mlir
// MLIR Representation of a Protocol Box
cool.vtable @Human_as_Speaker {
    methods = {
        speak = @Human_speak_impl,
        shout = @Human_shout_impl
    }
}

// Creating the boxed object (Fat Pointer)
%box = cool.box %human_instance, @Human_as_Speaker : (!cool.struct<Human>) -> !cool.protocol<Speaker>

```

### The Fat Pointer Layout

At the LLVM level, the `!cool.protocol<Speaker>` is lowered to a struct containing two pointers:

1. **`data_ptr`**: A pointer to the actual struct data (e.g., the `Human` instance).
2. **`vtable_ptr`**: A pointer to the static vtable generated above.

---

## Ownership and Protocols

A critical safety check for Milestone 2 is **Boxed Ownership**. When a struct is moved into a protocol box, the box becomes the new owner.

```python
fn run_speech(s: move Speaker):
    s.speak()
    # At the end of this scope, the compiler must know how to free 's'
    # because 's' could be a Human or a Robot (different sizes).

```

### The "Opaque Delete"

To handle this, every VTable in Coolscript includes a mandatory **`__delete`** entry.

1. The compiler generates a specialized `__delete_Human` function.
2. This function is added to the `@Human_as_Speaker` vtable.
3. When `run_speech` ends, it calls `s.vtable->__delete(s.data)`.

---

## Milestone 2 Integration Plan

1. **Registry**: Implement the `ProtocolRegistry` in your symbol table.
2. **Verification**: During `InstantiateOp`, query the registry to validate constraints.
3. **Boxing**: Implement the `cool.box` operation and vtable generation in the MLIR-to-LLVM lowering pass.
