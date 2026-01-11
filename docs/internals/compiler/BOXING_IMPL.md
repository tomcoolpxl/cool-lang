To lower the `cool.box` operation into LLVM IR, the compiler must transform a high-level ownership-aware "Box" into a low-level **Fat Pointer**. This requires creating a static structure for the VTable and a runtime structure for the boxed object itself.

The following C++ logic demonstrates how to implement this pass within the MLIR framework.

---

## C++ Lowering Pass for `cool.box`

This pass performs two critical tasks: it ensures a static VTable exists for the type implementation and then packages the data pointer and VTable pointer into an LLVM struct.

```cpp
// MLIR Lowering Pass: CoolBoxToLLVM
LogicalResult matchAndRewrite(cool::BoxOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // 1. Define the LLVM Struct Type for the Fat Pointer
    // Layout: { i8* data_ptr, vtable_struct* vptr }
    auto dataPtrType = rewriter.getI8Type().getPointerTo();
    auto vtablePtrType = getVTableStructType(op.getProtocol()).getPointerTo();
    auto fatPtrType = LLVM::LLVMStructType::getLiteral(
        getContext(), {dataPtrType, vtablePtrType});

    // 2. Get the static VTable symbol for this implementation
    // e.g., @Human_as_Speaker
    auto vtableGlobal = getOrInsertVTable(op.getVTableSymbol());

    // 3. Construct the Fat Pointer instance
    Value fatPtr = rewriter.create<LLVM::UndefOp>(loc, fatPtrType);

    // Insert the Data Pointer (the actual struct instance)
    Value dataPtr = adaptor.getInstance(); 
    dataPtr = rewriter.create<LLVM::BitcastOp>(loc, dataPtrType, dataPtr);
    fatPtr = rewriter.create<LLVM::InsertValueOp>(loc, fatPtr, dataPtr, 0);

    // Insert the VTable Pointer
    Value vptr = rewriter.create<LLVM::AddressOfOp>(loc, vtablePtrType, vtableGlobal);
    fatPtr = rewriter.create<LLVM::InsertValueOp>(loc, fatPtr, vptr, 1);

    // 4. Replace the high-level box with the LLVM Fat Pointer
    rewriter.replaceOp(op, fatPtr);
    return success();
}

```

---

## Static VTable Generation

The VTable must be generated as a global constant in the LLVM module. It includes the "Opaque Delete" function to ensure that when a Protocol Box is burned, the underlying concrete memory is freed correctly.

### LLVM Global VTable Layout

```llvm
; Example for Human as Speaker
@Human_as_Speaker_vtable = internal constant {
    void (i8*)*, ; speak() implementation
    void (i8*)*, ; shout() implementation
    void (i8*)* ; __delete() destructor
} {
    void (i8*)* @Human_speak_impl,
    void (i8*)* @Human_shout_impl,
    void (i8*)* @__delete_Human
}

```

---

## The "Opaque Delete" Mechanism

Because the code calling a Protocol method doesn't know the size of the underlying struct, it cannot call `free()` directly. Instead, it delegates to the VTable.

### Step-by-Step Execution:

1. **Ownership Ends**: A `move Speaker` variable goes out of scope.
2. **Compiler Injection**: The compiler injects a call to the third entry in the VTable.
3. **Execution**: The call jumps to `@__delete_Human(void* data)`.
4. **Concrete Destruction**: This function casts the `void*` back to `Human*`, calls its internal destructor (to burn any fields inside), and then calls `cs_free()`.

---

## Milestone 2: Technical Summary

With the addition of Generics and Protocol Boxing, Coolscript now supports the full spectrum of modern systems programming:

| Feature | Implementation | Benefit |
| --- | --- | --- |
| **Generics** | Monomorphization | No runtime overhead, specialized speed. |
| **Static Traits** | Protocol Constraints | Safe, reusable generic logic. |
| **Dynamic Traits** | VTable Boxing | Heterogeneous collections (e.g., `List[Speaker]`). |
| **Safe Cleanup** | Opaque Destructors | Deterministic memory safety for dynamic types. |

---

## The Final Compiler Ecosystem

Your compiler architecture is now complete for Milestone 2.

* **Frontend**: PEG Parser (Indentation-aware).
* **Middle-end**: MLIR (Ownership, Generics, and Protocol Boxing).
* **Backend**: LLVM (Static binary, Opaque Destructors).
* **Runtime**: C-based (Tasks, Channels, Memory).
