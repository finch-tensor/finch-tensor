# MLIR buffer ownership and resizing

## The ownership rule

A `Buffer` backed by a `NumpyBuffer` is owned by Python for its entire
lifetime, including for the duration of any kernel call that touches it.
MLIR is handed a pointer and a length into that memory.

The resize path is: MLIR calls out to a function pointer supplied by
`numpy_buffer_resize_callback`, and MLIR receives back a fresh
`(pointer, length)` pair to use for the rest of the call. MLIR never decides
where the new memory comes from or frees the old memory itself.

This makes `deserialize_from_mlir` a no-op, the same way
`deserialize_from_c` is: by the time the kernel call returns, the
callback has already reassigned `buf.arr` in Python. There is nothing left
to sync.

## What crosses the boundary

A resizable buffer argument is serialized to a `MLIRNumpyBuffer` struct
`{arr: PyObject*, data: ptr, length: size_t, size: fn ptr}`
and passed as a single `!llvm.ptr`.

The function body calls `memref.load` / `memref.store` / `memref.dim`
exactly as it would on a plain memref argument, reusing the existing
memref-level codegen for the common load and store pipeline, and only
changing how the buffer's identity crosses the function boundary and
how a resize is requested.

## Life of a resizable buffer argument

1. `serialize_to_mlir`: build a `MLIRNumpyBuffer`.
2. `mlir_unpack`: read `data`/`length` out of the incoming
   `!llvm.ptr`, then construct a local `memref<?xT>` value via
   `llvm.mlir.undef` → `insertvalue` allocated ptr / aligned ptr /
   offset=0 / size / stride=1 → `unrealized_conversion_cast` to
   `memref<?xT>`. The slot for this buffer remembers both the local memref
   and the original incoming `!llvm.ptr`, since both resize and repack
   need to get back to the handle.
3. `Resize`: load the callback function pointer out of the
   handle, `llvm.call` it with the new length, get back a new
   `data` pointer, rebuild the local memref value from it, and rebind
   the slot to that new value. The old memref value is never read again
   after this point, so "resizing" means the slot in the context's `slots`
   dict starts pointing at a different value.
4. `mlir_repack`: read the current base pointer and length of
    the memref the slot holds now and store them into the
    handle's `data` and `length` fields. This makes a resize
    visible outside the function if the buffer is also a Python-level
    argument.
5. `deserialize_from_mlir`: no-op. The callback already mutated the
   `NumpyBuffer` object directly during the resize step.
