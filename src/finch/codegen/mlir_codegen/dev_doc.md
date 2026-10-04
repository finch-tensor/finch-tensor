# MLIR buffer ownership and resizing

## The ownership rule

A `Buffer` backed by a `NumpyBuffer` is owned by Python for its entire
lifetime, including for the duration of any kernel call that touches it.
MLIR is handed a pointer and a length into that memory.

The resize path is: MLIR calls out to a function pointer supplied by
`numpy_buffer_resize_callback`, and MLIR receives back the current
data pointer to use for the rest of the call. MLIR never decides
where the new memory comes from or frees the old memory itself.

This makes `deserialize_from_mlir` a no-op: by the time the kernel call
returns, the callback has already reassigned `buf.arr` in Python. There is
nothing left to sync.

## What crosses the boundary

A resizable buffer argument is serialized to a `MLIRNumpyBuffer` struct
`{arr: PyObject*, data: ptr, length: size_t, resize: fn ptr}`
and passed as a single `!llvm.ptr`.

The function body loads the current descriptor from the mutable box and
converts it to a `memref<?xT>`. It then uses `memref.load`, `memref.store`,
and `memref.dim` in the same way as a plain memref argument.

## Life of a resizable buffer argument

1. `serialize_to_mlir`: build a `MLIRNumpyBuffer`.
2. `mlir_unpack`: read `data`/`length` out of the incoming
   `!llvm.ptr`, then construct a local `memref<?xT>` value via
   `llvm.mlir.undef` → `insertvalue` allocated ptr / aligned ptr /
   offset=0 / size / stride=1 → `unrealized_conversion_cast` to
   `memref<?xT>`. The descriptor is stored in a mutable box. The slot remembers
   both the box and the original incoming `!llvm.ptr`: buffer operations load
   the current descriptor from the box, while resize and repack use the original
   pointer to access and update the serialized buffer fields.
3. `Resize`: load the callback function pointer out of the
   handle, `llvm.call` it with the new length, get back a new
   `data` pointer, and build a new memref descriptor from that pointer and the
   requested length. Store the new descriptor in the slot's mutable box.
   The slot and box pointer remain unchanged; future buffer operations
   load the resized descriptor from the box.
4. `mlir_repack`: load the current descriptor from the mutable box, extract its
   data pointer and length, and store them in the handle's `data` and `length`
   fields. This keeps the serialized buffer metadata synchronized with the
   descriptor used by the kernel.
5. `deserialize_from_mlir`: no-op. The callback already mutated the
   owning `NumpyBuffer` object's NumPy array during the resize step.

## MemrefBuffer ownership and resizing

A `MemrefBuffer` holds an MLIR-owned allocation. It crosses the function
boundary as a pointer to its mutable `memref<?xT>` descriptor.

- `mlir_unpack` uses the incoming descriptor pointer as its mutable box. Buffer
operations load the current descriptor from the box and convert it to a memref.

- `Resize` loads the current descriptor, converts it to a memref, and calls
`memref.realloc`. It then converts the resized memref back to its descriptor
representation and stores it in the same mutable box.

- `mlir_repack` and `deserialize_from_mlir` are no-ops because resize updates
the `MemrefBuffer` descriptor in place. `construct_from_mlir` wraps a returned
descriptor pointer in a `MemrefBuffer` without copying the element data.
