from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from finch.algebra import FType, ftypes
from finch.codegen.mlir_codegen import (
    MLIRBufferFType,
    MLIRContext,
    MLIRUnpackableFType,
    construct_from_mlir,
    mlir_cast_value,
    mlir_ctype,
    mlir_type,
    serialize_to_mlir,
)
from finch.finch_assembly import Buffer
from finch.finch_assembly.nodes import AssemblyExpression


@dataclass
class MLIRMemrefBufferFields:
    box: str


@dataclass
class MLIRMemrefBufferMethods:
    alloc: str
    resize: str
    free: str


class MLIRMemrefBufferLibrary:
    """
    Class that invokes the compiled MLIR helpers.

    """

    def __init__(
        self,
        comp,
        ftype: MemrefBufferFType,
        methods: MLIRMemrefBufferMethods,
    ):
        self.comp = comp
        self.ftype = ftype
        self.methods = methods

    def alloc(self, length):
        if length < 0:
            raise ValueError("Buffer length cannot be negative")

        desc = mlir_ctype(self.ftype.mlir_buffer_type())()
        box = ctypes.c_void_p(ctypes.addressof(desc))
        self.comp[2].invoke(
            self.methods.alloc,
            ctypes.pointer(box),
            ctypes.pointer(mlir_ctype(ftypes.intp)(length)),
        )
        return desc

    def resize(self, descriptor, new_length):
        if new_length < 0:
            raise ValueError("Buffer length cannot be negative")

        box = ctypes.c_void_p(ctypes.addressof(descriptor))
        self.comp[2].invoke(
            self.methods.resize,
            ctypes.pointer(box),
            ctypes.pointer(mlir_ctype(ftypes.intp)(new_length)),
        )
        return descriptor

    def free(self, descriptor):
        box = ctypes.c_void_p(ctypes.addressof(descriptor))
        self.comp[2].invoke(
            self.methods.free,
            ctypes.pointer(box),
        )


class MLIRMemrefBufferBackend:
    """
    Class that compiles the MLIR helper method.

    """

    _library: ClassVar[dict[MemrefBufferFType, MLIRMemrefBufferLibrary]] = {}

    @classmethod
    def gen_code(
        cls,
        ctx: MLIRContext,
        ftype: MemrefBufferFType,
    ) -> MLIRMemrefBufferMethods:
        methods = MLIRMemrefBufferMethods(
            alloc="memref_alloc",
            resize="memref_resize",
            free="memref_free",
        )
        memref_type = ftype.mlir_buffer_type()
        desc_type = ftype.mlir_descriptor_type()
        feed = ctx.feed
        inner = f"{feed}{ctx.tab}"

        # alloc library function
        ctx.exec(
            f"{feed}func.func @{methods.alloc}(%box: !llvm.ptr, %length: index) "
            f"attributes {{llvm.emit_c_interface}} {{\n"
            f"{inner}%buffer = memref.alloc(%length) : {memref_type}\n"
            f"{inner}%desc = builtin.unrealized_conversion_cast %buffer "
            f": {memref_type} to {desc_type}\n"
            f"{inner}llvm.store %desc, %box : {desc_type}, !llvm.ptr\n"
            f"{inner}func.return\n"
            f"{feed}}}"
        )

        # resize library function
        ctx.exec(
            f"{feed}func.func @{methods.resize}("
            f"%box: !llvm.ptr, %length: index) "
            f"attributes {{llvm.emit_c_interface}} {{\n"
            f"{inner}%desc = llvm.load %box : !llvm.ptr -> {desc_type}\n"
            f"{inner}%buffer = builtin.unrealized_conversion_cast %desc "
            f": {desc_type} to {memref_type}\n"
            f"{inner}%resized = memref.realloc %buffer(%length) "
            f": {memref_type} to {memref_type}\n"
            f"{inner}%resized_desc = builtin.unrealized_conversion_cast %resized "
            f": {memref_type} to {desc_type}\n"
            f"{inner}llvm.store %resized_desc, %box : {desc_type}, !llvm.ptr\n"
            f"{inner}func.return\n"
            f"{feed}}}"
        )

        # free library function
        ctx.exec(
            f"{feed}func.func @{methods.free}(%box: !llvm.ptr) "
            f"attributes {{llvm.emit_c_interface}} {{\n"
            f"{inner}%desc = llvm.load %box : !llvm.ptr -> {desc_type}\n"
            f"{inner}%buffer = builtin.unrealized_conversion_cast %desc "
            f": {desc_type} to {memref_type}\n"
            f"{inner}memref.dealloc %buffer : {memref_type}\n"
            f"{inner}func.return\n"
            f"{feed}}}"
        )
        return methods

    @classmethod
    def library(cls, ftype: MemrefBufferFType) -> MLIRMemrefBufferLibrary:
        if ftype in cls._library:
            return cls._library[ftype]

        from finch.codegen.mlir_codegen.mlir import load_mlir_engine

        ctx = MLIRContext()
        methods = cls.gen_code(ctx, ftype)
        library = MLIRMemrefBufferLibrary(
            comp=load_mlir_engine(ctx.emit_global()),
            ftype=ftype,
            methods=methods,
        )
        cls._library[ftype] = library
        return library


class MemrefBuffer(Buffer):
    """
    Class that provides Python access to an MLIR-owned memref buffer.

    """

    def __init__(
        self,
        memref,
        dtype: FType,
        library: MLIRMemrefBufferLibrary,
    ):
        self._dtype = dtype
        self._library = library
        self.buffer = memref

    def __del__(self):
        if hasattr(self, "_library") and hasattr(self, "buffer"):
            self._library.free(self.buffer)

    @property
    def ftype(self):
        return MemrefBufferFType(self._dtype)

    @property
    def castbuffer(self):
        from mlir.runtime import ranked_memref_to_numpy

        return ranked_memref_to_numpy(ctypes.pointer(self.buffer))

    def length(self):
        return np.intp(self.buffer.shape[0])

    def load(self, index):
        value = self.castbuffer[index]
        return construct_from_mlir(self.ftype.element_type, value)

    def store(self, index, value):
        serialized = serialize_to_mlir(self.ftype.element_type, value)
        self.castbuffer[index] = getattr(serialized, "value", serialized)

    def resize(self, new_length):
        self.buffer = self._library.resize(
            self.buffer,
            new_length,
        )


class MemrefBufferFType(MLIRBufferFType, MLIRUnpackableFType):
    """
    A ftype for memref buffers that defines MLIR operations.

    """

    def __init__(self, element_type: FType):
        self._element_type = element_type

    def __eq__(self, other):
        if not isinstance(other, MemrefBufferFType):
            return False
        return self._element_type == other._element_type

    def __hash__(self):
        return hash(("MemrefBufferFType", self._element_type))

    @property
    def element_type(self):
        return self._element_type

    def __call__(self, length: int = 0):
        library = MLIRMemrefBufferBackend.library(self)
        descriptor = library.alloc(length)
        return MemrefBuffer(descriptor, self.element_type, library)

    # This is the pointer type pass into the params
    def mlir_type(self):
        return "!llvm.ptr"

    # This is the rank-1 sized memref type used for MLIR buffer operations.
    def mlir_buffer_type(self):
        return f"memref<?x{mlir_type(self.element_type)}>"

    # this is the rank-1 memref descriptor type
    def mlir_descriptor_type(self):
        return "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"

    # Return the current length of the MLIR NumPy buffer
    def mlir_length(self, ctx: MLIRContext, buf: MLIRMemrefBufferFields):
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        dim = ctx.constant(0, "index")
        result = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{result} = memref.dim {buffer}, {dim} : "
            f"{self.mlir_buffer_type()}"
        )
        return result

    # Load the buffer into an SSA value
    def mlir_load(
        self,
        ctx: MLIRContext,
        buf: MLIRMemrefBufferFields,
        idx: AssemblyExpression,
    ):
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        index = mlir_cast_value(ctx, ctx(idx), idx.result_type, ftypes.intp)
        result = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{result} = memref.load {buffer}[{index}] : "
            f"{self.mlir_buffer_type()}"
        )
        return result

    # Store a value into the memref buffer
    def mlir_store(
        self,
        ctx: MLIRContext,
        buf: MLIRMemrefBufferFields,
        idx: AssemblyExpression,
        value: AssemblyExpression,
    ):
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        index = mlir_cast_value(ctx, ctx(idx), idx.result_type, ftypes.intp)
        val = ctx(value)
        ctx.exec(
            f"{ctx.feed}memref.store {val}, {buffer}[{index}] : "
            f"{self.mlir_buffer_type()}"
        )

    # Resize the memref buffer using memref.realloc
    def mlir_resize(
        self,
        ctx: MLIRContext,
        buf: MLIRMemrefBufferFields,
        new_len: AssemblyExpression,
    ):
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        result = ctx.new_ssa()
        length = mlir_cast_value(ctx, ctx(new_len), new_len.result_type, ftypes.intp)
        memref_t = self.mlir_buffer_type()
        ctx.exec(
            f"{ctx.feed}{result} = memref.realloc {buffer}({length}) : "
            f"{memref_t} to {memref_t}"
        )
        desc = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{desc} = builtin.unrealized_conversion_cast "
            f"{result} : {memref_t} to {desc_t}"
        )
        ctx.exec(f"{ctx.feed}llvm.store {desc}, {buf.box} : {desc_t}, !llvm.ptr")

    # Use the pointer as the descriptor box.
    def mlir_unpack(self, ctx: MLIRContext, _, val):
        return MLIRMemrefBufferFields(ctx(val))

    def mlir_repack(self, ctx, var_n, obj):
        # Buffer operations already update the incoming descriptor in place.
        pass

    # Pass the address of the Python-owned descriptor across the MLIR ABI.
    def serialize_to_mlir(self, obj: MemrefBuffer):
        return ctypes.c_void_p(ctypes.addressof(obj.buffer))

    def deserialize_from_mlir(self, obj, mlir_buffer):
        # No copy-back is needed because Python and MLIR share the same allocation.
        pass

    # Wrap a MLIR descriptor for use as a Python Memref Buffer.
    def construct_from_mlir(self, mlir_buffer):
        descriptor = ctypes.cast(
            mlir_buffer, ctypes.POINTER(mlir_ctype(self.mlir_buffer_type()))
        ).contents
        return MemrefBuffer(
            descriptor, self.element_type, MLIRMemrefBufferBackend.library(self)
        )
