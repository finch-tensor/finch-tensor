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
    buffer: str


@dataclass(frozen=True)
class MLIRMemrefBufferMethods:
    alloc: str
    resize: str
    free: str


class MLIRMemrefBufferLibrary:
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

        desc = mlir_ctype(self.ftype.mlir_type())()
        self.comp[2].invoke(
            self.methods.alloc,
            ctypes.pointer(mlir_ctype(ftypes.intp)(length)),
            ctypes.pointer(desc),
        )
        return desc

    def resize(self, descriptor, new_length):
        if new_length < 0:
            raise ValueError("Buffer length cannot be negative")

        resized = type(descriptor)()
        self.comp[2].invoke(
            self.methods.resize,
            ctypes.pointer(ctypes.pointer(descriptor)),
            ctypes.pointer(mlir_ctype(ftypes.intp)(new_length)),
            ctypes.pointer(resized),
        )
        return resized

    def free(self, descriptor):
        self.comp[2].invoke(
            self.methods.free,
            ctypes.pointer(ctypes.pointer(descriptor)),
        )


class MLIRMemrefBufferBackend:
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
        memref_type = ftype.mlir_type()
        feed = ctx.feed
        inner = f"{feed}{ctx.tab}"

        ctx.exec(
            f"{feed}func.func @{methods.alloc}(%length: index) -> {memref_type} "
            f"attributes {{llvm.emit_c_interface}} {{\n"
            f"{inner}%buffer = memref.alloc(%length) : {memref_type}\n"
            f"{inner}func.return %buffer : {memref_type}\n"
            f"{feed}}}"
        )
        ctx.exec(
            f"{feed}func.func @{methods.resize}("
            f"%buffer: {memref_type}, %length: index) -> {memref_type} "
            f"attributes {{llvm.emit_c_interface}} {{\n"
            f"{inner}%resized = memref.realloc %buffer(%length) "
            f": {memref_type} to {memref_type}\n"
            f"{inner}func.return %resized : {memref_type}\n"
            f"{feed}}}"
        )
        ctx.exec(
            f"{feed}func.func @{methods.free}(%buffer: {memref_type}) "
            f"attributes {{llvm.emit_c_interface}} {{\n"
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
        return np.intp(self.buffer.sizes[0])

    def load(self, index):
        value = self.castbuffer[index]
        return construct_from_mlir(self.ftype.element_type, value)

    def store(self, index, value):
        self.castbuffer[index] = serialize_to_mlir(self.ftype.element_type, value)

    def resize(self, new_length):
        new_descriptor = self._library.resize(
            self.buffer,
            new_length,
        )
        self.buffer = new_descriptor


class MemrefBufferFType(MLIRBufferFType, MLIRUnpackableFType):
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

    def mlir_type(self):
        return f"memref<?x{mlir_type(self.element_type)}>"

    def mlir_length(self, ctx: MLIRContext, buf: MLIRMemrefBufferFields):
        dimension = ctx.constant(0, "index")
        result = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{result} = memref.dim {buf.buffer}, {dimension} "
            f": {self.mlir_type()}"
        )
        return result

    def mlir_load(
        self,
        ctx: MLIRContext,
        buf: MLIRMemrefBufferFields,
        idx: AssemblyExpression,
    ):
        index = mlir_cast_value(ctx, ctx(idx), idx.result_type, ftypes.intp)
        result = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{result} = memref.load {buf.buffer}[{index}] "
            f": {self.mlir_type()}"
        )
        return result

    def mlir_store(
        self,
        ctx: MLIRContext,
        buf: MLIRMemrefBufferFields,
        idx: AssemblyExpression,
        value: AssemblyExpression,
    ):
        index = mlir_cast_value(ctx, ctx(idx), idx.result_type, ftypes.intp)
        stored_value = ctx(value)
        ctx.exec(
            f"{ctx.feed}memref.store {stored_value}, {buf.buffer}[{index}] "
            f": {self.mlir_type()}"
        )

    def mlir_resize(
        self,
        ctx: MLIRContext,
        buf: MLIRMemrefBufferFields,
        new_len: AssemblyExpression,
    ): ...

    def mlir_unpack(self, ctx: MLIRContext, _, val):
        return MLIRMemrefBufferFields(ctx(val))

    def mlir_repack(
        self,
        ctx: MLIRContext,
        var_n: str,
        obj: MLIRMemrefBufferFields,
    ): ...

    def serialize_to_mlir(self, obj: MemrefBuffer):
        return obj.buffer

    def deserialize_from_mlir(self, obj: MemrefBuffer, mlir_buffer):
        # this is handled by the resize callback
        pass

    def construct_from_mlir(self, mlir_buffer):
        return MemrefBuffer(
            mlir_buffer, self.element_type, MLIRMemrefBufferBackend.library(self)
        )
