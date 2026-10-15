import ctypes
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import numpy as np

import numba
import numba.typed

from finch.algebra import FType, TupleFType, ftype, ftypes, np_dtype
from finch.codegen.c_codegen import CBufferFType, CContext, CUnpackableFType, c_type
from finch.codegen.mlir_codegen import (
    MLIRBufferFType,
    MLIRContext,
    llvm_type,
    mlir_cast_value,
    mlir_type,
)
from finch.codegen.numba_codegen import NumbaBufferFType
from finch.finch_assembly import Buffer
from finch.finch_assembly.nodes import AssemblyExpression
from finch.util import qual_str


class NumbaBufferFields(NamedTuple):
    arr: str
    obj: str


class CBufferFields(NamedTuple):
    data: str
    length: str
    obj: str


@dataclass
class MLIRBufferFields:
    box: str
    obj: str


def _tuple_leaves(dtype: np.dtype, names=(), idxs=(), offset=0):
    """Yield the field names, field indices, dtype, and byte offset of each
    scalar field of a structured dtype."""
    assert dtype.fields is not None
    for k, (name, (field_dtype, field_offset, *_)) in enumerate(dtype.fields.items()):
        if field_dtype.names is None:
            yield (*names, name), (*idxs, k), field_dtype, offset + field_offset
        else:
            yield from _tuple_leaves(
                field_dtype, (*names, name), (*idxs, k), offset + field_offset
            )


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.py_object), ctypes.c_size_t)
def numpy_buffer_resize_callback(buf_ptr, new_length):
    """
    A Python callback function that resizes the NumPy array.
    """
    buf = buf_ptr.contents.value
    buf.arr = np.resize(buf.arr, new_length)
    return buf.arr.ctypes.data


class CNumpyBuffer(ctypes.Structure):
    _fields_ = [
        ("arr", ctypes.py_object),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("resize", type(numpy_buffer_resize_callback)),
    ]


class MLIRNumpyBuffer(ctypes.Structure):
    _fields_ = [
        ("arr", ctypes.py_object),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("resize", type(numpy_buffer_resize_callback)),
    ]


class NumpyBuffer(Buffer):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the Buffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr = arr

    @property
    def owner(self):
        return self.arr

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a NumpyBufferFType.
        """
        return NumpyBufferFType(ftype(self.arr.dtype))

    # TODO should be property
    def length(self):
        return self.arr.size

    def load(self, index: int):
        value = self.arr[index]
        if isinstance(self.ftype.element_type, TupleFType):
            return tuple(
                value[name] for name in self.ftype.element_type.struct_fieldnames
            )
        return value

    def store(self, index: int, value):
        self.arr[index] = value

    def resize(self, new_length: int):
        self.arr = np.resize(self.arr, new_length)

    def __str__(self):
        arr_str = str(self.arr).replace("\n", "")
        return f"np_buf({arr_str})"

    def __repr__(self):
        arr_repr = repr(self.arr).replace("\n", "")
        return f"NumpyBuffer({arr_repr})"


class NumpyBufferFType(
    CBufferFType, NumbaBufferFType, MLIRBufferFType, CUnpackableFType
):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class.
    """

    def __init__(self, element_type: FType):
        self._element_type = ftype(np_dtype(ftype(element_type)))

    @property
    def _dtype(self):
        return np_dtype(self._element_type)

    def __eq__(self, other):
        if not isinstance(other, NumpyBufferFType):
            return False
        return self._element_type == other._element_type

    def __str__(self):
        return f"np_buf_t({qual_str(self._dtype.type)})"

    def __repr__(self):
        return f"NumpyBufferFType({repr(self._element_type)})"

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return ftypes.intp

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._element_type

    def __hash__(self):
        return hash(self._element_type)

    def __call__(self, len: int = 0, element_type: FType | None = None):
        return NumpyBuffer(np.zeros(len, dtype=self._dtype))

    def _normalize_layout(self, obj):
        # Compiled tuple accesses assume the packed layout of this ftype, while
        # a borrowed array (e.g. from Julia) may be aligned.
        if self._dtype.names is not None and obj.arr.dtype != self._dtype:
            obj.arr = obj.arr.astype(self._dtype)

    def c_type(self):
        return ctypes.POINTER(CNumpyBuffer)

    def c_length(self, ctx: "CContext", buf: CBufferFields):
        return buf.length

    def c_data(self, ctx: "CContext", buf: CBufferFields):
        return buf.data

    def _c_tuple_accessors(self, ctx: "CContext"):
        # Tuple elements keep NumPy's packed layout, which C structs don't share,
        # so each field is copied at its byte offset.
        t = ctx.ctype_name(c_type(self.element_type))
        name = t.removeprefix("struct ")
        load, store = f"finch_load_{name}", f"finch_store_{name}"
        size = self._dtype.itemsize
        fields = [
            ("".join(f".{n}" for n in names), offset)
            for names, _, _, offset in _tuple_leaves(self._dtype)
        ]
        loads = "".join(
            f"    memcpy(&v{path}, (const char*)data + i * {size} + {offset}, "
            f"sizeof(v{path}));\n"
            for path, offset in fields
        )
        stores = "".join(
            f"    memcpy((char*)data + i * {size} + {offset}, &v{path}, "
            f"sizeof(v{path}));\n"
            for path, offset in fields
        )
        ctx.add_header("#include <stddef.h>")
        ctx.add_header("#include <string.h>")
        ctx.add_header(
            f"static inline {t} {load}(const void* data, size_t i) {{\n"
            f"    {t} v;\n{loads}    return v;\n}}"
        )
        ctx.add_header(
            f"static inline void {store}(void* data, size_t i, {t} v) {{\n{stores}}}"
        )
        return load, store

    def c_load(self, ctx: "CContext", buf: CBufferFields, idx: "AssemblyExpression"):
        if isinstance(self.element_type, TupleFType):
            load, _ = self._c_tuple_accessors(ctx)
            return f"{load}({buf.data}, {ctx(idx)})"
        return f"({buf.data})[{ctx(idx)}]"

    def c_store(
        self,
        ctx: "CContext",
        buf: CBufferFields,
        idx: "AssemblyExpression",
        value: "AssemblyExpression",
    ):
        if isinstance(self.element_type, TupleFType):
            _, store = self._c_tuple_accessors(ctx)
            ctx.exec(f"{ctx.feed}{store}({buf.data}, {ctx(idx)}, {ctx(value)});")
            return
        ctx.exec(f"{ctx.feed}({buf.data})[{ctx(idx)}] = {ctx(value)};")

    def c_resize(self, ctx, buf: CBufferFields, new_len):
        new_len = ctx(ctx.cache("len", new_len))
        data = buf.data
        length = buf.length
        obj = buf.obj
        t = ctx.ctype_name(c_type(self.element_type))
        ctx.exec(
            f"{ctx.feed}{data} = ({t}*){obj}->resize(&{obj}->arr, {new_len});\n"
            f"{ctx.feed}{length} = {new_len};"
        )
        return

    def c_unpack(self, ctx, var_n, val):
        """
        Unpack the buffer into C context.
        """
        data = ctx.freshen(var_n, "data")
        length = ctx.freshen(var_n, "length")
        t = ctx.ctype_name(c_type(self.element_type))
        ctx.add_header("#include <stddef.h>")
        ctx.exec(
            f"{ctx.feed}{t}* {data} = ({t}*){ctx(val)}->data;\n"
            f"{ctx.feed}size_t {length} = {ctx(val)}->length;"
        )

        return CBufferFields(data, length, var_n)

    def c_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        ctx.exec(
            f"{ctx.feed}{lhs}->data = (void*){obj.data};\n"
            f"{ctx.feed}{lhs}->length = {obj.length};"
        )
        return

    def serialize_to_c(self, obj):
        """
        Serialize the NumPy buffer to a C-compatible structure.
        """
        self._normalize_layout(obj)
        data = ctypes.c_void_p(obj.arr.ctypes.data)
        length = obj.arr.size
        obj._self_obj = ctypes.py_object(obj)
        obj._c_callback = numpy_buffer_resize_callback
        obj._c_buffer = CNumpyBuffer(obj._self_obj, data, length, obj._c_callback)
        return ctypes.pointer(obj._c_buffer)

    def deserialize_from_c(self, obj, c_buffer):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """
        # this is handled by the resize callback

    def construct_from_c(self, c_buffer):
        """
        Construct a NumpyBuffer from a C-compatible structure.
        """
        return c_buffer.contents.arr

    def numba_type(self) -> type:
        return list[np.ndarray]

    def numba_jitclass_type(self) -> numba.types.Type:
        return numba.types.ListType(
            numba.types.Array(numba.from_dtype(self._dtype), 1, "C")
        )

    def numba_length(self, ctx, buf: NumbaBufferFields):
        arr = buf.arr
        return f"len({arr})"

    def numba_load(self, ctx, buf, idx):
        buf = cast(NumbaBufferFields, buf)
        arr = buf.arr
        if isinstance(self.element_type, TupleFType):
            idx = ctx(ctx.cache("idx", idx))
            fields = ", ".join(
                f"{arr}[{idx}]['{name}']"
                for name in self.element_type.struct_fieldnames
            )
            return f"({fields},)"
        return f"{arr}[{ctx(idx)}]"

    def numba_store(self, ctx, buf, idx, value=None):
        buf = cast(NumbaBufferFields, buf)
        arr = buf.arr
        if isinstance(self.element_type, TupleFType):
            idx = ctx(ctx.cache("idx", idx))
            value = ctx.cache("val", value)
            val_code = ctx(value)
            for i, name in enumerate(self.element_type.struct_fieldnames):
                ctx.exec(f"{ctx.feed}{arr}[{idx}]['{name}'] = {val_code}[{i}]")
            return
        ctx.exec(f"{ctx.feed}{arr}[{ctx(idx)}] = {ctx(value)}")

    def numba_resize(self, ctx, buf: NumbaBufferFields, new_len):
        arr = buf.arr
        resize = (
            "resize_records"
            if isinstance(self.element_type, TupleFType)
            else "numpy.resize"
        )
        ctx.exec(f"{ctx.feed}{arr} = {resize}({arr}, {ctx(new_len)})")

    def numba_unpack(self, ctx, var_n, val):
        """
        Unpack the buffer into Numba context.
        """
        arr = ctx.freshen(var_n, "arr")
        ctx.exec(f"{ctx.feed}{arr} = {ctx(val)}[0]")

        return NumbaBufferFields(arr, var_n)

    def numba_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from Numba context.
        """
        ctx.exec(f"{ctx.feed}{lhs}[0] = {obj.arr}")
        return

    def serialize_to_numba(self, obj):
        """
        Serialize the NumPy buffer to a Numba-compatible object.
        """
        return numba.typed.List([obj.arr])

    def deserialize_from_numba(self, obj, numba_buffer):
        obj.arr = numba_buffer[0]
        return

    def construct_from_numba(self, numba_buffer):
        """
        Construct a NumpyBuffer from a Numba-compatible object.
        """
        return NumpyBuffer(numba_buffer[0])

    def mlir_type(self):
        return "!llvm.ptr"

    def mlir_buffer_type(self):
        if isinstance(self.element_type, TupleFType):
            # Memrefs can't hold structs; tuple elements are accessed through
            # the descriptor's pointer instead (see `_mlir_tuple_ptr`).
            return "memref<?xi8>"
        return f"memref<?x{mlir_type(self.element_type)}>"

    def _mlir_packed_type(self, dtype=None):
        dtype = self._dtype if dtype is None else dtype
        fields = (
            llvm_type(mlir_type(ftype(field_dtype)))
            if field_dtype.names is None
            else self._mlir_packed_type(field_dtype)
            for field_dtype, *_ in dtype.fields.values()
        )
        return f"!llvm.struct<packed ({', '.join(fields)})>"

    def _mlir_tuple_ptr(self, ctx, buf, idx):
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        data = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{data} = llvm.extractvalue {desc}[1] : {desc_t}")
        i = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{i} = arith.index_cast {self.mlir_index(ctx, idx)} "
            ": index to i64"
        )
        ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{ptr} = llvm.getelementptr {data}[{i}] : "
            f"(!llvm.ptr, i64) -> !llvm.ptr, {self._mlir_packed_type()}"
        )
        return ptr

    def _mlir_repack_tuple(self, ctx, value, src_t, dst_t):
        result = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{result} = llvm.mlir.undef : {dst_t}")
        for _, idxs, _, _ in _tuple_leaves(self._dtype):
            pos = ", ".join(map(str, idxs))
            field = ctx.new_ssa()
            ctx.exec(f"{ctx.feed}{field} = llvm.extractvalue {value}[{pos}] : {src_t}")
            updated = ctx.new_ssa()
            ctx.exec(
                f"{ctx.feed}{updated} = llvm.insertvalue {field}, "
                f"{result}[{pos}] : {dst_t}"
            )
            result = updated
        return result

    # this is the rank-1 memref descriptor type
    def mlir_descriptor_type(self):
        return "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"

    # this is the callback buffer type
    def llvm_callback_type(self):
        return "!llvm.struct<(ptr, ptr, i64, ptr)>"

    # Create our own rank-1 memeref descriptor from a data pointer and length.
    def mlir_memref_from_pointer(self, ctx, data, length):
        memref_t = self.mlir_buffer_type()
        desc_t = self.mlir_descriptor_type()

        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.mlir.undef : {desc_t}")

        for id, val in (
            ("0", data),
            ("1", data),
            ("2", ctx.constant(0, "i64")),
            ("3, 0", length),
            ("4, 0", ctx.constant(1, "i64")),
        ):
            fresh = ctx.new_ssa()
            ctx.exec(
                f"{ctx.feed}{fresh} = llvm.insertvalue {val}, {desc}[{id}] : {desc_t}"
            )
            desc = fresh

        buf = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buf} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {memref_t}"
        )
        return buf

    # load the length of the NumPy buffer
    def mlir_length(self, ctx: "MLIRContext", buf: MLIRBufferFields):
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        c0 = ctx.constant(0, "index")
        res = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{res} = memref.dim {buffer}, {c0} : {self.mlir_buffer_type()}"
        )
        return res

    # Convert index values to mlir index types
    def mlir_index(self, ctx: "MLIRContext", idx: "AssemblyExpression"):
        i = ctx(idx)
        t = mlir_type(idx.result_type)
        if t == "index":
            return i
        cast_ = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{cast_} = arith.index_cast {i} : {t} to index")
        return cast_

    # Load buffer values of NumPy buffer
    def mlir_load(
        self, ctx: "MLIRContext", buf: MLIRBufferFields, idx: "AssemblyExpression"
    ):
        if isinstance(self.element_type, TupleFType):
            packed_t = self._mlir_packed_type()
            ptr = self._mlir_tuple_ptr(ctx, buf, idx)
            packed = ctx.new_ssa()
            ctx.exec(f"{ctx.feed}{packed} = llvm.load {ptr} : !llvm.ptr -> {packed_t}")
            return self._mlir_repack_tuple(
                ctx, packed, packed_t, mlir_type(self.element_type)
            )
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        i = self.mlir_index(ctx, idx)
        c0 = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{c0} = memref.load {buffer}[{i}] : {self.mlir_buffer_type()}"
        )
        return c0

    # Store values in the buffer
    def mlir_store(
        self,
        ctx: "MLIRContext",
        buf: Any,
        idx: Any,
        value: Any = None,
    ):
        if isinstance(self.element_type, TupleFType):
            packed_t = self._mlir_packed_type()
            packed = self._mlir_repack_tuple(
                ctx, ctx(value), mlir_type(self.element_type), packed_t
            )
            ptr = self._mlir_tuple_ptr(ctx, buf, idx)
            ctx.exec(f"{ctx.feed}llvm.store {packed}, {ptr} : {packed_t}, !llvm.ptr")
            return
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {buf.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {self.mlir_buffer_type()}"
        )
        new = ctx(value)
        i = self.mlir_index(ctx, idx)
        ctx.exec(
            f"{ctx.feed}memref.store {new}, {buffer}[{i}] : {self.mlir_buffer_type()}"
        )

    # Resize the NumPy buffer and update its memref descriptor.
    def mlir_resize(self, ctx, buf: MLIRBufferFields, new_len):
        len_idx = mlir_cast_value(ctx, ctx(new_len), new_len.result_type, ftypes.intp)
        len = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{len} = arith.index_cast {len_idx} : index to i64")
        arr_ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{arr_ptr} = llvm.getelementptr {buf.obj}[0, 0] "
            f": (!llvm.ptr) -> !llvm.ptr, {self.llvm_callback_type()}"
        )

        cb_ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{cb_ptr} = llvm.getelementptr {buf.obj}[0, 3] "
            f": (!llvm.ptr) -> !llvm.ptr, {self.llvm_callback_type()}"
        )
        cb = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{cb} = llvm.load {cb_ptr} : !llvm.ptr -> !llvm.ptr")

        data = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{data} = llvm.call {cb}({arr_ptr}, {len}) "
            f": !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr"
        )

        buffer = self.mlir_memref_from_pointer(ctx, data, len)
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{desc} = builtin.unrealized_conversion_cast "
            f"{buffer} : {self.mlir_buffer_type()} to {desc_t}"
        )
        ctx.exec(f"{ctx.feed}llvm.store {desc}, {buf.box} : {desc_t}, !llvm.ptr")

    # Unpack the NumPy buffer and create its memeref descriptor
    def mlir_unpack(self, ctx: MLIRContext, _var_n, val):
        obj = ctx(val)

        data_ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{data_ptr} = llvm.getelementptr {obj}[0, 1] "
            f": (!llvm.ptr) -> !llvm.ptr, {self.llvm_callback_type()}"
        )
        data = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{data} = llvm.load {data_ptr} : !llvm.ptr -> !llvm.ptr")

        length_ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{length_ptr} = llvm.getelementptr {obj}[0, 2] "
            f": (!llvm.ptr) -> !llvm.ptr, {self.llvm_callback_type()}"
        )
        length = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{length} = llvm.load {length_ptr} : !llvm.ptr -> i64")

        buffer = self.mlir_memref_from_pointer(ctx, data, length)
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{desc} = builtin.unrealized_conversion_cast "
            f"{buffer} : {self.mlir_buffer_type()} to {desc_t}"
        )
        box = ctx.new_ssa()
        count = ctx.constant(1, "i64")
        ctx.exec(
            f"{ctx.feed}{box} = llvm.alloca {count} x {desc_t} : (i64) -> !llvm.ptr"
        )
        ctx.exec(f"{ctx.feed}llvm.store {desc}, {box} : {desc_t}, !llvm.ptr")

        return MLIRBufferFields(box, obj)

    # Update the buffer wrapper with the resized pointer and length.
    def mlir_repack(self, ctx: MLIRContext, _lhs, obj):
        memref_t = self.mlir_buffer_type()
        desc_t = self.mlir_descriptor_type()
        desc = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{desc} = llvm.load {obj.box} : !llvm.ptr -> {desc_t}")
        buffer = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{buffer} = builtin.unrealized_conversion_cast "
            f"{desc} : {desc_t} to {memref_t}"
        )

        len_ssa = self.mlir_length(ctx, obj)
        len_i64 = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{len_i64} = arith.index_cast {len_ssa} : index to i64")
        len_ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{len_ptr} = llvm.getelementptr {obj.obj}[0, 2] "
            f": (!llvm.ptr) -> !llvm.ptr, {self.llvm_callback_type()}"
        )
        ctx.exec(f"{ctx.feed}llvm.store {len_i64}, {len_ptr} : i64, !llvm.ptr")

        data_ssa = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{data_ssa} = "
            f"memref.extract_aligned_pointer_as_index {buffer} "
            f": {memref_t} -> index"
        )
        data_i64 = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{data_i64} = arith.index_cast {data_ssa} : index to i64")
        data_llvm = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{data_llvm} = llvm.inttoptr {data_i64} : i64 to !llvm.ptr")
        data_ptr = ctx.new_ssa()
        ctx.exec(
            f"{ctx.feed}{data_ptr} = llvm.getelementptr {obj.obj}[0, 1] "
            f": (!llvm.ptr) -> !llvm.ptr, {self.llvm_callback_type()}"
        )
        ctx.exec(f"{ctx.feed}llvm.store {data_llvm}, {data_ptr} : !llvm.ptr, !llvm.ptr")

    # Package the NumPy array, data pointer, length, and resize callback for MLIR
    def serialize_to_mlir(self, obj):
        self._normalize_layout(obj)
        result = MLIRNumpyBuffer(
            ctypes.py_object(obj),
            ctypes.c_void_p(obj.arr.ctypes.data),
            obj.arr.size,
            numpy_buffer_resize_callback,
        )
        return ctypes.cast(ctypes.pointer(result), ctypes.c_void_p)

    def deserialize_from_mlir(self, obj, mlir_buffer):
        # The resize callback updates the owning NumPy buffer directly.
        pass

    def construct_from_mlir(self, mlir_buffer):
        # zero copy
        result = ctypes.cast(mlir_buffer, ctypes.POINTER(MLIRNumpyBuffer))
        return result.contents.arr
