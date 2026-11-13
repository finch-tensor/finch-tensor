from pathlib import Path

import numpy as np

import numba

from finch.algebra import FType, ftype
from finch.codegen.buffers.numpy_buffer import NumpyBuffer, NumpyBufferFType
from finch.util import qual_str


class MMapBuffer(NumpyBuffer):
    """
    A NumpyBuffer whose data lives in a memory-mapped file.
    """

    def __init__(self, path: str | Path, dtype=np.float64, mode: str = "r"):
        if mode not in ("r", "r+", "c"):
            raise ValueError(f"mode must be 'r', 'r+' or 'c', got {mode}")
        self.path = Path(path)
        self.mode = mode
        super().__init__(np.memmap(self.path, dtype=dtype, mode=mode))

    @property
    def ftype(self):
        return MMapBufferFType(ftype(self.arr.dtype), readonly=self.mode == "r")

    def resize(self, new_length: int):
        if self.mode != "r+":
            raise ValueError(f"cannot resize a buffer mapped with mode {self.mode}")
        self.arr.flush()
        self.arr = np.memmap(
            self.path, dtype=self.arr.dtype, mode="r+", shape=(new_length,)
        )

    def __str__(self):
        arr_str = str(self.arr).replace("\n", "")
        return f"mmap_buf({arr_str})"

    def __repr__(self):
        path = str(self.path)
        return f"MMapBuffer({path!r}, dtype={self.arr.dtype}, mode={self.mode!r})"


class MMapBufferFType(NumpyBufferFType):
    """
    The ftype of an MMapBuffer.
    """

    def __init__(self, element_type: FType, readonly: bool = True):
        super().__init__(element_type)
        self.readonly = readonly

    def __eq__(self, other):
        return super().__eq__(other) and self.readonly == other.readonly

    def __hash__(self):
        return hash((self._element_type, self.readonly))

    def __str__(self):
        mode = "r" if self.readonly else "rw"
        return f"mmap_buf_t({qual_str(self._dtype.type)}, {mode})"

    def __repr__(self):
        return f"MMapBufferFType({self._element_type!r}, readonly={self.readonly})"

    def __call__(self, len: int = 0, element_type: FType | None = None):
        # a new buffer would need a file to map
        raise NotImplementedError("cannot allocate a new MMapBuffer from its ftype")

    def numba_jitclass_type(self) -> numba.types.Type:
        return numba.types.ListType(
            numba.types.Array(
                numba.from_dtype(self._dtype), 1, "C", readonly=self.readonly
            )
        )
