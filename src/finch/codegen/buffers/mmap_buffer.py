from pathlib import Path

import numpy as np

from finch.codegen.buffers.numpy_buffer import NumpyBuffer


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
