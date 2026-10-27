"""Binsparse compliance-test executables which round-trip through Finch."""

import sys

import numpy as np

from binsparse import CustomTensor, load_binsparse, save_binsparse
from reference_cli.converters import (
    _load_fill_value,
    _load_header,
    _npy_from_tensor,
    _tensor_from_npy,
)

import finch as ft


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return 2
    operation, *paths = args
    if operation == "npy_to_binsparse" and len(paths) == 5:
        tensor_in, pattern_in, fill_in, header_in, tensor_out = paths
        header = _load_header(header_in)
        tensor = _tensor_from_npy(
            np.load(tensor_in, allow_pickle=False),
            np.asarray(np.load(pattern_in, allow_pickle=False), dtype=bool),
            _load_fill_value(fill_in),
            header,
        )
        save_binsparse(
            ft.to_binsparse(ft.from_binsparse(tensor)),
            tensor_out,
            alias=header["format"] != "custom",
            header=header,
        )
        return 0
    if operation == "binsparse_to_npy" and len(paths) == 4:
        tensor_in, tensor_out, pattern_out, fill_out = paths
        tensor = ft.to_binsparse(ft.from_binsparse(load_binsparse(tensor_in)))
        dense, pattern, fill = _npy_from_tensor(tensor)
        np.save(tensor_out, dense)
        np.save(pattern_out, pattern)
        np.save(fill_out, np.asarray(fill))
        return 0
    if operation == "binsparse_to_binsparse" and len(paths) == 2:
        tensor_in, tensor_out = paths
        tensor = load_binsparse(tensor_in)
        save_binsparse(
            ft.to_binsparse(ft.from_binsparse(tensor)),
            tensor_out,
            alias=not isinstance(tensor, CustomTensor),
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
