from __future__ import annotations

import json
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from finch.algebra import ftype
from finch.codegen import NumpyBuffer, NumpyBufferFType
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    OverrideTensor,
    SparseCOOLevel,
    SparseListLevel,
    element,
)

BINSPARSE_VERSION = "0.1.0"


class SwizzleTensor(OverrideTensor):
    __match_args__ = ("body", "dims")

    body: FiberTensor
    dims: tuple[int, ...]

    def __init__(self, body: FiberTensor, dims: tuple[int, ...]):
        self.body = body
        self.dims = tuple(dims)

    def __repr__(self):
        return f"SwizzleTensor(body={self.body}, dims={self.dims})"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(self.body.shape[d]) for d in self.dims)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def fill_value(self):
        return self.body.fill_value

    @property
    def element_type(self):
        return self.body.element_type

    @property
    def device(self):
        return self.body.device

    @property
    def ftype(self):
        return self.body.ftype

    def item(self):
        return self.body.item()

    def to_scipy(self):
        dense_arr = self.to_numpy()
        import scipy.sparse as sps

        return sps.coo_matrix(dense_arr)

    def to_numpy(self) -> np.ndarray:
        dense_arr, _, _ = tensor_to_npy(self)
        return dense_arr


def swizzle(tns: Any, *dims: Any) -> Any:
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        perm = tuple(dims[0])
    else:
        perm = tuple(dims)
    match tns:
        case SwizzleTensor(body, current_dims):
            new_dims = tuple(current_dims[d] for d in perm)
            if new_dims == tuple(range(body.ndim)):
                return body
            return SwizzleTensor(body, new_dims)
        case FiberTensor():
            if perm == tuple(range(tns.ndim)):
                return tns
            return SwizzleTensor(tns, perm)
        case _:
            if perm == tuple(range(tns.ndim)):
                return tns
            return SwizzleTensor(tns, perm)


bspread_tensor_lookup: dict[str, dict[str, Any]] = {
    "DVEC": {
        "level": {
            "level_desc": "dense",
            "rank": 1,
            "level": {"level_desc": "element"},
        }
    },
    "DMAT": {
        "level": {
            "level_desc": "dense",
            "rank": 1,
            "level": {
                "level_desc": "dense",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        }
    },
    "DMATR": {
        "level": {
            "level_desc": "dense",
            "rank": 1,
            "level": {
                "level_desc": "dense",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        }
    },
    "DMATC": {
        "transpose": [1, 0],
        "level": {
            "level_desc": "dense",
            "rank": 1,
            "level": {
                "level_desc": "dense",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        },
    },
    "CVEC": {
        "level": {
            "level_desc": "sparse",
            "rank": 1,
            "level": {"level_desc": "element"},
        }
    },
    "CSR": {
        "level": {
            "level_desc": "dense",
            "rank": 1,
            "level": {
                "level_desc": "sparse",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        }
    },
    "CSC": {
        "transpose": [1, 0],
        "level": {
            "level_desc": "dense",
            "rank": 1,
            "level": {
                "level_desc": "sparse",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        },
    },
    "DCSR": {
        "level": {
            "level_desc": "sparse",
            "rank": 1,
            "level": {
                "level_desc": "sparse",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        }
    },
    "DCSC": {
        "transpose": [1, 0],
        "level": {
            "level_desc": "sparse",
            "rank": 1,
            "level": {
                "level_desc": "sparse",
                "rank": 1,
                "level": {"level_desc": "element"},
            },
        },
    },
    "COO": {
        "level": {
            "level_desc": "sparse",
            "rank": 2,
            "level": {"level_desc": "element"},
        }
    },
    "COOR": {
        "level": {
            "level_desc": "sparse",
            "rank": 2,
            "level": {"level_desc": "element"},
        }
    },
    "COOC": {
        "transpose": [1, 0],
        "level": {
            "level_desc": "sparse",
            "rank": 2,
            "level": {"level_desc": "element"},
        },
    },
}

bspwrite_format_order = [
    "DVEC",
    "DMATR",
    "DMATC",
    "CVEC",
    "CSR",
    "CSC",
    "DCSR",
    "DCSC",
    "COOR",
    "COOC",
]

bspread_type_lookup = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float64,
    "bint8": np.bool_,
}

bspwrite_type_lookup = {
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bint8",
}


def bspread_check_version(version_str: str) -> None:
    parts = [int(p) for p in version_str.split(".")]
    supp = [int(p) for p in BINSPARSE_VERSION.split(".")]
    if parts[0] != supp[0] or parts[1] != supp[1] or parts[2] > supp[2]:
        raise ValueError(
            f"unsupported Binsparse version {version_str};"  # noqa: G001
            f"expected {supp[0]}.{supp[1]}.x <= {BINSPARSE_VERSION}"
        )


def _is_hdf5(f: Any) -> bool:
    try:
        import h5py

        return isinstance(f, (h5py.File, h5py.Group))
    except ImportError:
        return False


def bspread_header(f: Any) -> dict[str, Any]:
    if _is_hdf5(f):
        val = f.attrs["binsparse"]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        parsed = json.loads(val)
        return parsed if "binsparse" in parsed else {"binsparse": parsed}
    if isinstance(f, Mapping) and "binsparse" in f:
        val = f["binsparse"]
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8")
        parsed = json.loads(str(val))
        return parsed if "binsparse" in parsed else {"binsparse": parsed}
    if isinstance(f, (str, Path)):
        p = Path(f)
        if p.is_dir() or p.name.endswith(".bspnpy"):
            with (p / "binsparse.json").open("r", encoding="utf-8") as jf:
                parsed = json.load(jf)
                return parsed if "binsparse" in parsed else {"binsparse": parsed}
        if p.suffix.lower() in (".h5", ".hdf5"):
            import h5py

            with h5py.File(p, "r") as hf:
                return bspread_header(hf)
        if p.suffix.lower() == ".npz":
            with np.load(p, allow_pickle=False) as archive:
                return bspread_header(archive)
    raise TypeError(f"Cannot read Binsparse header from {type(f)}")


def bspwrite_header(f: Any, val_str: str) -> None:
    if _is_hdf5(f):
        f.attrs["binsparse"] = val_str
        return
    if isinstance(f, dict):
        f["binsparse"] = np.asarray(val_str)
        return
    if isinstance(f, (str, Path)):
        p = Path(f)
        if p.is_dir() or p.name.endswith(".bspnpy"):
            p.mkdir(parents=True, exist_ok=True)
            with (p / "binsparse.json").open("w", encoding="utf-8") as jf:
                jf.write(val_str)
            return
    raise TypeError(f"Cannot write Binsparse header to {type(f)}")


def bspread_vector(f: Any, key: str) -> np.ndarray:
    if _is_hdf5(f):
        return np.asarray(f[key][()])
    if isinstance(f, Mapping) and key in f:
        return np.asarray(f[key])
    if isinstance(f, (str, Path)):
        p = Path(f)
        if p.is_dir() or p.name.endswith(".bspnpy"):
            return np.load(p / f"{key}.npy", allow_pickle=False)
    raise KeyError(f"Vector {key!r} not found in container")


def bspwrite_vector(f: Any, key: str, val: np.ndarray) -> None:
    if _is_hdf5(f):
        if key in f:
            del f[key]
        f.create_dataset(key, data=val)
        return
    if isinstance(f, dict):
        f[key] = val
        return
    if isinstance(f, (str, Path)):
        p = Path(f)
        if p.is_dir() or p.name.endswith(".bspnpy"):
            p.mkdir(parents=True, exist_ok=True)
            np.save(p / f"{key}.npy", val)
            return
    raise TypeError(f"Cannot write vector {key!r} to container {type(f)}")


def bspread_data(
    f: Any, desc: dict[str, Any], key: str, dtype_str: str | None = None
) -> np.ndarray:
    if dtype_str is None:
        dtype_str = desc["data_types"][key]

    if dtype_str.startswith("iso[") and dtype_str.endswith("]"):
        inner = dtype_str[4:-1]
        data = bspread_data(f, desc, key, inner)
        if key == "values":
            n = int(desc["number_of_stored_values"])
        elif key == "fill_value":
            n = 1
        else:
            n = len(data)
        if n == 0:
            return np.empty(0, dtype=data.dtype)
        return np.ascontiguousarray(np.broadcast_to(data[:1], (n,)))
    if dtype_str.startswith("complex[") and dtype_str.endswith("]"):
        inner = dtype_str[8:-1]
        data = bspread_data(f, desc, key, inner)
        if data.dtype == np.float32:
            return np.ascontiguousarray(data).view(np.complex64)
        if data.dtype == np.float64:
            return np.ascontiguousarray(data).view(np.complex128)
        raise ValueError(f"unsupported complex component type: {data.dtype}")
    if dtype_str in bspread_type_lookup:
        target_dtype = bspread_type_lookup[dtype_str]
        raw = bspread_vector(f, key)
        if target_dtype == np.bool_:
            return np.asarray(raw, dtype=np.uint8).view(np.bool_)
        return np.asarray(raw, dtype=target_dtype)
    raise ValueError(f"unknown binsparse type: {dtype_str}")


def bspwrite_data(f: Any, desc: dict[str, Any], key: str, data: Any) -> None:
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.complexfloating):
        if arr.dtype == np.complex64:
            float_arr = np.ascontiguousarray(arr).view(np.float32)
            desc["data_types"][key] = "complex[float32]"
            bspwrite_vector(f, key, float_arr)
        elif arr.dtype == np.complex128:
            float_arr = np.ascontiguousarray(arr).view(np.float64)
            desc["data_types"][key] = "complex[float64]"
            bspwrite_vector(f, key, float_arr)
        else:
            raise ValueError(f"unsupported complex dtype: {arr.dtype}")
    elif arr.dtype == np.bool_:
        desc["data_types"][key] = "bint8"
        bspwrite_vector(f, key, arr.view(np.uint8))
    else:
        dt = np.dtype(arr.dtype)
        if dt not in bspwrite_type_lookup:
            raise ValueError(f"Cannot write {arr.dtype} to binsparse")
        desc["data_types"][key] = bspwrite_type_lookup[dt]
        bspwrite_vector(f, key, arr)


def binsparse_format(header: dict[str, Any]) -> dict[str, Any]:
    fmt = header.get("format")
    if fmt == "custom":
        return header["custom"]
    if fmt in bspread_tensor_lookup:
        return bspread_tensor_lookup[fmt]
    raise ValueError(f"Unknown binsparse format: {fmt}")


def count_stored_values(lvl: Any) -> int:
    match lvl:
        case ElementLevel():
            val = lvl.val.arr if hasattr(lvl.val, "arr") else np.asarray(lvl.val)
            return int(len(val))
        case DenseLevel(child_lvl):
            return count_stored_values(child_lvl)
        case SparseListLevel(child_lvl):
            return count_stored_values(child_lvl)
        case SparseCOOLevel(child_lvl):
            return count_stored_values(child_lvl)
        case _:
            return 0


def bspread(f: Any) -> Any:
    if isinstance(f, (str, Path)):
        p = Path(f)
        if p.suffix.lower() in (".h5", ".hdf5"):
            import h5py

            with h5py.File(p, "r") as hf:
                return bspread(hf)
        if p.suffix.lower() == ".npz":
            with np.load(p, allow_pickle=False) as archive:
                return bspread(archive)
        if p.is_dir() or p.name.endswith(".bspnpy"):
            f = p

    desc = bspread_header(f)["binsparse"]
    bspread_check_version(desc["version"])

    if desc.get("structure", "general") != "general":
        raise ValueError(f"unsupported binsparse structure: {desc.get('structure')}")

    fmt = dict(binsparse_format(desc))
    ndim = len(desc["shape"])
    transpose = tuple(fmt.get("transpose", range(ndim)))
    stored_shape = tuple(desc["shape"][d] for d in transpose)

    lvl = bspread_level(f, desc, fmt["level"], stored_shape, depth=0)
    tns = FiberTensor(lvl)

    if transpose != tuple(range(ndim)):
        inv_transpose = tuple(transpose.index(i) for i in range(ndim))
        tns = swizzle(tns, *inv_transpose)

    return tns


def bspread_level(
    f: Any,
    desc: dict[str, Any],
    fmt_level: dict[str, Any],
    stored_shape: tuple[int, ...],
    depth: int = 0,
) -> Any:
    level_desc = fmt_level["level_desc"]
    match level_desc:
        case "element":
            val_arr = bspread_data(f, desc, "values")
            if "fill_value" in desc.get("data_types", {}):
                fill_val = bspread_data(f, desc, "fill_value")[0]
            else:
                fill_val = val_arr.dtype.type(0) if val_arr.size > 0 else np.float64(0)
            elem_ftype = element(
                fill_val,
                ftype(val_arr.dtype),
                ftype(np.intp),
                NumpyBufferFType,
            )
            return ElementLevel(elem_ftype, NumpyBuffer(val_arr))
        case "dense":
            rank = fmt_level["rank"]
            shape = stored_shape[depth : depth + rank]
            child_lvl = bspread_level(
                f, desc, fmt_level["level"], stored_shape, depth + rank
            )
            lvl = child_lvl
            for s in reversed(shape):
                lvl = DenseLevel(lvl, np.intp(s))
            return lvl
        case "sparse":
            rank = fmt_level["rank"]
            shape = stored_shape[depth : depth + rank]
            child_lvl = bspread_level(
                f, desc, fmt_level["level"], stored_shape, depth + rank
            )
            if depth > 0:
                ptr = bspread_data(f, desc, f"pointers_to_{depth}")
            else:
                first_idx = bspread_data(f, desc, f"indices_{depth}")
                ptr = np.array([0, len(first_idx)], dtype=np.intp)
            ptr_arr = np.asarray(ptr, dtype=np.intp)
            if rank == 1:
                idx_arr = np.asarray(
                    bspread_data(f, desc, f"indices_{depth}"), dtype=np.intp
                )
                return SparseListLevel(
                    child_lvl,
                    np.intp(shape[0]),
                    NumpyBuffer(ptr_arr),
                    NumpyBuffer(idx_arr),
                )
            tbl = tuple(
                NumpyBuffer(
                    np.asarray(
                        bspread_data(f, desc, f"indices_{depth + r}"), dtype=np.intp
                    )
                )
                for r in range(rank)
            )
            return SparseCOOLevel(
                child_lvl, tuple(np.intp(s) for s in shape), NumpyBuffer(ptr_arr), tbl
            )
        case _:
            raise ValueError(f"Unknown level descriptor: {level_desc}")


def bspwrite_level(
    f: Any, desc: dict[str, Any], fmt: dict[str, Any], lvl: Any, depth: int = 0
) -> None:
    match lvl:
        case ElementLevel():
            fmt["level_desc"] = "element"
            val_arr = lvl.val.arr if hasattr(lvl.val, "arr") else np.asarray(lvl.val)
            fill_val = lvl.ftype.fill_value
            if hasattr(fill_val, "value"):
                fill_val = fill_val.value
            bspwrite_data(f, desc, "values", val_arr)
            fill_arr = np.array(
                [fill_val],
                dtype=val_arr.dtype if val_arr.size > 0 else np.asarray(fill_val).dtype,
            )
            bspwrite_data(f, desc, "fill_value", fill_arr)
        case DenseLevel(child_lvl):
            fmt["level_desc"] = "dense"
            fmt["rank"] = 1
            fmt["level"] = {}
            bspwrite_level(f, desc, fmt["level"], child_lvl, depth + 1)
        case SparseListLevel(child_lvl, _, ptr, idx):
            fmt["level_desc"] = "sparse"
            fmt["rank"] = 1
            ptr_arr = ptr.arr if hasattr(ptr, "arr") else np.asarray(ptr)
            idx_arr = idx.arr if hasattr(idx, "arr") else np.asarray(idx)
            if depth > 0:
                bspwrite_data(f, desc, f"pointers_to_{depth}", ptr_arr)
            bspwrite_data(f, desc, f"indices_{depth}", idx_arr)
            fmt["level"] = {}
            bspwrite_level(f, desc, fmt["level"], child_lvl, depth + 1)
        case SparseCOOLevel(child_lvl, coo_shape, ptr, tbl):
            rank = len(coo_shape)
            fmt["level_desc"] = "sparse"
            fmt["rank"] = rank
            ptr_arr = ptr.arr if hasattr(ptr, "arr") else np.asarray(ptr)
            if depth > 0:
                bspwrite_data(f, desc, f"pointers_to_{depth}", ptr_arr)
            for r in range(rank):
                tbl_r = tbl[r].arr if hasattr(tbl[r], "arr") else np.asarray(tbl[r])
                bspwrite_data(f, desc, f"indices_{depth + r}", tbl_r)
            fmt["level"] = {}
            bspwrite_level(f, desc, fmt["level"], child_lvl, depth + rank)
        case _:
            raise TypeError(f"Unsupported level type: {type(lvl)}")


def find_format_alias(custom: dict[str, Any]) -> str | None:
    for alias_name in bspwrite_format_order:
        target = bspread_tensor_lookup[alias_name]
        if (
            custom.get("transpose") == target.get("transpose")
            and custom["level"] == target["level"]
        ):
            return alias_name
    return None


def bspwrite_tensor(
    f: Any, arr: Any, attrs: dict[str, Any] | None = None, alias: bool | None = None
) -> None:
    match arr:
        case SwizzleTensor(body, dims):
            transpose = [0] * len(dims)
            for i, d in enumerate(dims):
                transpose[d] = i
            root_lvl = body.lvl
            logical_shape = [int(s) for s in arr.shape]
        case FiberTensor(lvl):
            root_lvl = lvl
            logical_shape = [int(s) for s in arr.shape]
            transpose = list(range(len(logical_shape)))
        case _:
            raise TypeError(f"Unsupported tensor type for bspwrite: {type(arr)}")

    custom: dict[str, Any] = {"level": {}}
    desc: dict[str, Any] = {
        "version": BINSPARSE_VERSION,
        "format": "custom",
        "shape": logical_shape,
        "number_of_stored_values": count_stored_values(root_lvl),
        "fill": True,
        "data_types": {},
        "custom": custom,
    }
    if attrs:
        desc["attrs"] = attrs
    if transpose != list(range(len(logical_shape))):
        custom["transpose"] = transpose

    bspwrite_level(f, desc, custom["level"], root_lvl, 0)

    if alias is not False:
        matched = find_format_alias(custom)
        if matched is not None:
            desc["format"] = matched
            del desc["custom"]
        else:
            desc["format"] = "custom"
    else:
        desc["format"] = "custom"

    bspwrite_header(f, json.dumps({"binsparse": desc}, indent=2, sort_keys=True))


def bspwrite(
    target: Any,
    arr: Any,
    attrs: dict[str, Any] | None = None,
    alias: bool | None = None,
) -> None:
    if isinstance(target, (str, Path)):
        p = Path(target)
        if p.suffix.lower() in (".h5", ".hdf5"):
            import h5py

            with h5py.File(p, "w") as hf:
                bspwrite_tensor(hf, arr, attrs=attrs, alias=alias)
            return
        if p.suffix.lower() == ".npz":
            archive: dict[str, Any] = {}
            bspwrite_tensor(archive, arr, attrs=attrs, alias=alias)
            np.savez(p, **archive)
            return
        if p.is_dir() or p.name.endswith(".bspnpy"):
            p.mkdir(parents=True, exist_ok=True)
            bspwrite_tensor(p, arr, attrs=attrs, alias=alias)
            return
    bspwrite_tensor(target, arr, attrs=attrs, alias=alias)


def fread(filename: str | Path) -> Any:
    p = Path(filename)
    suffix = p.suffix.lower()
    if suffix in (".h5", ".hdf5") or p.name.endswith(".bspnpy") or suffix == ".npz":
        return bspread(p)
    raise ValueError(f"Unknown file extension for {filename}")


def fwrite(filename: str | Path, tns: Any) -> None:
    p = Path(filename)
    suffix = p.suffix.lower()
    if suffix in (".h5", ".hdf5") or p.name.endswith(".bspnpy") or suffix == ".npz":
        bspwrite(p, tns)
        return
    raise ValueError(f"Unknown file extension for {filename}")


def finch_tensor(
    dense: np.ndarray, pat: np.ndarray, fill_value: Any, header: dict[str, Any]
) -> Any:
    fmt = binsparse_format(header)
    transpose = tuple(fmt.get("transpose", range(dense.ndim)))

    if transpose != tuple(range(dense.ndim)):
        stored = np.transpose(dense, transpose)
        stored_pat = np.transpose(pat, transpose)
        inv_transpose = tuple(transpose.index(i) for i in range(len(transpose)))
    else:
        stored = dense
        stored_pat = pat
        inv_transpose = None

    if stored.ndim == 0:
        val_arr = (
            np.array([stored.item()], dtype=stored.dtype)
            if bool(stored_pat)
            else np.empty(0, dtype=stored.dtype)
        )
        elem_ftype = element(
            fill_value,
            ftype(stored.dtype),
            ftype(np.intp),
            NumpyBufferFType,
        )
        lvl = ElementLevel(elem_ftype, NumpyBuffer(val_arr))
        return FiberTensor(lvl)

    coords = [tuple(int(x) for x in c) for c in np.argwhere(stored_pat)]
    lvl = finch_level(fmt["level"], stored, coords, [()], 0, fill_value)
    tns = FiberTensor(lvl)
    if inv_transpose is not None:
        tns = swizzle(tns, *inv_transpose)
    return tns


def finch_level(
    fmt: dict[str, Any],
    stored: np.ndarray,
    coords: list[tuple[int, ...]],
    parents: list[tuple[int, ...]],
    depth: int,
    fill_value: Any,
) -> Any:
    level_desc = fmt["level_desc"]
    match level_desc:
        case "element":
            val_list = [stored[p] for p in parents]
            val_arr = np.asarray(val_list, dtype=stored.dtype)
            elem_ftype = element(
                fill_value,
                ftype(stored.dtype),
                ftype(np.intp),
                NumpyBufferFType,
            )
            return ElementLevel(elem_ftype, NumpyBuffer(val_arr))
        case "dense":
            rank = fmt["rank"]
            shape = stored.shape[depth : depth + rank]
            children = []
            for p in parents:
                children.extend(
                    (*p, *suffix) for suffix in product(*(range(s) for s in shape))
                )
            child_lvl = finch_level(
                fmt["level"], stored, coords, children, depth + rank, fill_value
            )
            lvl = child_lvl
            for s in reversed(shape):
                lvl = DenseLevel(lvl, np.intp(s))
            return lvl
        case "sparse":
            rank = fmt["rank"]
            shape = stored.shape[depth : depth + rank]
            children = []
            ptr = [0]
            for p in parents:
                p_len = len(p)
                seen = set()
                for c in coords:
                    if c[:p_len] == p:
                        suffix = c[depth : depth + rank]
                        if suffix not in seen:
                            seen.add(suffix)
                            children.append(c[: depth + rank])
                ptr.append(len(children))
            child_lvl = finch_level(
                fmt["level"], stored, coords, children, depth + rank, fill_value
            )
            ptr_arr = np.array(ptr, dtype=np.intp)
            if rank == 1:
                idx_arr = np.array([c[-1] for c in children], dtype=np.intp)
                return SparseListLevel(
                    child_lvl,
                    np.intp(shape[0]),
                    NumpyBuffer(ptr_arr),
                    NumpyBuffer(idx_arr),
                )
            tbl_arrs = tuple(
                NumpyBuffer(np.array([c[depth + r] for c in children], dtype=np.intp))
                for r in range(rank)
            )
            return SparseCOOLevel(
                child_lvl,
                tuple(np.intp(s) for s in shape),
                NumpyBuffer(ptr_arr),
                tbl_arrs,
            )
        case _:
            raise ValueError(f"Unknown level descriptor: {level_desc}")


def level_to_coo(
    lvl: Any, parents: list[tuple[int, ...]], shape: tuple[int, ...], depth: int = 0
) -> list[tuple[tuple[int, ...], Any]]:
    match lvl:
        case ElementLevel():
            val_arr = lvl.val.arr if hasattr(lvl.val, "arr") else np.asarray(lvl.val)
            return list(zip(parents, val_arr, strict=True))
        case DenseLevel(child_lvl, dimension):
            dim = int(dimension)
            children = []
            for p in parents:
                children.extend((*p, i) for i in range(dim))
            return level_to_coo(child_lvl, children, shape, depth + 1)
        case SparseListLevel(child_lvl, _, ptr, idx):
            ptr_arr = ptr.arr if hasattr(ptr, "arr") else np.asarray(ptr)
            idx_arr = idx.arr if hasattr(idx, "arr") else np.asarray(idx)
            children = []
            for p_idx, p in enumerate(parents):
                start = ptr_arr[p_idx]
                end = ptr_arr[p_idx + 1]
                for k in range(start, end):
                    children.append((*p, int(idx_arr[k])))
            return level_to_coo(child_lvl, children, shape, depth + 1)
        case SparseCOOLevel(child_lvl, coo_shape, ptr, tbl):
            ptr_arr = ptr.arr if hasattr(ptr, "arr") else np.asarray(ptr)
            tbl_arrs = [t.arr if hasattr(t, "arr") else np.asarray(t) for t in tbl]
            rank = len(coo_shape)
            children = []
            for p_idx, p in enumerate(parents):
                start = ptr_arr[p_idx]
                end = ptr_arr[p_idx + 1]
                for k in range(start, end):
                    coord = tuple(int(tbl_arrs[r][k]) for r in range(rank))
                    children.append((*p, *coord))
            return level_to_coo(child_lvl, children, shape, depth + rank)
        case _:
            raise TypeError(f"Unsupported level type: {type(lvl)}")


def tensor_to_npy(tns: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    match tns:
        case SwizzleTensor(body, dims):
            stored_dense, stored_pat, fill_val = tensor_to_npy(body)
            dense = np.transpose(stored_dense, dims)
            pat = np.transpose(stored_pat, dims)
            return dense, pat, fill_val
        case FiberTensor(lvl):
            fill_val = tns.fill_value
            if hasattr(fill_val, "value"):
                fill_val = fill_val.value
            elem_t = tns.element_type
            dt = elem_t.dtype if hasattr(elem_t, "dtype") else np.dtype(type(fill_val))
            shape = tuple(int(s) for s in tns.shape)
            if not shape:
                values = lvl.val.arr if hasattr(lvl.val, "arr") else np.asarray(lvl.val)
                if len(values) > 0:
                    dense = np.asarray(values[0], dtype=dt)
                    pat = np.asarray(True)
                else:
                    dense = np.asarray(fill_val, dtype=dt)
                    pat = np.asarray(False)
                return dense, pat, fill_val

            entries = level_to_coo(lvl, [()], shape, 0)
            if entries:
                first_val = entries[0][1]
                dt = np.result_type(dt, np.asarray(first_val).dtype)
            dense = np.full(shape, fill_val, dtype=dt)
            pat = np.zeros(shape, dtype=bool)
            for coord, val in entries:
                dense[coord] = val
                pat[coord] = True
            return dense, pat, fill_val
        case _:
            raise TypeError(f"Unsupported tensor type: {type(tns)}")


def dense_array(tns: Any) -> np.ndarray:
    dense, _, _ = tensor_to_npy(tns)
    return dense


def pattern_array(tns: Any) -> np.ndarray:
    _, pat, _ = tensor_to_npy(tns)
    return pat


def get_layout(header: dict[str, Any]) -> tuple[tuple[int, ...], list[tuple[str, int]]]:
    fmt = binsparse_format(header)
    transpose = tuple(fmt.get("transpose", range(len(header["shape"]))))
    return transpose, get_levels(fmt["level"])


def get_levels(fmt: dict[str, Any]) -> list[tuple[str, int]]:
    kind = fmt["level_desc"]
    if kind == "element":
        return [(kind, 0)]
    rank = fmt["rank"]
    here = [(kind, 1)] * rank if kind == "dense" else [(kind, rank)]
    return here + get_levels(fmt["level"])


def match_header(
    path_or_file: Any, requested: dict[str, Any], rename_aliases: bool = True
) -> None:
    if isinstance(path_or_file, (str, Path)):
        p = Path(path_or_file)
        if p.suffix.lower() in (".h5", ".hdf5"):
            import h5py

            with h5py.File(p, "r+") as f:
                _match_header_container(f, requested, rename_aliases)
            return
        if p.suffix.lower() == ".npz":
            with np.load(p, allow_pickle=False) as archive:
                archive_dict = dict(archive)
            _match_header_container(archive_dict, requested, rename_aliases)
            np.savez(p, **archive_dict)
            return
    _match_header_container(path_or_file, requested, rename_aliases)


def _match_header_container(
    f: Any, requested: dict[str, Any], rename_aliases: bool
) -> None:
    desc_wrap = bspread_header(f)
    actual = desc_wrap["binsparse"]
    requested_custom = requested["format"] == "custom"
    actual_custom = actual["format"] == "custom"

    if (
        (actual_custom == requested_custom)
        and (requested_custom or rename_aliases)
        and get_layout(actual) == get_layout(requested)
    ):
        actual["custom"] = requested["custom"]

    for key, dtype in requested.get("data_types", {}).items():
        actual_dtype = actual.get("data_types", {}).get(key)
        if dtype == f"iso[{actual_dtype}]":
            data = bspread_vector(f, key)
            width = 2 if dtype.startswith("iso[complex[") else 1
            value = data[: min(width, len(data))]
            repeated = np.tile(value, len(data) // width)
            if not np.array_equal(data, repeated):
                raise ValueError(
                    f"{key} must have identical stored values to be written as {dtype}"
                )
            bspwrite_vector(f, key, value)
            actual["data_types"][key] = dtype

    bspwrite_header(f, json.dumps(desc_wrap, indent=2, sort_keys=True))
