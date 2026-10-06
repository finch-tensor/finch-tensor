#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Finch Developers
# SPDX-License-Identifier: MIT

"""Persistent Finch server for the Binsparse compliance test suite."""

import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

import finch as ft
from finch.fileio.binsparse import (
    bspread_header,
    bspwrite_header,
    dense_array,
    finch_tensor,
    match_header,
    pattern_array,
)


def cmd_binsparse_to_npy(args: list[str]) -> None:
    if len(args) != 4:
        raise ValueError("binsparse_to_npy requires 4 args: tensor_in tensor_out pattern_out fill_value_out")
    tensor_in, tensor_out, pattern_out, fill_value_out = args
    tns = ft.bspread(tensor_in)
    np.save(tensor_out, dense_array(tns))
    np.save(pattern_out, pattern_array(tns))
    fill_val = tns.fill_value
    if hasattr(fill_val, "value"):
        fill_val = fill_val.value
    np.save(fill_value_out, np.asarray(fill_val))


def cmd_npy_to_binsparse(args: list[str]) -> None:
    if len(args) != 5:
        raise ValueError("npy_to_binsparse requires 5 args: tensor_in pattern_in fill_value_in header_in tensor_out")
    tensor_in, pattern_in, fill_value_in, header_in, tensor_out = args
    dense = np.load(tensor_in, allow_pickle=False)
    pat = np.load(pattern_in, allow_pickle=False)
    fill_arr = np.load(fill_value_in, allow_pickle=False)
    fill_value = fill_arr.item() if fill_arr.ndim == 0 else fill_arr.reshape(-1)[0]
    with Path(header_in).open("r", encoding="utf-8") as f:
        header = json.load(f)
    tns = finch_tensor(dense, pat, fill_value, header)
    ft.bspwrite(tensor_out, tns, alias=(header.get("format") != "custom"))
    match_header(tensor_out, header)


def cmd_binsparse_to_binsparse(args: list[str]) -> None:
    if len(args) != 2:
        raise ValueError("binsparse_to_binsparse requires 2 args: tensor_in tensor_out")
    tensor_in, tensor_out = args
    header = bspread_header(tensor_in)["binsparse"]
    tns = ft.bspread(tensor_in)
    ft.bspwrite(tensor_out, tns, alias=(header.get("format") != "custom"))
    match_header(tensor_out, header, rename_aliases=False)


COMMANDS = {
    "binsparse_to_npy": cmd_binsparse_to_npy,
    "npy_to_binsparse": cmd_npy_to_binsparse,
    "binsparse_to_binsparse": cmd_binsparse_to_binsparse,
}


def write_response(response_path: str, exit_code: int, error_msg: str = "") -> None:
    resp: dict[str, Any] = {"exit_code": exit_code}
    if error_msg:
        resp["error"] = error_msg
    tmp_path = Path(response_path + ".tmp")
    dest_path = Path(response_path)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(resp, f)
    tmp_path.replace(dest_path)


def handle_request(line: str) -> bool:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        return True

    cmd = req.get("cmd")
    if cmd is None:
        return True
    if cmd == "shutdown":
        return False

    args = req.get("args", [])
    response_path = req.get("response")
    if not response_path:
        return True

    if cmd not in COMMANDS:
        write_response(response_path, 1, f"unknown command: {cmd}")
        return True

    try:
        COMMANDS[cmd](args)
        write_response(response_path, 0)
    except Exception as e:
        write_response(response_path, 1, str(e) or traceback.format_exc())

    return True


def server_main() -> None:
    if len(sys.argv) < 2:
        print("usage: finch_server.py <fifo_path> [ready_file]", file=sys.stderr)
        sys.exit(2)

    fifo_path = Path(sys.argv[1])
    ready_file = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    if ready_file is not None:
        ready_file.write_text("ready\n", encoding="utf-8")

    while True:
        with fifo_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                keep_running = handle_request(stripped)
                if not keep_running:
                    return


if __name__ == "__main__":
    server_main()
