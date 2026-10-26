import pytest

import finch.finch_logic as lgc
from finch import ffuncs

from .scripts.nodes import (
    create_asm_comprehensive_node,
    create_asm_dot_node,
    create_asm_if_node,
    create_log_simple_node,
    create_ntn_simple_node,
)


def test_log_printer(file_regression):
    prgm = create_log_simple_node()
    file_regression.check(str(prgm), extension=".txt")


@pytest.mark.parametrize(
    "nfused,names,expected",
    [
        (None, (), "scratch[]"),
        (None, ("time", "row", "column"), "scratch[time, row, column]"),
        (0, (), "scratch()[]"),
        (0, ("time", "row", "column"), "scratch()[time, row, column]"),
        (1, ("time", "row", "column"), "scratch(time)[row, column]"),
        (2, ("time", "row", "column"), "scratch(time, row)[column]"),
        (3, ("time", "row", "column"), "scratch(time, row, column)[]"),
    ],
)
def test_log_table_printer(nfused, names, expected):
    alias = lgc.HardAlias("scratch")
    if nfused is not None:
        alias = lgc.FusedAlias(alias, nfused)
    fields = tuple(map(lgc.Field, names))
    table = lgc.Table(alias, fields)
    assert str(table) == expected
    rhs = lgc.Table(lgc.HardAlias("input"), fields)
    input_text = f"input[{', '.join(names)}]"
    assert str(lgc.Query(table, rhs)) == f"{expected} = {input_text}"
    assert (
        str(lgc.QueryInto(table, lgc.Literal(ffuncs.add), rhs))
        == f"{expected} <<add>>= {input_text}"
    )


def test_ntn_printer(file_regression):
    prgm = create_ntn_simple_node()
    file_regression.check(str(prgm), extension=".txt")


def test_asm_printer_if(file_regression):
    prgm = create_asm_if_node()
    file_regression.check(str(prgm), extension=".txt")


def test_asm_printer_dot(file_regression):
    prgm = create_asm_dot_node()
    file_regression.check(str(prgm), extension=".txt")


def test_asm_printer_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    file_regression.check(str(prgm), extension=".txt")
