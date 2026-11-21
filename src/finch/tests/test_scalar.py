import pytest

import numpy as np

from finch.algebra import (
    DynamicFillFType,
    StaticFill,
    StaticFillFType,
    ffuncs,
    ftype,
    is_dynamic,
)
from finch.tensor import ConstantScalar, Scalar, ScalarFType


def test_a_scalar_keeps_a_dynamic_fill_apart_from_its_value():
    s = Scalar(np.float64(2.0))
    # The fill defaults to the value, and is a field of the scalar, so the value
    # stays out of the format.
    assert is_dynamic(s.ftype.fill_type)
    assert s.fill_value == 2.0
    assert [name for name, _ in s.ftype.struct_fields] == ["val", "fill"]
    s.val = np.float64(5.0)
    assert s.fill_value == 2.0


def test_a_static_fill_is_part_of_the_format():
    s = Scalar(np.float64(2.0)).with_fill(StaticFill(np.float64(0.0)))
    assert (s.val, s.fill_value) == (2.0, 0.0)
    assert s.ftype == ScalarFType(ftype(np.float64), StaticFillFType(np.float64(0.0)))
    assert [name for name, _ in s.ftype.struct_fields] == ["val"]
    assert not s.ftype.constant


def test_a_constant_scalar_is_marked_by_its_format():
    c = ConstantScalar(np.float64(3.0))
    static = Scalar(np.float64(3.0)).with_fill(StaticFill(np.float64(3.0)))
    # A constant and a writable scalar with the same fill compile apart.
    assert c.ftype.constant
    assert c.ftype != static.ftype
    assert c.with_fill(StaticFill(np.float64(3.0))) is c
    # Another fill makes it a writable scalar of the same value.
    other = c.with_fill(StaticFill(np.float64(0.0)))
    assert type(other) is Scalar
    assert (other.val, other.fill_value) == (3.0, 0.0)
    with pytest.raises(ValueError):
        c.declare(np.float64(0.0), ffuncs.add, ())


def test_declaring_a_scalar_resets_its_value_and_dynamic_fill():
    s = Scalar(np.float64(2.0))
    s.declare(1.0, ffuncs.add, ())
    assert (s.val, s.fill_value) == (1.0, 1.0)
    # Updates change the value, not the fill it was declared with.
    s.access((), ffuncs.add).increment(np.float64(4.0))
    assert (s.val, s.fill_value) == (5.0, 1.0)


def test_declaring_a_scalar_keeps_its_static_fill():
    s = Scalar(np.float64(2.0)).with_fill(StaticFill(np.float64(0.0)))
    s.declare(0.0, ffuncs.add, ())
    assert s.val == 0.0
    with pytest.raises(ValueError):
        s.declare(1.0, ffuncs.add, ())


@pytest.mark.parametrize(
    "fill",
    [StaticFillFType(np.float64(1.5)), DynamicFillFType(ftype(np.float64))],
    ids=["static", "dynamic"],
)
def test_a_constructed_scalar_round_trips_through_its_fields(fill):
    fmt = ScalarFType(ftype(np.float64), fill)
    s = fmt.construct(())
    assert s.ftype == fmt
    fields = [fmt.struct_getattr(s, name) for name, _ in fmt.struct_fields]
    t = fmt.from_fields(*fields)
    assert t.ftype == fmt
    assert (t.val, t.fill_value) == (s.val, s.fill_value)
