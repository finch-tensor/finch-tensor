import math

import pytest

import numpy as np

import finch
from finch.algebra import (
    DynamicFill,
    TupleFType,
    cansplitpush,
    ffuncs,
    ftype,
    init_value,
    is_annihilator,
    is_associative,
    is_distributive,
    is_idempotent,
    is_identity,
    promote_type,
    repeat_operator,
)
from finch.algebra.ftypes import FDType, NamedTupleFType, none_, np_dtype


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (finch.bool, np.bool_),
        (finch.int8, np.int8),
        (finch.int16, np.int16),
        (finch.int32, np.int32),
        (finch.int64, np.int64),
        (finch.uint8, np.uint8),
        (finch.uint16, np.uint16),
        (finch.uint32, np.uint32),
        (finch.uint64, np.uint64),
        (finch.float16, np.float16),
        (finch.float32, np.float32),
        (finch.float64, np.float64),
        (finch.complex64, np.complex64),
        (finch.complex128, np.complex128),
        (finch.bool_, bool),
        (finch.int_, int),
        (finch.float_, float),
        (finch.complex_, complex),
        (finch.str_, str),
        (none_, type(None)),
    ],
)
def test_numpy_dtype(dtype: FDType, expected):
    assert dtype.dtype == np.dtype(expected)
    assert np_dtype(dtype) == np.dtype(expected)
    assert np.dtype(dtype) == np.dtype(expected)


def test_tuple_numpy_dtype_roundtrip():
    dtype = TupleFType.from_tuple(
        (finch.int32, TupleFType.from_tuple((finch.float64, finch.bool)))
    )
    expected = np.dtype(
        [
            ("element_0", np.int32),
            ("element_1", [("element_0", np.float64), ("element_1", np.bool_)]),
        ]
    )
    assert dtype.dtype == expected
    assert np.dtype(dtype) == expected
    assert np_dtype(dtype) == expected
    assert finch.ftype(np_dtype(dtype)) == dtype
    assert np_dtype(TupleFType.from_tuple(())) == np.dtype([])


def test_numpy_dtype_custom_fdtype():
    class CustomDType(FDType):
        @property
        def dtype(self) -> np.dtype:
            return np.dtype(np.int32)

        def __eq__(self, other):
            return type(self) is type(other)

        def __hash__(self):
            return hash(type(self))

        def __call__(self, val):
            return self.dtype.type(val)

    dtype = CustomDType()
    assert np_dtype(dtype) == np.dtype(np.int32)
    assert np_dtype(TupleFType.from_tuple((dtype,))) == np.dtype(
        [("element_0", np.int32)]
    )


def test_numpy_dtype_rejects_non_data_types():
    dtype = NamedTupleFType("Point", [("x", finch.float64)])
    with pytest.raises(TypeError, match="Unsupported NumPy dtype"):
        np_dtype(dtype)
    with pytest.raises(TypeError, match="Unsupported NumPy dtype"):
        np_dtype(TupleFType.from_tuple((dtype,)))


def test_algebra_selected():
    assert is_distributive(ffuncs.mul.ftype, ffuncs.add.ftype)
    assert is_distributive(ffuncs.mul.ftype, ffuncs.sub.ftype)
    assert is_distributive(ffuncs.and_.ftype, ffuncs.or_.ftype)
    assert is_distributive(ffuncs.and_.ftype, ffuncs.xor.ftype)
    assert is_distributive(ffuncs.or_.ftype, ffuncs.and_.ftype)
    assert is_distributive(ffuncs.logical_and.ftype, ffuncs.logical_or.ftype)
    assert is_distributive(ffuncs.logical_and.ftype, ffuncs.logical_xor.ftype)
    assert is_distributive(ffuncs.logical_or.ftype, ffuncs.logical_and.ftype)
    assert is_annihilator(ffuncs.add.ftype, math.inf)
    assert is_annihilator(ffuncs.mul.ftype, 0)
    assert is_annihilator(ffuncs.or_.ftype, -1)
    assert is_annihilator(ffuncs.and_.ftype, 0)
    assert is_annihilator(ffuncs.logaddexp.ftype, math.inf)
    assert is_annihilator(ffuncs.logical_or.ftype, True)
    assert is_annihilator(ffuncs.logical_and.ftype, False)
    assert is_identity(ffuncs.add.ftype, 0)
    assert is_identity(ffuncs.mul.ftype, 1)
    assert is_identity(ffuncs.or_.ftype, False)
    assert is_identity(ffuncs.and_.ftype, -1)
    assert is_identity(ffuncs.truediv.ftype, 1)
    assert is_identity(ffuncs.lshift.ftype, 0)
    assert is_identity(ffuncs.rshift.ftype, 0)
    assert is_identity(ffuncs.pow.ftype, 1)
    assert is_identity(ffuncs.truediv.ftype, 1)
    assert is_identity(ffuncs.logaddexp.ftype, -math.inf)
    assert is_identity(ffuncs.logical_or.ftype, False)
    assert is_identity(ffuncs.logical_and.ftype, True)
    assert is_identity(ffuncs.max.ftype, -math.inf)
    assert is_identity(ffuncs.min.ftype, math.inf)
    assert is_identity(ffuncs.choose(0).ftype, 0)
    assert is_associative(ffuncs.add.ftype)
    assert is_associative(ffuncs.mul.ftype)
    assert is_associative(ffuncs.choose(0).ftype)
    assert is_associative(ffuncs.logical_and.ftype)
    assert is_associative(ffuncs.logical_xor.ftype)
    assert is_associative(ffuncs.logical_or.ftype)
    assert is_associative(ffuncs.logaddexp.ftype)
    assert init_value(ffuncs.and_.ftype, finch.bool) is np.True_
    assert init_value(ffuncs.or_.ftype, finch.bool) is np.False_
    assert init_value(ffuncs.xor.ftype, finch.bool) is np.False_
    assert init_value(ffuncs.logaddexp.ftype, finch.float64) == -math.inf
    assert init_value(ffuncs.logical_and.ftype, finch.bool_) is True
    assert init_value(ffuncs.logical_or.ftype, finch.bool_) is False
    assert init_value(ffuncs.logical_xor.ftype, finch.bool_) is False
    assert is_idempotent(ffuncs.and_.ftype)
    assert is_idempotent(ffuncs.or_.ftype)
    assert is_idempotent(ffuncs.logical_and.ftype)
    assert is_idempotent(ffuncs.logical_or.ftype)
    assert is_idempotent(ffuncs.min.ftype)
    assert is_idempotent(ffuncs.max.ftype)
    assert is_idempotent(ffuncs.minby.ftype)
    assert is_idempotent(ffuncs.maxby.ftype)
    assert is_idempotent(ffuncs.add.ftype) is False
    assert is_idempotent(ffuncs.mul.ftype) is False
    assert is_idempotent(ffuncs.xor.ftype) is False
    assert is_idempotent(ffuncs.logical_xor.ftype) is False
    assert is_idempotent(ffuncs.logaddexp.ftype) is False
    assert repeat_operator(ffuncs.add.ftype) is ffuncs.mul
    assert repeat_operator(ffuncs.mul.ftype) is ffuncs.pow
    assert repeat_operator(ffuncs.and_.ftype) is None
    assert cansplitpush(ffuncs.add.ftype) is True
    assert cansplitpush(ffuncs.and_.ftype) is False
    assert ffuncs.choose(0)(0, 2, 3) == 2
    assert ffuncs.choose(0)(0, 0) == 0
    assert ffuncs.choose(np.nan)(np.nan, 4.0) == 4.0
    assert ffuncs.minby((1, 10), (2, 20)) == (1, 10)
    assert ffuncs.minby((2, 10), (1, 20)) == (1, 20)
    assert ffuncs.minby((1, 10), (1, 20)) == (1, 10)
    assert ffuncs.maxby((2, 10), (1, 20)) == (2, 10)
    assert ffuncs.maxby((1, 10), (2, 20)) == (2, 20)
    assert ffuncs.maxby((1, 10), (1, 20)) == (1, 20)
    assert ffuncs.last((1, 2, 3)) == 3
    assert ffuncs.scaled_power(2.0) is ffuncs.scaled_square
    assert ffuncs.add_scaled_power(2.0) is ffuncs.add_scaled_square
    assert ffuncs.root_scaled_power(2.0) is ffuncs.root_scaled_square
    assert ffuncs.scaled_square(np.float64(0.0)) == (np.float64(0.0), 0.0)
    assert ffuncs.scaled_square(np.float64(3.0)) == (np.float64(1.0), 3.0)
    scaled_sum = ffuncs.add_scaled_square((1.0, 3.0), (1.0, 4.0))
    assert scaled_sum == (1.5625, 4.0)
    assert ffuncs.root_scaled_square(scaled_sum) == 5.0
    scaled_square_nan = ffuncs.add_scaled_square((0.0, 0.0), (1.0, math.nan))
    assert math.isnan(scaled_square_nan[0])
    assert math.isnan(scaled_square_nan[1])
    scaled_power_nan = ffuncs.add_scaled_power(3.0)((0.0, 0.0), (1.0, math.nan))
    assert math.isnan(scaled_power_nan[0])
    assert math.isnan(scaled_power_nan[1])
    scaled_negative_zero = ffuncs.scaled_negative_power(-2.0)(np.float64(0.0))
    assert math.isinf(scaled_negative_zero[0])
    assert scaled_negative_zero[1] == 0.0
    scaled_negative_sum = ffuncs.add_scaled_negative_power(-2.0)((1.0, 2.0), (1.0, 4.0))
    assert scaled_negative_sum == (1.25, 2.0)
    assert math.isclose(
        ffuncs.root_scaled_negative_power(-2.0)(scaled_negative_sum),
        2.0 / math.sqrt(1.25),
    )
    scaled_negative_nan = ffuncs.add_scaled_negative_power(-2.0)(
        (1.0, 2.0), (1.0, math.nan)
    )
    assert math.isnan(scaled_negative_nan[0])
    assert math.isnan(scaled_negative_nan[1])
    assert (
        ffuncs.root_scaled_negative_power(-2.0)(
            ffuncs.add_scaled_negative_power(-2.0)((1.0, 2.0), (math.inf, 0.0))
        )
        == 0.0
    )


def test_python_scalar_promotion_uses_weak_bottom():
    assert promote_type(finch.bool, finch.bool_) == finch.bool
    assert promote_type(finch.bool_, finch.bool) == finch.bool
    assert promote_type(finch.int8, finch.int_) == finch.int8
    assert promote_type(finch.int_, finch.int8) == finch.int8
    assert promote_type(finch.int32, finch.int_) == finch.int32
    assert promote_type(finch.int_, finch.int32) == finch.int32
    assert promote_type(finch.uint8, finch.int_) == finch.uint8
    assert promote_type(finch.int_, finch.uint8) == finch.uint8
    assert promote_type(finch.int64, finch.bool_) == finch.int64
    assert promote_type(finch.bool_, finch.int64) == finch.int64
    assert promote_type(finch.float32, finch.int_) == finch.float32
    assert promote_type(finch.int_, finch.float32) == finch.float32
    assert promote_type(finch.float32, finch.float_) == finch.float32
    assert promote_type(finch.float_, finch.float32) == finch.float32
    assert promote_type(finch.complex64, finch.float_) == finch.complex64
    assert promote_type(finch.float_, finch.complex64) == finch.complex64
    assert promote_type(finch.complex64, finch.complex_) == finch.complex64
    assert promote_type(finch.complex_, finch.complex64) == finch.complex64
    tuple_type = TupleFType.from_tuple((finch.int32, finch.float32))
    promoted_tuple_type = TupleFType.from_tuple((finch.int64, finch.float32))
    assert isinstance(tuple_type, FDType)
    assert (
        promote_type(
            tuple_type,
            TupleFType.from_tuple((finch.int64, finch.int_)),
        )
        == promoted_tuple_type
    )
    assert (
        ffuncs.where.ftype.return_type(
            finch.bool,
            tuple_type,
            TupleFType.from_tuple((finch.int64, finch.int_)),
        )
        == promoted_tuple_type
    )
    assert (
        ffuncs.choose((0, 0)).ftype.return_type(
            tuple_type,
            TupleFType.from_tuple((finch.int64, finch.int_)),
        )
        == promoted_tuple_type
    )


def test_ftype_recognizes_numpy_dtype_aliases():
    int_long = finch.int32 if np.dtype(np.long) == np.dtype(np.int32) else finch.int64
    uint_long = (
        finch.uint32 if np.dtype(np.ulong) == np.dtype(np.uint32) else finch.uint64
    )
    uintp = finch.uint32 if np.uintp == np.uint32 else finch.uint64
    cases = [
        (np.long, int_long),
        (np.ulong, uint_long),
        (np.intp, finch.intp),
        (np.uintp, uintp),
        (np.longlong, finch.int64),
        (np.ulonglong, finch.uint64),
        (np.float16, finch.float16),
    ]

    for np_type, finch_type in cases:
        assert finch.ftype(np_type) == finch_type
        assert finch.ftype(np_type(1)) == finch_type
        assert finch.ftype(np.dtype(np_type)) == finch_type


def test_floor_divide_return_type_handles_all_integer_dtypes():
    dtypes = [
        finch.bool,
        finch.int8,
        finch.int16,
        finch.int32,
        finch.int64,
        finch.uint8,
        finch.uint16,
        finch.uint32,
        finch.uint64,
    ]

    for x1 in dtypes:
        for x2 in dtypes:
            assert isinstance(ffuncs.floordiv.ftype.return_type(x1, x2), FDType)


def test_same_ffunc():
    assert ffuncs.same(1, 1)
    assert not ffuncs.same(1, 2)
    assert ffuncs.same(float("nan"), float("nan"))
    assert ffuncs.same(np.float32(np.nan), np.float64(np.nan))
    assert not ffuncs.same(float("nan"), 1.0)
    assert ffuncs.same(None, None)
    assert not ffuncs.not_same(1, 1)
    assert ffuncs.not_same(1, 2)
    assert not ffuncs.not_same(float("nan"), float("nan"))
    assert not ffuncs.not_same(None, None)


def test_same_ffunc_dunder_overload():
    class LeftSame:
        def __same__(self, other):
            return np.False_

    class RightSame:
        def __rsame__(self, other):
            return np.True_

    class LeftDefers:
        def __same__(self, other):
            return NotImplemented

    assert ffuncs.same(LeftSame(), RightSame()) is np.False_
    assert ffuncs.same(LeftDefers(), RightSame()) is np.True_


def test_samehash():
    class SameHash:
        def __samehash__(self):
            return ("samehash", 1)

    assert ffuncs.samehash(1) == 1
    assert ffuncs.samehash(np.float64(np.nan)) == ("nan", finch.float64)
    assert ffuncs.samehash(np.float32(np.nan)) == ("nan", finch.float32)
    assert ffuncs.samehash(SameHash()) == ("samehash", 1)


@pytest.mark.parametrize("fill", [0, 5, finch.algebra.StaticFill(5)])
def test_init_write(fill):
    op = ffuncs.init_write(fill)
    z = fill.value if isinstance(fill, finch.algebra.StaticFill) else fill
    assert op(17, z) == 17
    assert op(17, z + 1) == z + 1


@pytest.mark.parametrize(
    "op",
    [
        value
        for value in vars(ffuncs).values()
        if isinstance(value, finch.algebra.FinchOperator)
    ],
)
def test_function_types(op):
    assert isinstance(op, finch.algebra.FTyped)
    op_type = finch.ftype(op)
    assert isinstance(op_type, finch.algebra.FinchOperatorFType)
    assert isinstance(finch.algebra.arity(op_type), (int, float))
    assert isinstance(is_associative(op_type), bool)
    assert isinstance(is_idempotent(op_type), bool)


def test_function_properties_dispatch_on_types():
    add_t, mul_t = finch.ftype(ffuncs.add), finch.ftype(ffuncs.mul)
    assert is_identity(add_t, 0)
    assert is_annihilator(mul_t, 0)
    assert is_distributive(mul_t, add_t)
    assert not is_distributive(add_t, mul_t)
    assert repeat_operator(add_t) is ffuncs.mul
    assert init_value(add_t, finch.int64) == 0
    assert cansplitpush(add_t)
    assert finch.algebra.return_type(add_t, finch.int64, finch.int64) == finch.int64
    static = finch.ftype(ffuncs.init_write(finch.algebra.StaticFill(0)))
    dynamic = finch.ftype(ffuncs.init_write(finch.algebra.DynamicFill(0)))
    assert is_identity(static, 0)
    assert not is_identity(dynamic, 0)


@pytest.mark.parametrize(
    "factory, first, second",
    [
        (ffuncs.choose, DynamicFill(np.int64(0)), DynamicFill(np.int64(3))),
        (ffuncs.scaled_power, 3.0, 4.0),
        (ffuncs.add_scaled_power, 3.0, 4.0),
        (ffuncs.scaled_negative_power, -3.0, -4.0),
        (ffuncs.add_scaled_negative_power, -3.0, -4.0),
        (ffuncs.root_scaled_power, 3.0, 4.0),
        (ffuncs.root_scaled_negative_power, -3.0, -4.0),
    ],
)
def test_function_ftypes_do_not_specialize_runtime_data(factory, first, second):
    left, right = factory(first), factory(second)
    assert left != right
    assert ftype(left) == ftype(right)
    assert hash(ftype(left)) == hash(ftype(right))
    assert type(ftype(left)) is not finch.algebra.SingletonOperatorFType
    assert not hasattr(left, "return_type")
    assert not hasattr(left, "is_associative")
    assert finch.algebra.is_associative(left.ftype) == finch.algebra.is_associative(
        right.ftype
    )
    field_values = [getattr(right, name) for name in ftype(right).struct_fieldnames]
    assert ftype(left).from_fields(*field_values) == right


def test_operator_types_own_properties():
    for value in vars(ffuncs).values():
        if isinstance(value, finch.algebra.FinchOperator):
            assert type(ftype(value)) is not finch.algebra.SingletonOperatorFType
            for name in (
                "arity",
                "return_type",
                "is_associative",
                "is_commutative",
                "is_identity",
            ):
                assert hasattr(ftype(value), name)
                assert not hasattr(value, name)
    assert ftype(ffuncs.add) != ftype(ffuncs.mul)
    assert finch.algebra.is_distributive(ftype(ffuncs.mul), ftype(ffuncs.add))
