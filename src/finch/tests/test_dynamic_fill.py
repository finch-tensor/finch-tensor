"""
Tests for Known/Dynamic fill values (`finch.algebra.fill`), ConstantScalar
specialization, and 0-d scalar operand lowering on compiled backends.
"""

import builtins

import pytest

import numpy as np

import finch
from finch import (
    ConstantScalar,
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    element,
)
from finch.algebra import (
    DynamicFill,
    DynamicFillError,
    apply_fill,
    ffuncs,
    ftype,
    is_annihilator,
    is_dynamic,
    is_identity,
)
from finch.autoschedule import (
    COMPILE_NUMBA,
    INTERPRET_ASSEMBLY,
    INTERPRET_NOTATION,
    DefaultLogicFormatter,
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)
from finch.finch_logic import Literal, LogicLoader, MapJoin, Query, Reorder
from finch.finch_notation.interpreter import NotationInterpreter
from finch.symbolic import UnvalidatedForm

from .conftest import finch_assert_allclose


def test_dynamic_fill_identity():
    d = DynamicFill(np.float64)
    assert ftype(d) == ftype(np.float64)
    assert d == DynamicFill(np.float64)
    assert hash(d) == hash(DynamicFill(np.float64))
    assert d != DynamicFill(np.int64)
    assert d != 0.0


def test_dynamic_fill_same():
    d = DynamicFill(np.float64)
    assert ffuncs.same(d, DynamicFill(np.float64))
    assert not ffuncs.same(d, DynamicFill(np.int64))
    assert not ffuncs.same(d, 3.0)
    assert not ffuncs.same(3.0, d)
    assert ffuncs.samehash(d) == ffuncs.samehash(DynamicFill(np.float64))


def test_apply_fill_known_folds():
    assert apply_fill(ffuncs.add, 1.0, 2.0) == 3.0
    assert np.isnan(apply_fill(ffuncs.add, float("nan"), 2.0))


def test_apply_fill_dynamic_propagates():
    d = DynamicFill(np.float64)
    assert apply_fill(ffuncs.add, 2.0, d) == DynamicFill(np.float64)
    assert apply_fill(ffuncs.add, d, d) == DynamicFill(np.float64)
    r = apply_fill(ffuncs.add, np.int64(1), d)
    assert isinstance(r, DynamicFill)
    assert ftype(r) == ftype(np.float64)


def test_apply_fill_annihilator_refinement():
    # A Known annihilator determines the fill regardless of the Dynamic args.
    assert apply_fill(ffuncs.mul, 0.0, DynamicFill(np.float64)) == 0.0
    assert apply_fill(ffuncs.add, float("inf"), DynamicFill(np.float64)) == float("inf")
    assert apply_fill(ffuncs.logical_and, False, DynamicFill(np.bool_)) is False


def test_predicates_conservative_on_dynamic():
    d = DynamicFill(np.float64)
    assert not is_annihilator(ffuncs.mul, d)
    assert not is_identity(ffuncs.mul, d)
    assert is_annihilator(ffuncs.mul, 0.0)
    assert is_identity(ffuncs.mul, 1.0)


@pytest.mark.parametrize(
    "ctx",
    [INTERPRET_ASSEMBLY, COMPILE_NUMBA],
    ids=["interpret_assembly", "compile_numba"],
)
def test_compiled_scalar_operand(ctx):
    arr = np.array([1.0, 2.0, 3.0])
    x = finch.asarray(arr)
    out = finch.compute(finch.lazy(x) * 2.0, ctx=ctx)
    finch_assert_allclose(out, arr * 2.0)


def _cached_scheduler():
    executor = LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        ),
        cache=True,
    )
    return executor, LogicNormalizer(executor)


def test_constant_scalar_compiles_per_value():
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        out = finch.compute(finch.lazy(x) + ConstantScalar(v), ctx=ctx)
        finch_assert_allclose(out, arr + v)
    assert len(executor.cached_kernels) == 5


def test_constant_scalar_caches_same_value():
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for _ in range(5):
        out = finch.compute(finch.lazy(x) + ConstantScalar(2.0), ctx=ctx)
        finch_assert_allclose(out, arr + 2.0)
    assert len(executor.cached_kernels) == 1


def test_plain_scalar_single_kernel():
    # The design-goal regression: a loop over distinct plain scalar values
    # compiles exactly one kernel.
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        out = finch.compute(finch.lazy(x) + v, ctx=ctx)
        finch_assert_allclose(out, arr + v)
    assert len(executor.cached_kernels) == 1


def test_constant_scalar_inlines_to_literal():
    x = finch.lazy(finch.asarray(np.arange(3.0)))
    y = x + ConstantScalar(2.0)
    queries = [s for s in y.ctx.trace() if isinstance(s, Query)]
    # One query binds the input table, one the mapjoin; no scalar binding.
    assert len(queries) == 2
    (mapjoin_q,) = [q for q in queries if q.lhs == y.data]
    match mapjoin_q.rhs:
        case Reorder(MapJoin(Literal(_), args), _):
            assert Literal(2.0) in args
        case _:
            raise AssertionError(f"unexpected rhs: {mapjoin_q.rhs}")


def test_plain_scalar_becomes_binding():
    x = finch.lazy(finch.asarray(np.arange(3.0)))
    y = x + 2.0
    queries = [s for s in y.ctx.trace() if isinstance(s, Query)]
    # The plain scalar is a real 0-d tensor binding: input table, scalar
    # table, and the mapjoin.
    assert len(queries) == 3


def test_all_constant_scalars_fold():
    y = finch.add(ConstantScalar(3), ConstantScalar(4.5))
    assert float(finch.compute(y)) == 7.5


def test_scalar_annihilator_keeps_known_fill():
    # x * v: the tensor's Known 0.0 fill annihilates mul, so the output fill
    # stays Known while the kernel is still shared across values.
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for v in [1.0, 2.0, 3.0]:
        out = finch.compute(finch.lazy(x) * v, ctx=ctx)
        finch_assert_allclose(out, arr * v)
        assert out.fill_value == 0.0
    assert len(executor.cached_kernels) == 1


@pytest.mark.parametrize(
    "ctx",
    [INTERPRET_NOTATION, COMPILE_NUMBA],
    ids=["interpret_notation", "compile_numba"],
)
@pytest.mark.parametrize(
    "op,np_op,values",
    [
        (finch.add, np.add, [0.0, 1.0, -3.0, float("nan")]),
        (finch.multiply, np.multiply, [0.0, 1.0, -3.0, float("nan")]),
        (finch.subtract, np.subtract, [0.0, 1.0, -3.0, float("nan")]),
        # nan omitted for maximum: numba's max() drops nans where np.maximum
        # propagates them (pre-existing backend divergence).
        (finch.maximum, np.maximum, [0.0, 1.0, -3.0]),
    ],
    ids=["add", "mul", "sub", "maximum"],
)
def test_scalar_correctness_matrix(ctx, op, np_op, values):
    arr = np.array([[1.0, 0.0], [-2.0, 4.0]])
    x = finch.asarray(arr)
    for v in values:
        out = finch.compute(op(finch.lazy(x), v), ctx=ctx)
        result = out.to_numpy()
        expected = np_op(arr, v)
        np.testing.assert_array_equal(result, expected)
        assert not is_dynamic(out.fill_value)
        # The tensor's Known 0.0 fill annihilates mul (finch's sparse algebra
        # deliberately treats 0 * anything as 0).
        expected_fill = 0.0 if op is finch.multiply else np_op(0.0, v)
        assert ffuncs.same(out.fill_value, expected_fill)


def test_stale_fill_regression():
    # The second call's result must reflect the second value, in both the
    # stored values and the fill metadata.
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    out_a = finch.compute(finch.lazy(x) + 2.0, ctx=ctx)
    out_b = finch.compute(finch.lazy(x) + 7.0, ctx=ctx)
    finch_assert_allclose(out_b, arr + 7.0)
    assert out_b.fill_value == 7.0
    finch_assert_allclose(out_a, arr + 2.0)
    assert out_a.fill_value == 2.0
    assert len(executor.cached_kernels) == 1


def test_order_independence():
    # A leading annihilator-valued call must not specialize the shared kernel.
    def run(values):
        executor, ctx = _cached_scheduler()
        arr = np.arange(3.0)
        x = finch.asarray(arr)
        outs = {v: finch.compute(finch.lazy(x) * v, ctx=ctx).to_numpy() for v in values}
        return outs, len(executor.cached_kernels)

    outs_a, kernels_a = run([0.0, 2.0])
    outs_b, kernels_b = run([2.0, 0.0])
    assert kernels_a == kernels_b == 1
    for v in [0.0, 2.0]:
        np.testing.assert_array_equal(outs_a[v], outs_b[v])


def _cached_galley_scheduler():
    from finch.autoschedule.galley_optimize import GalleyLogicalOptimizer

    executor = LogicExecutor(
        GalleyLogicalOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        ),
        cache=True,
    )
    return executor, LogicNormalizer(executor)


def test_galley_scalar_cache_counts():
    executor, ctx = _cached_galley_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for v in [1.0, 2.0, 3.0]:
        out = finch.compute(finch.lazy(x) + v, ctx=ctx)
        finch_assert_allclose(out, arr + v)
        assert out.fill_value == v
    for v in [0.0, 2.0, 5.0]:
        out = finch.compute(finch.lazy(x) * v, ctx=ctx)
        finch_assert_allclose(out, arr * v)
        assert out.fill_value == 0.0
    # one kernel for the adds, one for the muls
    assert len(executor.cached_kernels) == 2


def test_galley_order_independence():
    def run(values):
        executor, ctx = _cached_galley_scheduler()
        arr = np.arange(3.0)
        x = finch.asarray(arr)
        outs = {v: finch.compute(finch.lazy(x) * v, ctx=ctx).to_numpy() for v in values}
        return outs, len(executor.cached_kernels)

    outs_a, kernels_a = run([0.0, 2.0])
    outs_b, kernels_b = run([2.0, 0.0])
    assert kernels_a == kernels_b == 1
    for v in [0.0, 2.0]:
        np.testing.assert_array_equal(outs_a[v], outs_b[v])


class _DynamicFillFiber(FiberTensor):
    """A fiber tensor that binds with a dynamic fill, so one kernel serves
    instances differing only in fill value."""

    @property
    def argument_ftype(self):
        return self.ftype.with_fill(DynamicFill(self.ftype.element_type))


def _sparse_fiber(fill, cls=FiberTensor):
    # 3x3 dense-over-sparse-list: [[1, 0, 0], [1, 2, 0], [0, 0, 1]] with the
    # given background at the unstored positions.
    dtype = np.float64
    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))
    lvl = DenseLevel(
        SparseListLevel(
            ElementLevel(
                element(dtype(fill), ftype(dtype), ftype(np.intp), NumpyBufferFType),
                data,
            ),
            np.intp(3),
            ptr,
            idx,
        ),
        np.intp(3),
    )
    stored = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], dtype=dtype)
    dense_equiv = np.where(stored != 0, stored, fill)
    # the stored 0 positions really hold the fill in this pattern
    return cls(lvl), dense_equiv


def _numba_backend():
    from finch.codegen import NumbaCompiler
    from finch.compile import NotationCompiler
    from finch.finch_assembly import AssemblySimplify, LowerPackedStructSlots

    return NotationCompiler(
        NumbaCompiler(),
        ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify()),
    )


# Restricted to numba: sparse looplets fail on the assembly interpreter with
# Known fills too (pre-existing 'i_stop' scoping issue).
@pytest.mark.parametrize(
    "backend",
    [_numba_backend],
    ids=["compile_numba"],
)
def test_sparse_dynamic_fill_channel(backend):
    # A kernel compiled once against a dynamic-fill sparse format reads the
    # fill from the struct channel: gap positions reflect each instance's
    # actual fill value.
    executor = LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(DefaultLogicFormatter(LogicCompiler(backend())))
        ),
        cache=True,
    )
    sched = LogicNormalizer(executor)
    b_np = np.full((3, 3), 10.0)
    b = finch.asarray(b_np)
    for fill in [3.0, 7.0]:
        a, a_np = _sparse_fiber(fill, cls=_DynamicFillFiber)
        out = finch.compute(finch.lazy(a) + finch.lazy(b), ctx=sched)
        np.testing.assert_array_equal(out.to_numpy(), a_np + b_np)
    assert len(executor.cached_kernels) == 1


class _DynamicRejectingLoader(UnvalidatedForm, LogicLoader):
    """Wraps a loader, refusing any binding with a dynamic fill."""

    def __init__(self, ctx):
        self.ctx = ctx

    def lower(self, prgm, bindings, stats, stats_factory):
        if builtins.any(is_dynamic(t.fill_value) for t in bindings.values()):
            raise DynamicFillError("dynamic fills unsupported here")
        return self.ctx(prgm, bindings, stats, stats_factory)


def test_backend_refusal_propagates():
    # The executor does not recover from a backend that cannot express a
    # runtime fill; the error reaches the caller so the backend (or the user)
    # can decide what to do, rather than silently recompiling per value.
    loader = _DynamicRejectingLoader(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )
    )
    ctx = LogicNormalizer(LogicExecutor(loader, cache=True))
    x = finch.asarray(np.arange(3.0))
    with pytest.raises(DynamicFillError):
        finch.compute(finch.lazy(x) + 2.0, ctx=ctx)
