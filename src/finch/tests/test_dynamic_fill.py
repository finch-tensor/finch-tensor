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
    DynamicFillFType,
    StaticFill,
    StaticFillFType,
    apply_fill,
    as_fill,
    ffuncs,
    ftype,
    is_annihilator,
    is_dynamic,
    is_identity,
)
from finch.autoschedule import (
    COMPILE_NUMBA,
    INTERPRET_ASSEMBLY,
    INTERPRET_LOGIC,
    INTERPRET_NOTATION,
    CompilerFormLowerer,
    DefaultLogicFactorizer,
    DefaultLogicFormatter,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    NotationGenerator,
)
from finch.autoschedule.normalize import inline_constant_scalars
from finch.finch_logic import (
    Aggregate,
    Field,
    FillTable,
    HardAlias,
    Literal,
    LogicLoader,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from finch.finch_notation.interpreter import NotationInterpreter
from finch.symbolic import UnvalidatedForm

from .conftest import finch_assert_allclose


def test_dynamic_fill_identity():
    d = DynamicFill(np.float64(0.0))
    # A fill's ftype is the fill of its format, which records only its type.
    assert ftype(d) == DynamicFillFType(np.float64)
    assert d.element_type == ftype(np.float64)
    assert d == DynamicFill(np.float64(0.0))
    assert hash(d) == hash(DynamicFill(np.float64(0.0)))
    assert d != DynamicFill(np.int64(0))
    assert d != 0.0


def test_dynamic_fill_keeps_its_value_but_not_its_identity():
    """A Dynamic fill knows its value; kernels just may not specialize on it.
    Equality is by dtype so one kernel serves every fill of that dtype."""
    a = DynamicFill(np.float64(3.0))
    b = DynamicFill(np.float64(5.0))
    assert a.value == 3.0
    assert b.value == 5.0
    assert a == b and hash(a) == hash(b)


def test_static_fill_does_not_equal_its_raw_value():
    assert StaticFill(0) == StaticFill(0)
    assert StaticFill(0) != 0
    assert StaticFill(0).value == 0
    assert as_fill(0) == StaticFill(0)
    assert as_fill(StaticFill(0)) == StaticFill(0)


def test_dynamic_fill_same():
    d = DynamicFill(np.float64(0.0))
    assert ffuncs.same(d, DynamicFill(np.float64(0.0)))
    assert not ffuncs.same(d, DynamicFill(np.int64(0)))
    assert not ffuncs.same(d, 3.0)
    assert not ffuncs.same(3.0, d)
    assert hash(ffuncs.canonical(d)) == hash(
        ffuncs.canonical(DynamicFill(np.float64(0.0)))
    )


def test_apply_fill_known_folds():
    # A static result stays static only while the algebra can act on its value.
    one, two, zero = StaticFill(1.0), StaticFill(2.0), StaticFill(0.0)
    assert apply_fill(ffuncs.add, one, zero) == one
    assert apply_fill(ffuncs.mul, two, zero) == zero
    # The value still folds when it cannot be compiled against; it is only the
    # marking that goes, so a chain of such folds shares one kernel.
    folded = apply_fill(ffuncs.add, one, two)
    assert folded.value == 3.0
    assert is_dynamic(folded)
    assert np.isnan(apply_fill(ffuncs.add, StaticFill(float("nan")), two).value)


def test_fill_ftypes():
    # A static fill's value is part of its format, but only the element type
    # of a dynamic fill is.
    assert StaticFill(np.float64(1.0)).ftype == StaticFillFType(np.float64(1.0))
    assert StaticFillFType(np.nan) == StaticFillFType(np.nan)
    assert StaticFillFType(np.float64(1.0)) != StaticFillFType(np.float64(2.0))
    assert DynamicFillFType(np.float64) == DynamicFill(np.float64(3.0)).ftype
    assert DynamicFillFType(np.float64) != DynamicFillFType(np.int64)
    # Calling a fill format makes one of its fills.
    fill = DynamicFillFType(np.float64)(7)
    assert isinstance(fill, DynamicFill)
    assert fill.value == 7.0
    assert fill.element_type == ftype(np.float64)
    assert StaticFillFType(np.float64(1.0))(1) == StaticFill(np.float64(1.0))
    with pytest.raises(ValueError):
        StaticFillFType(np.float64(1.0))(2)


def test_apply_fill_dynamic_propagates():
    d = DynamicFill(np.float64(4.0))
    assert apply_fill(ffuncs.add, StaticFill(2.0), d) == DynamicFill(np.float64(0.0))
    assert apply_fill(ffuncs.add, d, d) == DynamicFill(np.float64(0.0))
    r = apply_fill(ffuncs.add, StaticFill(np.int64(1)), d)
    assert isinstance(r, DynamicFill)
    assert r.element_type == ftype(np.float64)
    # The value still folds; it is simply not something to compile against.
    assert r.value == 5.0


def test_apply_fill_annihilator_refinement():
    # A Static annihilator determines the fill regardless of the Dynamic args,
    # so the result is Static again.
    two = DynamicFill(np.float64(2.0))
    # The result takes the element type of the operation.
    assert apply_fill(ffuncs.mul, StaticFill(0.0), two) == StaticFill(np.float64(0))
    inf = StaticFill(float("inf"))
    assert apply_fill(ffuncs.add, inf, two) == StaticFill(np.float64(np.inf))
    assert apply_fill(
        ffuncs.logical_and, StaticFill(False), DynamicFill(np.bool_(True))
    ) == StaticFill(np.False_)


def test_predicates_conservative_on_dynamic():
    d = DynamicFill(np.float64(0.0))
    assert not is_annihilator(ffuncs.mul.ftype, d)
    assert not is_identity(ffuncs.mul.ftype, d)
    assert is_annihilator(ffuncs.mul.ftype, 0.0)
    assert is_identity(ffuncs.mul.ftype, 1.0)


@pytest.mark.parametrize(
    "ctx",
    [INTERPRET_ASSEMBLY, COMPILE_NUMBA],
    ids=["interpret_assembly", "compile_numba"],
)
def test_compiled_scalar_operand(ctx):
    arr = np.array([1.0, 2.0, 3.0])
    x = finch.asarray(arr)
    out = finch.compute(finch.defer(x) * 2.0, ctx=ctx)
    finch_assert_allclose(out, arr * 2.0)


def _cached_scheduler():
    executor = LogicExecutor(
        DefaultLogicFactorizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    CompilerFormLowerer(LogicCompiler(NotationInterpreter()))
                )
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
        out = finch.compute(finch.defer(x) + ConstantScalar(v), ctx=ctx)
        finch_assert_allclose(out, arr + v)
    assert len(executor.cached_kernels) == 5


def test_constant_scalar_caches_same_value():
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for _ in range(5):
        out = finch.compute(finch.defer(x) + ConstantScalar(2.0), ctx=ctx)
        finch_assert_allclose(out, arr + 2.0)
    assert len(executor.cached_kernels) == 1


def test_plain_scalar_shares_one_kernel():
    executor, ctx = _cached_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        out = finch.compute(finch.defer(x) + v, ctx=ctx)
        finch_assert_allclose(out, arr + v)
    assert len(executor.cached_kernels) == 2
    for v in [float(i) for i in range(6, 16)]:
        out = finch.compute(finch.defer(x) + v, ctx=ctx)
        finch_assert_allclose(out, arr + v)
    assert len(executor.cached_kernels) == 2


def test_an_accumulating_fill_does_not_compile_per_iteration():
    """Chaining a specializable constant must not cost a kernel per step.

    Combining static fills computes an arbitrary value, so `x + 1` repeated
    walks the output fill through 1, 2, 3, ... A static fill is compared by
    value, so leaving those static keyed a kernel per iteration -- the
    O(#iterations) compilation this whole scheme exists to avoid. `apply_fill`
    keeps a fill static only while its value is one the algebra can act on.
    """
    arr = np.arange(3.0)
    counts = []
    for trips in (1, 2, 4, 8, 16):
        executor, ctx = _cached_scheduler()
        acc = finch.asarray(arr)
        for _ in range(trips):
            acc = finch.compute(finch.defer(acc) + 1, ctx=ctx)
        finch_assert_allclose(acc, arr + trips)
        counts.append(len(executor.cached_kernels))
    # Flat once the loop body has been seen, not linear in the trip count.
    assert counts[-1] == counts[-2] == counts[2], counts


def test_constant_scalar_inlines_to_literal():
    """The trace binds every operand as a table; the normalizer inlines the
    constant and drops the binding it came from."""
    x = finch.defer(finch.asarray(np.arange(3.0)))
    y = x + ConstantScalar(2.0)

    trace = Plan(tuple(s for s in y.ctx.trace() if isinstance(s, Query)))
    assert len(trace.bodies) == 3

    inlined = inline_constant_scalars(trace)
    assert isinstance(inlined, Plan)
    assert len(inlined.bodies) == 2
    assert (
        len(
            [
                q
                for q in inlined.bodies
                if not isinstance(q, Query)
                or isinstance(q.rhs, Table)
                and (
                    not isinstance(q.rhs.tns, Literal)
                    or isinstance(q.rhs.tns.val, ConstantScalar)
                )
            ]
        )
        == 0
    )
    assert all(isinstance(q, Query) for q in inlined.bodies)
    (mapjoin_q,) = [q for q in inlined.bodies if q.lhs.tns == y.data]  # ty: ignore[unresolved-attribute]
    assert isinstance(mapjoin_q, Query)
    match mapjoin_q.rhs:
        case Reorder(MapJoin(Literal(_), args), _):
            assert Reorder(Literal(2.0), ()) in args
        case _:
            raise AssertionError(f"unexpected rhs: {mapjoin_q.rhs}")


def test_a_produced_constant_keeps_its_binding():
    """
    We shouldn't create `Query(Table(a, ()), Literal(v))` via inlining of constants.
    """
    out = finch.compute(finch.defer(ConstantScalar(1)))
    assert float(np.asarray(out)) == float(np.asarray(ConstantScalar(1)))


def test_plain_scalar_becomes_binding():
    x = finch.defer(finch.asarray(np.arange(3.0)))
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
        out = finch.compute(finch.defer(x) * v, ctx=ctx)
        finch_assert_allclose(out, arr * v)
        assert out.fill_value == 0.0
    # 1.0 is mul's identity, so it inlines and simplifies to its own kernel;
    # 2.0 and 3.0 share the other.
    assert len(executor.cached_kernels) == 2


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
        out = finch.compute(op(finch.defer(x), v), ctx=ctx)
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
    out_a = finch.compute(finch.defer(x) + 2.0, ctx=ctx)
    out_b = finch.compute(finch.defer(x) + 7.0, ctx=ctx)
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
        outs = {
            v: finch.compute(finch.defer(x) * v, ctx=ctx).to_numpy() for v in values
        }
        return outs, len(executor.cached_kernels)

    outs_a, kernels_a = run([0.0, 2.0])
    outs_b, kernels_b = run([2.0, 0.0])
    assert kernels_a == kernels_b == 2
    for v in [0.0, 2.0]:
        np.testing.assert_array_equal(outs_a[v], outs_b[v])


def _cached_galley_scheduler():
    from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
        GalleyLogicFactorizer,
    )

    executor = LogicExecutor(
        GalleyLogicFactorizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    CompilerFormLowerer(LogicCompiler(NotationInterpreter()))
                )
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
        out = finch.compute(finch.defer(x) + v, ctx=ctx)
        finch_assert_allclose(out, arr + v)
        assert out.fill_value == v
    for v in [0.0, 2.0, 5.0]:
        out = finch.compute(finch.defer(x) * v, ctx=ctx)
        finch_assert_allclose(out, arr * v)
        assert out.fill_value == 0.0
    assert len(executor.cached_kernels) == 4


def test_galley_order_independence():
    def run(values):
        executor, ctx = _cached_galley_scheduler()
        arr = np.arange(3.0)
        x = finch.asarray(arr)
        outs = {
            v: finch.compute(finch.defer(x) * v, ctx=ctx).to_numpy() for v in values
        }
        return outs, len(executor.cached_kernels)

    outs_a, kernels_a = run([0.0, 2.0])
    outs_b, kernels_b = run([2.0, 0.0])
    assert kernels_a == kernels_b == 2
    for v in [0.0, 2.0]:
        np.testing.assert_array_equal(outs_a[v], outs_b[v])


class _DynamicFillFiber(FiberTensor):
    """A fiber tensor that binds with a dynamic fill, so one kernel serves
    instances differing only in fill value."""

    @property
    def ftype(self):
        base = super().ftype
        return base.with_fill(base.fill_type.as_dynamic())


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
        DefaultLogicFactorizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler(backend())))
            )
        ),
        cache=True,
    )
    sched = LogicNormalizer(executor)
    b_np = np.full((3, 3), 10.0)
    b = finch.asarray(b_np)
    for fill in [3.0, 7.0]:
        a, a_np = _sparse_fiber(fill, cls=_DynamicFillFiber)
        out = finch.compute(finch.defer(a) + finch.defer(b), ctx=sched)
        np.testing.assert_array_equal(out.to_numpy(), a_np + b_np)
    assert len(executor.cached_kernels) == 1


class _DynamicRejectingLoader(UnvalidatedForm, LogicLoader):
    """Wraps a loader, refusing any binding with a dynamic fill."""

    def __init__(self, ctx):
        self.ctx = ctx

    def lower(self, prgm, bindings, stats, stats_factory):
        if builtins.any(is_dynamic(t.fill_type) for t in bindings.values()):
            raise DynamicFillError("dynamic fills unsupported here")
        return self.ctx(prgm, bindings, stats, stats_factory)


@pytest.mark.parametrize(
    "loader",
    [_numba_backend, NotationInterpreter],
    ids=["compile_numba", "interpret_notation"],
)
def test_declare_sets_a_dynamic_fill(loader):
    # A kernel computes the dynamic fill of its output from the fills of its
    # inputs, rather than keeping the fill the output was constructed with.
    i = Field("i")
    A, C = HardAlias("A"), HardAlias("C")
    one = Literal(np.float64(1.0))
    arg = MapJoin(Literal(ffuncs.add), (Table(A, (i,)), one))
    init = MapJoin(Literal(ffuncs.add), (FillTable(A), one))
    plan = Plan(
        (
            Query(
                Table(C, (i,)),
                Aggregate(Literal(ffuncs.overwrite), init, Reorder(arg, (i,)), ()),
            ),
            Produces((C,)),
        )
    )
    a = finch.BufferizedNDArray.from_numpy(
        np.array([1.0, 2.0]), fill_value=DynamicFill(np.float64(2.0))
    )
    c = finch.BufferizedNDArray.from_numpy(
        np.zeros(2), fill_value=DynamicFill(np.float64(0.0))
    )
    program = NotationGenerator()(plan, {A: a.ftype, C: c.ftype}, {}, None)
    loader()(program).main(a, c)
    assert c.fill_value == 3.0
    np.testing.assert_array_equal(c.to_numpy(), [2.0, 3.0])


# The logic interpreter computes the fills of its results from values, so they
# are static.
@pytest.mark.parametrize(
    "ctx, dynamic",
    [(None, True), (INTERPRET_LOGIC, False)],
    ids=["default", "interpret_logic"],
)
@pytest.mark.parametrize(
    "op, expected",
    [
        (lambda x: finch.reshape(x, (4,)), [9.0, 1.0, 2.0, 9.0]),
        (lambda x: finch.sort(finch.reshape(x, (4,))), [1.0, 2.0, 9.0, 9.0]),
        (finch.empty_like, [[9.0, 9.0], [9.0, 9.0]]),
    ],
    ids=["reshape", "sort", "empty_like"],
)
def test_lazy_functions_keep_a_dynamic_fill(ctx, dynamic, op, expected):
    # Functions which use the fill of a tensor as data read it at runtime, so a
    # dynamic fill stays dynamic rather than becoming a literal.
    x = finch.BufferizedNDArray.from_numpy(
        np.array([[9.0, 1.0], [2.0, 9.0]]), fill_value=DynamicFill(np.float64(9.0))
    )
    lazy_result = op(finch.defer(x))
    assert is_dynamic(lazy_result.fill)
    result = finch.compute(lazy_result, **({} if ctx is None else {"ctx": ctx}))
    assert bool(is_dynamic(result.fill)) == dynamic
    assert result.fill_value == 9.0
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_backend_refusal_propagates():
    # The executor does not recover from a backend that cannot express a
    # runtime fill; the error reaches the caller so the backend (or the user)
    # can decide what to do, rather than silently recompiling per value.
    loader = _DynamicRejectingLoader(
        DefaultLogicFactorizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    CompilerFormLowerer(LogicCompiler(NotationInterpreter()))
                )
            )
        )
    )
    ctx = LogicNormalizer(LogicExecutor(loader, cache=True))
    x = finch.asarray(np.arange(3.0))
    with pytest.raises(DynamicFillError):
        finch.compute(finch.defer(x) + 2.0, ctx=ctx)


def test_ftype_reports_a_fill_and_tensor_reports_a_value():
    """The two layers of the contract: an ftype says *whether* a kernel may
    specialize on the fill, a tensor just gives the value."""
    from finch.algebra import AbstractFill, AbstractFillFType

    arr = finch.asarray(np.arange(3.0))
    assert isinstance(arr.ftype.fill_type, AbstractFillFType)
    assert not isinstance(arr.fill_value, AbstractFill)
    # Only a static fill's value is part of the format.
    assert arr.fill_value == arr.ftype.fill_type.value

    s = finch.Scalar(np.float64(2.0))
    assert isinstance(s.ftype.fill_type, AbstractFillFType)
    assert s.fill_value == 2.0
    # A Scalar declares its fill dynamic, which keeps the value out of the
    # kernel cache key. The scalar holds the value itself.
    assert is_dynamic(s.ftype.fill_type)


def test_executor_output_keeps_a_dynamic_fill():
    """An output fill that could not be determined from the static inputs must
    stay dynamic, so the next computation cannot specialize on it."""
    executor, ctx = _cached_galley_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)

    out = finch.compute(finch.defer(x) + 5.0, ctx=ctx)
    finch_assert_allclose(out, arr + 5.0)

    # The value resolved, and it is the right one ...
    assert out.fill_value == 5.0
    # ... but nothing downstream may compile against it.
    assert is_dynamic(out.ftype.fill_type)


def test_dynamic_output_feeds_a_reusable_kernel():
    """Chaining off a dynamically filled output reuses one kernel across fills,
    which is the whole point of not collapsing the output to a static fill."""
    executor, ctx = _cached_galley_scheduler()
    arr = np.arange(3.0)
    x = finch.asarray(arr)

    for v in [2.0, 3.0, 4.0]:
        step1 = finch.compute(finch.defer(x) + v, ctx=ctx)
        assert is_dynamic(step1.ftype.fill_type)
        step2 = finch.compute(finch.defer(step1) * 2.0, ctx=ctx)
        finch_assert_allclose(step2, (arr + v) * 2.0)

    # One kernel for the add, one for the multiply -- not one per fill value.
    assert len(executor.cached_kernels) == 2
