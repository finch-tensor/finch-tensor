import ast
import importlib
import operator
import textwrap

import pytest

import numpy as np

import finch
from finch.autoschedule import (
    DefaultLogicFactorizer,
    DefaultLogicFormatter,
    DefaultLoopOrderer,
    LogicCapture,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)
from finch.finch_fused import jit
from finch.finch_fused import nodes as fzd
from finch.finch_fused.calls import _transparent_cache, wrap_calls
from finch.finch_fused.cfg_builder import (
    fused_build_cfg,
    fused_desugar,
    number_statements,
)
from finch.finch_fused.dataflow import (
    LivenessAnalysis,
    insert_lazy_and_compute,
    maybedefer,
)
from finch.finch_fused.parser import (
    fused_function_to_python_ast,
    parse_fused_function,
)
from finch.finch_notation.interpreter import NotationInterpreter
from finch.interface import add, asarray, matmul, sum
from finch.interface.lazy import LazyTensor
from finch.tensor.scalar import ConstantScalar, ScalarFType

from .conftest import finch_assert_allclose


def test_parse_simple_function_with_control_flow_and_calls():
    def simple_fn(fn, n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = fn(total, i, scale=2)
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    result = parse_fused_function(simple_fn)

    expected = fzd.Function(
        fzd.Literal("simple_fn"),
        (fzd.Variable("fn"), fzd.Variable("n")),
        fzd.Block(
            (
                fzd.Assign(fzd.Variable("total"), fzd.Literal(ConstantScalar(0))),
                fzd.For(
                    fzd.Variable("i"),
                    fzd.Call(fzd.Literal(range), (fzd.Variable("n"),)),
                    fzd.Block(
                        (
                            fzd.If(
                                fzd.Compare(
                                    fzd.Variable("i"),
                                    fzd.Literal(operator.lt),
                                    fzd.Variable("n"),
                                ),
                                fzd.Block(
                                    (
                                        fzd.Assign(
                                            fzd.Variable("total"),
                                            fzd.Call(
                                                fzd.Variable("fn"),
                                                (
                                                    fzd.Variable("total"),
                                                    fzd.Variable("i"),
                                                ),
                                                (
                                                    fzd.Keyword(
                                                        "scale",
                                                        fzd.Literal(2),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                                fzd.Block(
                                    (
                                        fzd.Assign(
                                            fzd.Variable("total"),
                                            fzd.BinaryOp(
                                                fzd.Variable("total"),
                                                fzd.Literal(operator.sub),
                                                fzd.Literal(ConstantScalar(1)),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                fzd.While(
                    fzd.Compare(
                        fzd.Variable("total"),
                        fzd.Literal(operator.lt),
                        fzd.Variable("n"),
                    ),
                    fzd.Block(
                        (
                            fzd.Assign(
                                fzd.Variable("total"),
                                fzd.BinaryOp(
                                    fzd.Variable("total"),
                                    fzd.Literal(operator.add),
                                    fzd.Literal(ConstantScalar(1)),
                                ),
                            ),
                        )
                    ),
                ),
                fzd.Return((fzd.Variable("total"),)),
            )
        ),
    )

    assert result == expected


def test_parse_rejects_local_function_definitions():
    def with_local_fn(x):
        def inner(y):
            return y + 1

        return inner(x)

    with pytest.raises(ValueError, match="Local functions are not supported"):
        parse_fused_function(with_local_fn)


def test_parse_rejects_for_else_blocks():
    def with_for_else(n):
        for i in range(n):
            n = n + i
        else:
            n = n + 1
        return n

    with pytest.raises(ValueError, match="For-else blocks are not supported"):
        parse_fused_function(with_for_else)


def test_parse_rejects_while_else_blocks():
    def with_while_else(n):
        while n < 3:
            n = n + 1
        else:
            n = n + 2
        return n

    with pytest.raises(ValueError, match="While-else blocks are not supported"):
        parse_fused_function(with_while_else)


def test_parse_reverse_parse_is_lossless_on_supported_subset():
    """
    Round-tripping recovers the source. Numeric literals survive as literals:
    the parser leaves them alone, and a constant only becomes a ConstantScalar
    later, when `defer` hands it to the lazy layer.
    """

    def roundtrip_fn(n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = total + i
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    expected_source = textwrap.dedent("""\
        def roundtrip_fn(n):
            total = 0
            for i in range(n):
                if i < n:
                    total = total + i
                else:
                    total = total - 1
            while total < n:
                total = total + 1
            return total
        """)
    expected_fn = ast.parse(expected_source).body[0]

    fused_fn = parse_fused_function(roundtrip_fn)
    roundtrip_fn_ast = fused_function_to_python_ast(fused_fn)

    assert ast.dump(expected_fn, include_attributes=False) == ast.dump(
        roundtrip_fn_ast,
        include_attributes=False,
    )


def test_cfg_builder():
    def simple_fn(fn, n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = fn(total, i)
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    fused_fn = parse_fused_function(simple_fn)
    numbered_fn, _ = number_statements(fused_fn)
    desugared_fn = fused_desugar(numbered_fn)
    cfg = fused_build_cfg(desugared_fn)

    # We won't assert on the exact structure of the CFG here, but we can at least
    # check that it has the expected number of blocks. The exact number of blocks
    # may depend on how the CFG builder handles certain constructs, so this is a
    # somewhat loose check.
    assert (
        len(cfg.blocks) >= 5
    )  # Entry block, for loop block, if block, while block, return block


def _build_liveness(fn):
    """Helper: parse, number, desugar, build CFG, run liveness."""
    fused_fn = parse_fused_function(fn)
    numbered_fn, _ = number_statements(fused_fn)
    desugared_fn = fused_desugar(numbered_fn)
    cfg = fused_build_cfg(desugared_fn)
    liveness = LivenessAnalysis(cfg)
    liveness.analyze()
    return liveness, cfg


def _all_live_names(liveness, cfg):
    """Union of all live variable names across all blocks (in and out)."""
    names = set()
    for block in cfg.blocks.values():
        names |= {v.name for v in liveness.output_states[block.id]}
        names |= {v.name for v in liveness.input_states[block.id]}
    return names


def _transformed_jit_source(fn):
    transformed_fn = insert_lazy_and_compute(wrap_calls(parse_fused_function(fn)))
    assert isinstance(transformed_fn, fzd.Function)
    return ast.unparse(fused_function_to_python_ast(transformed_fn)) + "\n"


def test_liveness_straight_line():
    """Parameters must appear live at the function entry block."""

    def fn(a, b):
        c = add(a, b)
        return c  # noqa: RET504

    liveness, cfg = _build_liveness(fn)

    # After desugaring, the function body is a single block.
    # live-IN (output_states) of that block = {a, b}, since both are used.
    # c is defined and consumed in the same block so it never crosses a block
    # boundary and does not appear in any block-boundary state.
    all_live_in = set()
    for block in cfg.blocks.values():
        all_live_in |= {v.name for v in liveness.output_states[block.id]}

    assert "a" in all_live_in
    assert "b" in all_live_in
    assert "c" not in all_live_in


def test_liveness_dead_variable():
    """A variable assigned but never used afterwards must not be live after."""

    def fn(a, b):
        unused = add(a, b)  # noqa: F841
        c = matmul(a, b)
        return c  # noqa: RET504

    liveness, cfg = _build_liveness(fn)

    exit_block = list(cfg.blocks.values())[-1]
    live_at_exit = {v.name for v in liveness.input_states[exit_block.id]}
    assert "unused" not in live_at_exit


def test_liveness_loop_carried():
    """Loop-carried variables must be live at the top of the loop body."""

    def fn(n):
        total = 0
        for _i in range(n):
            total = total + 1
        return total

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "total" in names
    assert "n" in names


def test_liveness_multi_loop_carried():
    """Multiple loop-carried variables must all be live inside the loop."""

    def fn(A, B, C, n):
        D = matmul(A, B)
        E = add(A, C)
        for _i in range(n):
            D = add(D, E)
        return D

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "D" in names
    assert "E" in names
    assert "n" in names


def test_liveness_if_branch_merges():
    """Variables used in either branch must be live before the if."""

    def fn(cond, a, b):
        if cond:  # noqa: SIM108
            result = add(a, b)
        else:
            result = matmul(a, b)
        return result

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "a" in names
    assert "b" in names
    assert "result" in names


def test_jit_straight_line():
    """A jit function with no loops should produce the same result as eager."""

    def simple_fn(A, B):
        C = matmul(A, B)
        return C  # noqa: RET504

    @jit
    def opt_fn(A, B):
        C = matmul(A, B)
        return C  # noqa: RET504

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[5, 6], [7, 8]]))

    finch_assert_allclose(opt_fn(A, B), simple_fn(A, B))


def test_jit_straight_line_inserted_code(file_regression):
    def opt_fn(A, B):
        C = matmul(A, B)
        return C  # noqa: RET504

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_eager_only_operation_warns_and_computes():
    def simple_fn(A):
        return finch.linalg.inv(A)

    @jit
    def opt_fn(A):
        return finch.linalg.inv(A)

    A = asarray(np.array([[1.0, 2.0], [3.0, 5.0]]))

    with pytest.warns(RuntimeWarning, match="inv"):
        result = opt_fn(A)

    finch_assert_allclose(result, simple_fn(A))


def test_jit_return_expr():
    """A jit function with no loops should produce the same result as eager."""

    def simple_fn(A, B):
        return matmul(A, B), matmul(A, B)

    @jit
    def opt_fn(A, B):
        return matmul(A, B), matmul(A, B)

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[5, 6], [7, 8]]))

    finch_assert_allclose(opt_fn(A, B), simple_fn(A, B))


def test_jit_return_expr_inserted_code(file_regression):
    def opt_fn(A, B):
        return matmul(A, B), matmul(A, B)

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_two_independent_ops():
    """Two independent tensor ops whose results are both used."""

    def simple_fn(A, B, C):
        D = matmul(A, B)
        E = add(A, C)
        F = add(D, E)
        return F  # noqa: RET504

    @jit
    def opt_fn(A, B, C):
        D = matmul(A, B)
        E = add(A, C)
        F = add(D, E)
        return F  # noqa: RET504

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    C = asarray(np.array([[1, 1], [1, 1]]))

    finch_assert_allclose(opt_fn(A, B, C), simple_fn(A, B, C))


def test_jit_two_independent_ops_inserted_code(file_regression):
    def opt_fn(A, B, C):
        D = matmul(A, B)
        E = add(A, C)
        F = add(D, E)
        return F  # noqa: RET504

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


@pytest.mark.parametrize(
    ("operand", "bound"),
    [
        pytest.param(ConstantScalar(2.0), 0, id="constant_scalar"),
        pytest.param(1.0, 0, id="specializable_literal"),
        pytest.param(2.0, 1, id="runtime_literal"),
    ],
)
def test_a_constant_operand_is_not_bound_as_a_tensor(operand, bound):
    """A constant costs no runtime binding, however it was written.

    Everything is bound as a table on the way in; `inline_constant_scalars`
    then replaces the constants with their values and drops those bindings. A
    value the optimizer cannot act on stays a binding, which is the point.
    """
    capture = _CollectBindings(LogicCompiler(NotationInterpreter()))
    executor = LogicExecutor(
        DefaultLogicFactorizer(DefaultLoopOrderer(DefaultLogicFormatter(capture)))
    )
    arr = np.arange(3.0)
    with finch.with_default_scheduler(LogicNormalizer(executor)):
        result = finch.compute(finch.defer(asarray(arr)) + operand)
    finch_assert_allclose(result, arr + float(np.asarray(operand)))
    scalars = [
        t
        for bindings in capture.all_bindings
        for t in bindings.values()
        if isinstance(t, ScalarFType)
    ]
    assert len(scalars) == bound


def test_maybedefer_defers_every_tensor():
    """`maybedefer` makes no judgement about fills -- it only defers tensors."""
    A = asarray(np.arange(3.0))
    (lazy_A, lazy_c, plain) = maybedefer((A, ConstantScalar(2.0), 2.0))
    assert isinstance(lazy_A, LazyTensor)
    assert isinstance(lazy_c, LazyTensor)
    assert plain == 2.0


class _CollectBindings(LogicCapture):
    """A LogicCapture that keeps the bindings of every lowering, not just the last."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.all_bindings: list[dict] = []

    def lower(self, prgm, bindings, stats, stats_factory):
        self.all_bindings.append(bindings.copy())
        return super().lower(prgm, bindings, stats, stats_factory)


def test_jit_inlines_literal_operands():
    """A specializable literal in a jit body reaches the kernel as a constant."""

    @jit
    def opt_fn(A):
        return add(A, 1.0)

    capture = _CollectBindings(LogicCompiler(NotationInterpreter()))
    executor = LogicExecutor(
        DefaultLogicFactorizer(DefaultLoopOrderer(DefaultLogicFormatter(capture)))
    )

    arr = np.arange(3.0)
    with finch.with_default_scheduler(LogicNormalizer(executor)):
        result = opt_fn(asarray(arr))
    finch_assert_allclose(result, arr + 1.0)
    scalars = [
        t
        for bindings in capture.all_bindings
        for t in bindings.values()
        if isinstance(t, ScalarFType)
    ]
    assert not scalars


@pytest.mark.parametrize("trip_counts", [(1, 2, 4, 8, 16)])
def test_jit_constant_folded_in_a_loop_does_not_grow_the_kernel_cache(trip_counts):
    """A constant incremented in a loop must not cost a kernel per iteration.

    `n` folds to a fresh value on every trip. Without the demotion in
    `maybedefer` each fresh value is a fresh literal, so the program -- and the
    kernel compiled from it -- differs per iteration and the cache grows without
    bound in the trip count.
    """

    @jit
    def const_loop(A, k):
        n = 1
        B = A
        for _i in range(k):
            n = n + 1
            B = add(B, n)
        return B

    arr = np.arange(3.0)
    counts = []
    for k in trip_counts:
        executor = LogicExecutor(
            DefaultLogicFactorizer(
                DefaultLoopOrderer(
                    DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
                )
            ),
            cache=True,
        )
        with finch.with_default_scheduler(LogicNormalizer(executor)):
            result = const_loop(asarray(arr), k)
        want = arr.copy()
        n = 1
        for _ in range(k):
            n += 1
            want = want + n
        finch_assert_allclose(result, want)
        counts.append(len(executor.cached_kernels))

    # Flat, not linear: once the loop body has been compiled the cache stops
    # growing, however many more trips are added.
    assert counts[-1] == counts[-2] == counts[len(counts) // 2], counts


def test_jit_scalar_loop():
    """A loop with a scalar iteration count and tensor accumulation."""

    def simple_fn(A, n):
        B = A
        for _i in range(n):
            B = add(B, A)
        return B

    @jit
    def opt_fn(A, n):
        B = A
        for _i in range(n):
            B = add(B, A)
        return B

    A = asarray(np.array([[1, 0], [0, 1]], dtype=float))

    finch_assert_allclose(opt_fn(A, 3), simple_fn(A, 3))


def test_jit_scalar_loop_inserted_code(file_regression):
    def opt_fn(A, n):
        B = A
        for _i in range(n):
            B = add(B, A)
        return B

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_dependent_loop():
    """A loop with an iterator that depends on a computation."""

    def simple_fn(A, n):
        B = A
        for _i in range(sum(B).item()):
            B = add(B, A)
        return B

    @jit
    def opt_fn(A, n):
        B = A
        for _i in range(sum(B).item()):
            B = add(B, A)
        return B

    A = asarray(np.array([[1, 0], [0, 1]], dtype=int))

    finch_assert_allclose(opt_fn(A, 3), simple_fn(A, 3))


def test_jit_dependent_loop_inserted_code(file_regression):
    def opt_fn(A, n):
        B = A
        for _i in range(sum(B).item()):
            B = add(B, A)
        return B

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_if_branch():
    """A jit function with an if/else over tensor ops."""

    def simple_fn(A, B, use_matmul):
        if use_matmul:  # noqa: SIM108
            result = matmul(A, B)
        else:
            result = add(A, B)
        return result

    @jit
    def opt_fn(A, B, use_matmul):
        if use_matmul:  # noqa: SIM108
            result = matmul(A, B)
        else:
            result = add(A, B)
        return result

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))

    finch_assert_allclose(opt_fn(A, B, True), simple_fn(A, B, True))
    finch_assert_allclose(opt_fn(A, B, False), simple_fn(A, B, False))


def test_jit_if_branch_inserted_code(file_regression):
    def opt_fn(A, B, use_matmul):
        if use_matmul:  # noqa: SIM108
            result = matmul(A, B)
        else:
            result = add(A, B)
        return result

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_while():
    """A jit function with a while loop."""

    def simple_fn(A, B, n):
        C = A
        while n > 0:
            C = add(C, B)
            n = n - 1
        return C

    @jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = add(C, B)
            n = n - 1
        return C

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    n = 3

    finch_assert_allclose(opt_fn(A, B, n), simple_fn(A, B, n))


def test_jit_while_inserted_code(file_regression):
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = add(C, B)
            n = n - 1
        return C

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_module_function():
    """A jit function with a function from a module."""

    def simple_fn(A, B, n):
        C = A
        while n > 0:
            C = finch.interface.add(C, B)
            n = n - 1
        return C

    @jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = finch.interface.add(C, B)
            n = n - 1
        return C

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    n = 3

    finch_assert_allclose(opt_fn(A, B, n), simple_fn(A, B, n))


def test_jit_module_function_inserted_code(file_regression):
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = finch.interface.add(C, B)
            n = n - 1
        return C

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def test_jit_local_module_function():
    """A jit function with a function from a module."""

    xp = finch.interface

    def simple_fn(A, B, n):
        C = A
        while n > 0:
            C = xp.add(C, B)
            n = n - 1
        return C

    @jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = xp.add(C, B)
            n = n - 1
        return C

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    n = 3

    finch_assert_allclose(simple_fn(A, B, n), opt_fn(A, B, n))


def test_jit_local_module_function_inserted_code(file_regression):
    xp = finch.interface

    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = xp.add(C, B)
            n = n - 1
        return C

    file_regression.check(_transformed_jit_source(opt_fn), extension=".py")


def _transparent_jit_source(fn):
    fused_fn = wrap_calls(parse_fused_function(fn, closure_as_params=True))
    transformed_fn = insert_lazy_and_compute(fused_fn, transparent=True)
    assert isinstance(transformed_fn, fzd.Function)
    return ast.unparse(fused_function_to_python_ast(transformed_fn)) + "\n"


@pytest.fixture
def scheduler_calls(monkeypatch):
    """Counts how many times `compute` invokes the scheduler."""
    fuse_module = importlib.import_module("finch.interface.fuse")
    get_scheduler = fuse_module.get_default_scheduler
    calls = []

    def counting_scheduler():
        scheduler = get_scheduler()

        def run(prgm):
            calls.append(prgm)
            return scheduler(prgm)

        return run

    monkeypatch.setattr(fuse_module, "get_default_scheduler", counting_scheduler)
    return calls


def _whose_turn(xp, S):
    return xp.sum(S)


def _generate_child(xp, S, W):
    turn = _whose_turn(xp, S)
    return S + W, turn * 2


def _helper_chain(xp, A, B):
    C, turn = _generate_child(xp, A, B)
    return xp.matmul(C, B) + turn


def test_jit_traces_into_helpers(scheduler_calls):
    """Straight-line helpers are fused with their caller into a single kernel."""

    @jit
    def opt_fn(A, B):
        D = _helper_chain(finch, A, B)
        return D  # noqa: RET504

    A = asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = asarray(np.array([[1.0, 0.0], [2.0, 1.0]]))

    result = opt_fn(A, B)
    assert len(scheduler_calls) == 1
    finch_assert_allclose(result, _helper_chain(finch, A, B))


def test_jit_transparent_helper_inserted_code(file_regression):
    file_regression.check(_transparent_jit_source(_generate_child), extension=".py")


def _matrix_power(A, n):
    if n == 1:
        return A
    return matmul(A, _matrix_power(A, n - 1))


def test_jit_recursive_helper():
    @jit
    def opt_fn(A):
        return _matrix_power(A, 3)

    A = asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

    finch_assert_allclose(opt_fn(A), _matrix_power(A, 3))
    assert _transparent_cache[_matrix_power.__code__] is not None


class _NormMixin:
    def _norm(self, xp, b):
        return xp.sqrt(xp.sum(b * b))


class _Solver(_NormMixin):
    def residual(self, xp, b, *, scale=2.0):
        n = self._norm(xp, b)
        return n * scale


def test_jit_method_helpers():
    """Bound methods, mixins and keyword-only defaults are traced into."""
    solver = _Solver()

    @jit
    def opt_fn(b):
        return solver.residual(finch, b)

    b = asarray(np.array([3.0, 4.0]))

    finch_assert_allclose(opt_fn(b), solver.residual(finch, b))
    assert _transparent_cache[_Solver.residual.__code__] is not None
    assert _transparent_cache[_NormMixin._norm.__code__] is not None


def _double(u):
    return u + u


def _square(u):
    return u * u


def _pick_flux(name):
    if name == "double":
        return _double
    return _square


def test_jit_function_valued_local():
    def simple_fn(u, name):
        flux = _pick_flux(name)
        return flux(u)

    @jit
    def opt_fn(u, name):
        flux = _pick_flux(name)
        return flux(u)

    u = asarray(np.array([1.0, 2.0, 3.0]))

    finch_assert_allclose(opt_fn(u, "double"), simple_fn(u, "double"))
    finch_assert_allclose(opt_fn(u, "square"), simple_fn(u, "square"))


def _integrate(f, y, steps):
    for i in range(steps):
        y = y + f(i, y)
    return y


def test_jit_lambda_closing_over_tensors():
    """A lambda closing over tensors compiles once and is reused for new values."""

    def simple_fn(A, B, y):
        return _integrate(lambda t, y: matmul(A, y) + B, y, 2)

    @jit
    def opt_fn(A, B, y):
        return _integrate(lambda t, y: matmul(A, y) + B, y, 2)

    A = asarray(np.array([[0.5, 0.0], [0.0, 0.5]]))
    B = asarray(np.array([1.0, 2.0]))
    y = asarray(np.array([1.0, 1.0]))

    finch_assert_allclose(opt_fn(A, B, y), simple_fn(A, B, y))
    A_2 = asarray(np.array([[1.0, 1.0], [0.0, 1.0]]))
    finch_assert_allclose(opt_fn(A_2, y, B), simple_fn(A_2, y, B))
    lambda_codes = [
        code
        for code in _transparent_cache
        if code.co_name == "<lambda>" and code.co_filename == __file__
    ]
    assert len(lambda_codes) == 1
    assert _transparent_cache[lambda_codes[0]] is not None


def _eager_helper(x):
    k = int(finch.max(x))
    try:
        return x * k
    except TypeError:
        return x


def test_jit_opaque_helper_receives_computed_tensors():
    """Helpers that cannot be traced into receive computed tensors."""

    def simple_fn(A):
        B = A + 1
        return _eager_helper(B), np.asarray(B)

    @jit
    def opt_fn(A):
        B = A + 1
        return _eager_helper(B), np.asarray(B)

    A = asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

    result, array = opt_fn(A)
    expected, expected_array = simple_fn(A)
    finch_assert_allclose(result, expected)
    np.testing.assert_allclose(array, expected_array)
    assert _transparent_cache[_eager_helper.__code__] is None


def test_jit_subscripts_break_and_expression_statements(capsys):
    def simple_fn(A, meta):
        x = A[0, 1:]
        limit = meta["limit"]
        i = 0
        while i < 10:
            i = i + 1
            if i > limit:
                break
        print(i)
        return x * i

    @jit
    def opt_fn(A, meta):
        x = A[0, 1:]
        limit = meta["limit"]
        i = 0
        while i < 10:
            i = i + 1
            if i > limit:
                break
        print(i)
        return x * i

    A = asarray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    finch_assert_allclose(opt_fn(A, {"limit": 3}), simple_fn(A, {"limit": 3}))
    assert capsys.readouterr().out == "4\n4\n"


def test_jit_lazy_tensor_item():
    @jit
    def opt_fn(A):
        total = sum(A)
        return total.item() + 1

    A = asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert opt_fn(A) == 11.0


def test_jit_keyword_only_and_default_parameters():
    def simple_fn(A, B=None, *, scale=2.0):
        return A * scale

    @jit
    def opt_fn(A, B=None, *, scale=2.0):
        return A * scale

    A = asarray(np.array([1.0, 2.0]))

    finch_assert_allclose(opt_fn(A), simple_fn(A))
    finch_assert_allclose(opt_fn(A, scale=3.0), simple_fn(A, scale=3.0))


def _allclose(xp, a, b):
    return xp.all(xp.abs(a - b) <= 1e-8)


def _prune(xp, matrix, threshold):
    mask = (matrix >= threshold) | (matrix == xp.max(matrix, axis=0))
    return matrix * mask


def test_jit_branch_on_traced_helper_result():
    """Conditions are computed even when they come from a traced helper."""

    def simple_fn(A, iterations):
        current = A
        for i in range(iterations):
            previous = current
            current = _prune(finch, matmul(current, current), 0.1)
            if i > 0 and _allclose(finch, current, previous):
                break
        return current

    @jit
    def opt_fn(A, iterations):
        current = A
        for i in range(iterations):
            previous = current
            current = _prune(finch, matmul(current, current), 0.1)
            if i > 0 and _allclose(finch, current, previous):
                break
        return current

    A = asarray(np.array([[0.5, 0.5], [0.0, 1.0]]))

    finch_assert_allclose(opt_fn(A, 5), simple_fn(A, 5))
