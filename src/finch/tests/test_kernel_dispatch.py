import dataclasses
import logging

import pytest

import numpy as np
import scipy.sparse as sp

import finch as ft
from finch import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    element,
    ffuncs,
)
from finch.autoschedule import (
    COMPILE_NUMBA_DISPATCH,
    CompilerFormLowerer,
    DefaultLogicFactorizer,
    DefaultLogicFormatter,
    DefaultLogicWorkspacer,
    DefaultLoopOrderer,
    LibraryKernel,
    LogicCompiler,
    LogicExecutor,
    LogicFuseMerger,
    LogicKernelDispatcher,
    LogicNormalizer,
)
from finch.autoschedule.dispatcher import GEMM, canonicalize
from finch.autoschedule.dispatcher.kernels import gemm_pattern
from finch.compile import CompilerMode, NotationCompiler
from finch.finch_assembly import AssemblyInterpreter
from finch.finch_logic import (
    Aggregate,
    Field,
    HardAlias,
    Literal,
    LogicSimplify,
    MapJoin,
    Query,
    Table,
)

i, j, k = Field("i"), Field("j"), Field("k")
x, y, z = Field("x"), Field("y"), Field("z")
A, B, C, D = HardAlias("A"), HardAlias("B"), HardAlias("C"), HardAlias("D")


def mul(*args):
    return MapJoin(Literal(ffuncs.mul), args)


ZERO = Literal(0.0)


def total(arg, *idxs, init=ZERO):
    return Aggregate(Literal(ffuncs.add), init, arg, idxs)


MATMUL = Query(Table(C, (i, j)), total(mul(Table(A, (i, k)), Table(B, (k, j))), k))


def test_names_and_argument_order_dont_change_the_canonical_form():
    renamed = Query(Table(D, (x, y)), total(mul(Table(B, (z, y)), Table(A, (x, z))), z))
    canon, canon_2 = canonicalize(MATMUL), canonicalize(renamed)
    assert canon is not None
    assert canon_2 is not None
    assert canon.term == canon_2.term
    assert canon.operands == canon_2.operands


def test_nesting_scalars_and_the_type_of_the_init_dont_change_the_canonical_form():
    scaled = Query(
        Table(C, (i, j)),
        total(
            mul(Literal(2.0), mul(Table(A, (i, k)), Table(B, (k, j)))),
            k,
            init=Literal(np.float32(0)),
        ),
    )
    canon, canon_2 = canonicalize(MATMUL), canonicalize(scaled)
    assert canon is not None
    assert canon_2 is not None
    assert canon.term == canon_2.term
    assert canon_2.scalars == (Literal(2.0),)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            Query(Table(C, (j, i)), MATMUL.rhs),
            id="transposed-output",
        ),
        pytest.param(
            Query(
                Table(C, (i, j)),
                Aggregate(
                    Literal(ffuncs.max),
                    Literal(-np.inf),
                    MapJoin(Literal(ffuncs.add), (Table(A, (i, k)), Table(B, (k, j)))),
                    (k,),
                ),
            ),
            id="max-plus",
        ),
        pytest.param(
            Query(
                Table(C, (i, j)),
                total(mul(Table(A, (i, k)), Table(B, (k, j)), Table(D, (i, j))), k),
            ),
            id="three-operands",
        ),
        pytest.param(
            Query(
                Table(C, (i, j)),
                total(
                    mul(Literal(2.0), Table(A, (i, k)), Table(B, (k, j))),
                    k,
                    init=Literal(1.0),
                ),
            ),
            id="nonzero-init",
        ),
    ],
)
def test_other_computations_have_other_canonical_forms(query):
    canon, canon_2 = canonicalize(MATMUL), canonicalize(query)
    assert canon is not None
    assert canon_2 is not None
    assert canon.term != canon_2.term


def test_the_transpose_flags_of_gemm_give_four_canonical_forms():
    terms = set()
    for trans_a in (False, True):
        for trans_b in (False, True):
            canon = canonicalize(gemm_pattern(trans_a, trans_b))
            assert canon is not None
            terms.add(canon.term)
    assert len(terms) == 4


def dispatching(*kernels):
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(
                LogicSimplify(
                    DefaultLoopOrderer(
                        LogicKernelDispatcher(
                            DefaultLogicWorkspacer(
                                LogicFuseMerger(
                                    DefaultLogicFormatter(
                                        CompilerFormLowerer(
                                            LogicCompiler(
                                                NotationCompiler(
                                                    AssemblyInterpreter(),
                                                    mode=CompilerMode(
                                                        safe=True, debug=True
                                                    ),
                                                )
                                            )
                                        )
                                    )
                                )
                            ),
                            kernels,
                        )
                    )
                )
            )
        )
    )


def spied(kernel):
    calls = []

    def call(**kwargs):
        calls.append({name: kwargs[name] for name in ("alpha", *kernel.params)})
        return kernel.call(**kwargs)

    return dataclasses.replace(kernel, call=call), calls


def csr(arr):
    mat = sp.csr_matrix(arr)
    elem_t = element(0.0, ft.ftype(np.float64), ft.ftype(np.intp), NumpyBufferFType)
    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(elem_t, NumpyBuffer(mat.data)),
                np.intp(arr.shape[1]),
                NumpyBuffer(mat.indptr.astype(np.intp)),
                NumpyBuffer(mat.indices.astype(np.intp)),
            ),
            np.intp(arr.shape[0]),
        )
    )


rng = np.random.default_rng(0)
M, N, K = 3, 5, 4
A_np, B_np = rng.random((M, K)), rng.random((K, N))
A_np[0, 1] = A_np[2, 3] = 0.0
AT_np, BT_np = np.ascontiguousarray(A_np.T), np.ascontiguousarray(B_np.T)
S_np, X_np, W_np = rng.random((K, K)), rng.random(K), rng.random((M, N))
A_int, B_int = rng.integers(0, 9, (M, K)), rng.integers(0, 9, (K, N))

d = ft.defer
T = ft.matrix_transpose


def square(s):
    return ft.matmul(s, s)


def gram(s):
    return ft.matmul(s, T(s))


def gemm_params(alpha=1, trans_a=False, trans_b=False):
    return {"alpha": alpha, "trans_a": trans_a, "trans_b": trans_b}


@pytest.mark.parametrize(
    ("expr", "expected", "calls"),
    [
        pytest.param(
            lambda: ft.matmul(d(A_np), d(B_np)),
            A_np @ B_np,
            [gemm_params()],
            id="A@B",
        ),
        pytest.param(
            lambda: T(ft.matmul(T(d(B_np)), T(d(A_np)))),
            A_np @ B_np,
            [gemm_params()],
            id="(B.T@A.T).T",
        ),
        pytest.param(
            lambda: 2.0 * ft.matmul(d(A_np), d(B_np)),
            2.0 * A_np @ B_np,
            [gemm_params(alpha=2.0)],
            id="2*(A@B)",
        ),
        pytest.param(
            lambda: ft.matmul(2.0 * d(A_np), d(B_np) * 3.0),
            6.0 * A_np @ B_np,
            [gemm_params(alpha=6.0)],
            id="(2*A)@(B*3)",
        ),
        pytest.param(
            lambda: ft.matmul(T(d(AT_np)), d(B_np)),
            A_np @ B_np,
            [gemm_params(trans_a=True)],
            id="A.T@B",
        ),
        pytest.param(
            lambda: ft.matmul(d(A_np), T(d(BT_np))),
            A_np @ B_np,
            [gemm_params(trans_b=True)],
            id="A@B.T",
        ),
        pytest.param(
            lambda: T(ft.matmul(d(A_np), d(B_np))),
            (A_np @ B_np).T,
            [gemm_params(trans_a=True, trans_b=True)],
            id="(A@B).T",
        ),
        pytest.param(
            lambda: square(d(S_np)),
            S_np @ S_np,
            [gemm_params()],
            id="S@S",
        ),
        pytest.param(
            lambda: ft.matmul(d(A_np.astype(np.float32)), d(B_np.astype(np.float32))),
            A_np.astype(np.float32) @ B_np.astype(np.float32),
            [gemm_params()],
            id="float32",
        ),
        pytest.param(
            lambda: ft.multiply(d(A_np), d(A_np)),
            A_np * A_np,
            [],
            id="elementwise",
        ),
        pytest.param(
            lambda: ft.matmul(d(A_int), d(B_int)),
            A_int @ B_int,
            [],
            id="int64",
        ),
        pytest.param(
            lambda: ft.matmul(d(A_np.astype(np.float32)), d(B_np)),
            A_np.astype(np.float32) @ B_np,
            [],
            id="float32@float64",
        ),
        pytest.param(
            lambda: ft.max(
                ft.expand_dims(d(A_np), axis=2) + ft.expand_dims(d(B_np), axis=0),
                axis=1,
            ),
            np.max(A_np[:, :, None] + B_np[None, :, :], axis=1),
            [],
            id="max-plus",
        ),
        pytest.param(
            lambda: ft.matmul(d(A_np), d(X_np)),
            A_np @ X_np,
            [],
            id="matvec",
        ),
        pytest.param(
            lambda: -ft.matmul(d(A_np), d(B_np)),
            -(A_np @ B_np),
            [],
            id="two-queries",
        ),
        pytest.param(
            lambda: ft.multiply(d(W_np), ft.matmul(d(A_np), d(B_np))),
            W_np * (A_np @ B_np),
            [],
            id="sddmm",
        ),
        pytest.param(
            lambda: ft.matmul(d(csr(A_np)), d(B_np)),
            A_np @ B_np,
            [],
            id="sparse",
        ),
    ],
)
def test_gemm_dispatch(expr, expected, calls):
    gemm, gemm_calls = spied(GEMM)
    out = ft.compute(expr(), ctx=dispatching(gemm))
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6)
    assert gemm_calls == calls


def syrk_pattern():
    product = mul(Table(A, (i, k)), Table(A, (j, k)))
    return Query(Table(C, (i, j)), total(product, k, init=Literal(0)))


SYRK = LibraryKernel(
    name="syrk",
    pattern=syrk_pattern,
    params={},
    accepts=GEMM.accepts,
    call=lambda *, A, alpha: alpha * (A @ A.T),
)


def test_a_kernel_which_reads_an_operand_twice_needs_one_tensor_twice():
    syrk, syrk_calls = spied(SYRK)
    ctx = dispatching(syrk)

    out = ft.compute(gram(d(S_np)), ctx=ctx)
    np.testing.assert_allclose(np.asarray(out), S_np @ S_np.T)
    assert syrk_calls == [{"alpha": 1}]

    # Two tensors with the same values are still two tensors.
    out = ft.compute(ft.matmul(d(S_np), T(d(S_np))), ctx=ctx)
    np.testing.assert_allclose(np.asarray(out), S_np @ S_np.T)
    assert syrk_calls == [{"alpha": 1}]

    out = ft.compute(ft.matmul(d(A_np), T(d(BT_np))), ctx=ctx)
    np.testing.assert_allclose(np.asarray(out), A_np @ B_np)
    assert syrk_calls == [{"alpha": 1}]


def test_compile_numba_dispatch_calls_gemm(caplog):
    with caplog.at_level(logging.DEBUG, logger="finch.autoschedule.dispatcher"):
        out = ft.compute(ft.matmul(d(A_np), d(B_np)), ctx=COMPILE_NUMBA_DISPATCH)
    np.testing.assert_allclose(np.asarray(out), A_np @ B_np)
    assert "Calling gemm" in caplog.text
