import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import Tensor, ffuncs, ftype, is_dynamic
from finch.compile import NotationCompiler, make_extent
from finch.compile.looplets import Run
from finch.finch_notation.interpreter import FullView
from finch.symbolic import PostOrderDFS, Reflector, ScopedDict
from finch.tensor import BufferizedNDArray


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_full_interpreter(shape):
    value = ntn.Variable("value", ftype(np.int64))
    full = ntn.Full(value, tuple(ntn.Literal(dim) for dim in shape))
    interpreter = ntn.NotationInterpreter(bindings=ScopedDict({"value": np.int64(7)}))
    tensor = interpreter(full)
    assert isinstance(tensor, FullView)
    assert isinstance(tensor, Tensor)
    assert tensor.fill_value == 7
    np.testing.assert_array_equal(tensor.to_numpy(), np.full(shape, 7))
    assert tensor.shape == shape
    assert tensor.ftype == full.result_type
    for r, dim in enumerate(shape):
        assert interpreter(ntn.Dimension(full, ntn.Literal(r))) == dim
    assert full.result_type.shape_type == tuple(ftype(dim) for dim in shape)
    assert is_dynamic(full.result_type.fill_type)
    access = ntn.Access(full, ntn.Read(), tuple(ntn.Literal(0) for _ in shape))
    assert interpreter(ntn.Unwrap(access)) == 7
    view = tensor.access((0,) * len(shape))
    assert view.shape == ()
    assert view.ftype.shape_type == ()
    assert view.item() == 7
    assert ntn.Full.from_children(*full.children) == full
    with pytest.raises(TypeError, match="read-only"):
        interpreter(ntn.Access(full, ntn.Update(ntn.Literal(ffuncs.add)), ()))


def test_full_compiler():
    value = ntn.Variable("value", ftype(np.int64))
    buf = BufferizedNDArray.from_numpy(np.zeros((2, 3), dtype=np.int64))
    output = ntn.Variable("output", buf.ftype)
    slot = ntn.Slot("output_slot", buf.ftype)
    i = ntn.Variable("i", ftype(np.int64))
    j = ntn.Variable("j", ftype(np.int64))
    shape = (ntn.Literal(np.int64(2)), ntn.Literal(np.int64(3)))
    full = ntn.Full(value, shape)
    op = ntn.Literal(ffuncs.overwrite)
    program = ntn.Module(
        (
            ntn.Function(
                ntn.Variable(
                    "read_full",
                    asm.AssemblyKernelFType(
                        "read_full",
                        (output.result_type, value.result_type),
                        buf.ftype,
                    ),
                ),
                (output, value),
                ntn.Block(
                    (
                        ntn.Unpack(slot, output),
                        ntn.Declare(slot, ntn.Literal(np.int64(0)), op, shape),
                        ntn.Loop(
                            i,
                            ntn.Call(
                                ntn.Literal(make_extent),
                                (
                                    ntn.Literal(np.int64(0)),
                                    ntn.Dimension(full, ntn.Literal(0)),
                                ),
                            ),
                            ntn.Loop(
                                j,
                                ntn.Call(
                                    ntn.Literal(make_extent),
                                    (
                                        ntn.Literal(np.int64(0)),
                                        ntn.Dimension(full, ntn.Literal(1)),
                                    ),
                                ),
                                ntn.Increment(
                                    ntn.Access(slot, ntn.Update(op), (i, j)),
                                    ntn.Unwrap(ntn.Access(full, ntn.Read(), (i, j))),
                                ),
                            ),
                        ),
                        ntn.Freeze(slot, op),
                        ntn.Repack(slot, output),
                        ntn.Return(output),
                    )
                ),
            ),
        )
    )
    interpreted = ntn.NotationInterpreter()(program)
    compiled = asm.AssemblyInterpreter()(NotationCompiler(Reflector())(program))  # ty: ignore[invalid-argument-type]
    for value in (np.int64(7), np.int64(11)):
        for module in (interpreted, compiled):
            actual = module.read_full(buf, value).to_numpy()
            np.testing.assert_array_equal(actual, np.full((2, 3), value))


def test_reads_of_a_literal_run_fold_to_the_literal():
    full = ntn.Full(ntn.Literal(0))
    access = ntn.Unwrap(
        ntn.Access(ntn.Looplet(Run(full), full.result_type), ntn.Read(), ())
    )
    assert ntn.LoopletSimplify.simplify(access) == ntn.Literal(0)
    term = ntn.Call(ntn.Literal(ffuncs.mul), (access, ntn.Literal(5)))
    assert ntn.LoopletSimplify()(term) == ntn.Literal(0)
    dynamic = ntn.Unwrap(
        ntn.Access(
            Run(ntn.Full(ntn.Variable("value", ftype(np.int64)))),  # ty: ignore[invalid-argument-type]
            ntn.Read(),
            (),
        )
    )
    assert ntn.LoopletSimplify.simplify(dynamic) is None
    term = ntn.Call(ntn.Literal(ffuncs.mul), (dynamic, ntn.Literal(5)))
    assert dynamic in PostOrderDFS(ntn.LoopletSimplify()(term))


@pytest.mark.parametrize(
    "op, val, dropped",
    [
        (ffuncs.add, 0.0, True),
        (ffuncs.mul, 1.0, True),
        (ffuncs.max, -np.inf, True),
        (ffuncs.add, 1.0, False),
        (ffuncs.overwrite, 0.0, False),
    ],
)
def test_increments_by_the_identity_are_dropped(op, val, dropped):
    tns = ntn.Variable("A", ftype(np.float64))
    i = ntn.Variable("i", ftype(np.intp))
    lhs = ntn.Access(tns, ntn.Update(ntn.Literal(op)), (i,))
    full = ntn.Full(ntn.Literal(val))
    run = ntn.Unwrap(
        ntn.Access(ntn.Looplet(Run(full), full.result_type), ntn.Read(), ())
    )
    for rhs in (ntn.Literal(val), run):
        expected = ntn.Block(())
        assert ntn.LoopletSimplify()(ntn.Increment(lhs, rhs)) == (
            expected if dropped else ntn.Increment(lhs, ntn.Literal(val))
        )

    # A dynamic fill may not be the identity, so its increment is kept.
    dynamic = ntn.Unwrap(
        ntn.Access(
            Run(ntn.Full(ntn.Variable("value", ftype(np.float64)))),  # ty: ignore[invalid-argument-type]
            ntn.Read(),
            (),
        )
    )
    assert ntn.LoopletSimplify()(ntn.Increment(lhs, dynamic)) == ntn.Increment(
        lhs, dynamic
    )


def test_loops_without_bodies_are_dropped():
    i = ntn.Variable("i", ftype(np.intp))
    tns = ntn.Variable("A", ftype(np.float64))
    lhs = ntn.Access(tns, ntn.Update(ntn.Literal(ffuncs.add)), (i,))
    ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(0), ntn.Literal(3)))
    loop = ntn.Loop(i, ext, ntn.Increment(lhs, ntn.Literal(0.0)))
    assert ntn.LoopletSimplify()(loop) == ntn.Block(())
    body = ntn.Increment(lhs, ntn.Literal(1.0))
    match ntn.LoopletSimplify()(ntn.Loop(i, ext, body)):
        case ntn.Loop(_, _, kept):
            assert kept == body
        case term:
            raise AssertionError(f"Expected the loop to be kept: {term}")


def test_loops_over_a_run_of_the_identity_are_dropped():
    # A sparse level lowers its gaps to a run of the fill value, so counting the
    # nonzeros of a gap increments by ne(0.0, 0.0), which must not leave a loop.
    i = ntn.Variable("i", ftype(np.intp))
    tns = ntn.Variable("A", ftype(np.int64))
    lhs = ntn.Access(tns, ntn.Update(ntn.Literal(ffuncs.add)), (i,))
    full = ntn.Full(ntn.Literal(0.0))
    run = ntn.Unwrap(
        ntn.Access(ntn.Looplet(Run(full), full.result_type), ntn.Read(), (i,))
    )
    rhs = ntn.Call(ntn.Literal(ffuncs.ne), (run, ntn.Literal(0.0)))
    ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(0), ntn.Literal(3)))
    loop = ntn.Loop(i, ext, ntn.Increment(lhs, rhs))
    assert ntn.LoopletSimplify()(loop) == ntn.Block(())


def test_full_scalar():
    full = ntn.Full(ntn.Literal(np.int64(4)))
    interpreter = ntn.NotationInterpreter()
    assert interpreter(full).ftype == full.result_type
    assert interpreter(ntn.Unwrap(full)) == 4
    assert str(full) == "full(4, ())"
