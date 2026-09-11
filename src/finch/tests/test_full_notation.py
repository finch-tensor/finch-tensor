import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import Tensor, ffuncs, ftype, is_dynamic
from finch.compile import NotationCompiler, make_extent
from finch.compile.looplets import Run
from finch.finch_notation.interpreter import FullView
from finch.symbolic import Reflector
from finch.tensor import BufferizedNDArray


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_full_interpreter(shape):
    value = ntn.Variable("value", ftype(np.int64))
    full = ntn.Full(value, tuple(ntn.Literal(dim) for dim in shape))
    interpreter = ntn.NotationInterpreter(bindings={"value": np.int64(7)})
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
    assert is_dynamic(full.result_type.fill_value)
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
                ntn.Variable("read_full", buf.ftype),
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
    compiled = asm.AssemblyInterpreter()(NotationCompiler(Reflector())(program))
    for value in (np.int64(7), np.int64(11)):
        for module in (interpreted, compiled):
            actual = module.read_full(buf, value).to_numpy()
            np.testing.assert_array_equal(actual, np.full((2, 3), value))


def test_full_run_annihilator():
    access = ntn.Unwrap(ntn.Access(Run(ntn.Full(ntn.Literal(0))), ntn.Read(), ()))
    term = ntn.Call(ntn.Literal(ffuncs.mul), (access, ntn.Literal(5)))
    assert ntn.LoopletSimplify.simplify(term) == access
    dynamic = ntn.Unwrap(
        ntn.Access(
            Run(ntn.Full(ntn.Variable("value", ftype(np.int64)))),
            ntn.Read(),
            (),
        )
    )
    term = ntn.Call(ntn.Literal(ffuncs.mul), (dynamic, ntn.Literal(5)))
    assert ntn.LoopletSimplify.simplify(term) is None


def test_full_scalar():
    full = ntn.Full(ntn.Literal(np.int64(4)))
    interpreter = ntn.NotationInterpreter()
    assert interpreter(full).ftype == full.result_type
    assert interpreter(ntn.Unwrap(full)) == 4
    assert str(full) == "full(4, ())"
