import pytest

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype
from finch.compile.looplets import Lookup, Run, Switch, Thunk
from finch.compile.lower import AssemblyContext, LoopletContext, SymbolicExtent
from finch.symbolic import PostOrderDFS


@pytest.mark.parametrize(
    "start, end, expected, point",
    [
        (2, 5, 39, False),
        (2, 2, 0, False),
        (2, 3, 12, False),
        (2, 3, 12, True),
    ],
)
def test_lookup_and_run(start, end, expected, point):
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    position = ntn.Variable("position", ftype(int))
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(0)))
    lookup_type = ntn.Full(position, (ntn.Literal(end - start),)).result_type
    run_type = ntn.Full(ntn.Literal(10), (ntn.Literal(end - start),)).result_type

    def lookup(ctx, idx):
        ctx.exec(asm.Assign(ctx.ctx(position), ctx.ctx(idx)))
        return Run(ntn.Full(position))

    body = ntn.Assign(
        result,
        ntn.Call(
            ntn.Literal(ffuncs.add),
            (
                result,
                ntn.Unwrap(
                    ntn.Access(
                        ntn.Looplet(Lookup(lookup), lookup_type), ntn.Read(), (idx,)
                    )
                ),
                ntn.Unwrap(
                    ntn.Access(
                        ntn.Looplet(Run(ntn.Full(ntn.Literal(10))), run_type),
                        ntn.Read(),
                        (idx,),
                    )
                ),
            ),
        ),
    )
    ext = SymbolicExtent(ntn.Literal(start), ntn.Literal(end))
    if point:
        ext = SymbolicExtent.point(ntn.Literal(start))
    LoopletContext(ctx, idx)(ext, body)
    interpreter = asm.AssemblyInterpreter()
    interpreter(asm.Block(ctx.emit()))
    assert interpreter(ctx(result)) == expected


@pytest.mark.parametrize("with_run", [False, True])
@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize(
    "start, end, point",
    [(2, 5, False), (2, 2, False), (2, 3, False), (3, 4, True)],
)
def test_loop_lowering_preserves_iterations(with_run, guarded, start, end, point):
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(0)))
    tensor_type = ntn.Full(ntn.Literal(10), (ntn.Literal(end - start),)).result_type
    value: ntn.NotationExpression = ntn.Literal(10)
    if with_run:
        value = ntn.Unwrap(
            ntn.Access(
                ntn.Looplet(Run(ntn.Full(value)), tensor_type),
                ntn.Read(),
                (idx,),
            )
        )
    body: ntn.NotationStatement = ntn.Assign(
        result, ntn.Call(ntn.Literal(ffuncs.add), (result, value))
    )
    if guarded:
        body = ntn.If(ntn.Call(ntn.Literal(ffuncs.eq), (idx, ntn.Literal(3))), body)
    ext = (
        SymbolicExtent.point(ntn.Literal(start))
        if point
        else SymbolicExtent(ntn.Literal(start), ntn.Literal(end))
    )
    LoopletContext(ctx, idx)(ext, body)
    program = asm.Block(ctx.emit())
    interpreter = asm.AssemblyInterpreter()
    interpreter(program)
    assert interpreter(ctx(result)) == sum(
        10 for i in range(start, end) if not guarded or i == 3
    )
    assert any(isinstance(node, asm.ForLoop) for node in PostOrderDFS(program)) != point


def test_run_annihilator_simplifies_before_lookup():
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    tensor_type = ntn.Full(ntn.Literal(0), (ntn.Literal(3),)).result_type

    def lookup(ctx, idx):
        pytest.fail("An annihilated lookup should not be lowered")

    args = tuple(
        ntn.Unwrap(ntn.Access(ntn.Looplet(tns, tensor_type), ntn.Read(), (idx,)))
        for tns in (Run(ntn.Full(ntn.Literal(0))), Lookup(lookup))
    )
    body = ntn.LoopletSimplify()(
        ntn.Assign(result, ntn.Call(ntn.Literal(ffuncs.mul), args))
    )
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(1)))
    LoopletContext(ctx, idx)(SymbolicExtent(ntn.Literal(0), ntn.Literal(3)), body)
    interpreter = asm.AssemblyInterpreter()
    interpreter(asm.Block(ctx.emit()))
    assert interpreter(ctx(result)) == 0


@pytest.mark.parametrize(
    "start, end, expected, point",
    [
        (2, 5, 26, False),
        (2, 2, 0, False),
        (2, 3, 4, False),
        (2, 3, 4, True),
    ],
)
def test_lookup_switch_thunk(start, end, expected, point):
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    value = ntn.Variable("value", ftype(int))
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(0)))
    lookup_type = ntn.Full(value, (ntn.Literal(end - start),)).result_type

    def lookup(ctx, idx):
        return Switch(
            ctx.ctx(ntn.Call(ntn.Literal(ffuncs.lt), (idx, ntn.Literal(3)))),
            Thunk(
                preamble=lambda ctx, idx: asm.Assign(
                    ctx.ctx(value),
                    ctx.ctx(ntn.Call(ntn.Literal(ffuncs.mul), (idx, ntn.Literal(2)))),
                ),
                body=lambda ctx, ext: Run(ntn.Full(value)),
            ),
            Run(ntn.Full(ntn.Literal(11))),
        )

    body = ntn.Assign(
        result,
        ntn.Call(
            ntn.Literal(ffuncs.add),
            (
                result,
                ntn.Unwrap(
                    ntn.Access(
                        ntn.Looplet(Lookup(lookup), lookup_type), ntn.Read(), (idx,)
                    )
                ),
            ),
        ),
    )
    ext = SymbolicExtent(ntn.Literal(start), ntn.Literal(end))
    if point:
        ext = SymbolicExtent.point(ntn.Literal(start))
    LoopletContext(ctx, idx)(ext, body)
    interpreter = asm.AssemblyInterpreter()
    interpreter(asm.Block(ctx.emit()))
    assert interpreter(ctx(result)) == expected
