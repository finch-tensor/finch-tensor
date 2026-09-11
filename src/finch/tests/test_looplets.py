import pytest

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype
from finch.compile.looplets import Lookup, Run, Switch, Thunk
from finch.compile.lower import AssemblyContext, LoopletContext, SymbolicExtent


@pytest.mark.parametrize("start, end, expected", [(2, 5, 39), (2, 2, 12)])
def test_lookup_and_run(start, end, expected):
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    position = ntn.Variable("position", ftype(int))
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(0)))

    def lookup(ctx, idx):
        ctx.exec(asm.Assign(ctx.ctx(position), ctx.ctx(idx)))
        return Run(ntn.Full(position))

    body = ntn.Assign(
        result,
        ntn.Call(
            ntn.Literal(ffuncs.add),
            (
                result,
                ntn.Unwrap(ntn.Access(Lookup(lookup), ntn.Read(), (idx,))),
                ntn.Unwrap(
                    ntn.Access(Run(ntn.Full(ntn.Literal(10))), ntn.Read(), (idx,))
                ),
            ),
        ),
    )
    ext = SymbolicExtent(ntn.Literal(start), ntn.Literal(end))
    LoopletContext(ctx, idx)(ext, body)
    interpreter = asm.AssemblyInterpreter()
    interpreter(asm.Block(ctx.emit()))
    assert interpreter(ctx(result)) == expected


@pytest.mark.parametrize("start, end, expected", [(2, 5, 26), (2, 2, 4)])
def test_lookup_switch_thunk(start, end, expected):
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    value = ntn.Variable("value", ftype(int))
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(0)))

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
            (result, ntn.Unwrap(ntn.Access(Lookup(lookup), ntn.Read(), (idx,)))),
        ),
    )
    LoopletContext(ctx, idx)(SymbolicExtent(ntn.Literal(start), ntn.Literal(end)), body)
    interpreter = asm.AssemblyInterpreter()
    interpreter(asm.Block(ctx.emit()))
    assert interpreter(ctx(result)) == expected
