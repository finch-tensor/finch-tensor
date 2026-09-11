import pytest

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype
from finch.compile.looplets import Lookup, Run
from finch.compile.lower import AssemblyContext, LoopletContext, SymbolicExtent


@pytest.mark.parametrize("start, end, expected", [(2, 5, 39), (2, 2, 12)])
def test_terminal_lookup_and_run(start, end, expected):
    idx = ntn.Variable("i", ftype(int))
    result = ntn.Variable("result", ftype(int))
    position = ntn.Variable("position", ftype(int))
    ctx = AssemblyContext()
    ctx(ntn.Assign(result, ntn.Literal(0)))

    def lookup(ctx, idx):
        ctx.exec(asm.Assign(ctx.ctx(position), ctx.ctx(idx)))
        return ntn.Full(position)

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
