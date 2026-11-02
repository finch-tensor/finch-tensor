from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs
from finch.compile.lower import LoopletContext, LoopletPass, SymbolicExtent
from finch.finch_notation.proves import prove
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite


class Looplet(ABC):
    @property
    @abstractmethod
    def pass_request(self): ...

    def truncate(self, ctx, idx, ext):
        """Restrict the looplet to `ext`, a sub-extent of the one it covers."""
        return self


@dataclass
class Thunk(Looplet):
    preamble: Any = None
    body: Any = None
    epilogue: Any = None

    @property
    def pass_request(self):
        return ThunkPass()


class ThunkPass(LoopletPass):
    @property
    def priority(self):
        return 8

    def __call__(self, ctx: LoopletContext, idx, ext, body):
        def thunk_body(ctx, node: ntn.NotationNode):
            match node:
                case ntn.Access(Thunk() as thnk, mode, (j, *idxs)) if j == idx:
                    if (preamble := thnk.preamble) is not None:
                        ctx.exec(preamble(ctx, idx))
                    if (epilogue := thnk.epilogue) is not None:
                        ctx.post(epilogue(ctx, idx))
                    return ntn.Access(thnk.body(ctx, ext), mode, (j, *idxs))

        ctx_2 = ctx.scope()
        body = Rewrite(PostWalk(lambda x: thunk_body(ctx_2, x)))(body)
        ctx_2(ext, body)
        ctx.exec(asm.Block(ctx_2.emit()))


@dataclass
class Switch(Looplet):
    cond: Any
    if_true: Any
    if_false: Any

    @property
    def pass_request(self):
        return SwitchPass()


class SwitchPass(LoopletPass):
    @property
    def priority(self):
        return 7

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        # Lower one switch at a time, the remaining ones are lowered
        # recursively within each branch.
        for node in PostOrderDFS(body):
            match node:
                case ntn.Access(Switch() as switch, _, (j, *_)) if j == idx:
                    break

        def switch_node(node, branch):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)) if j == idx and tns == switch:
                    return ntn.Access(branch, mode, (j, *idxs))
            return None

        ctx_2 = ctx.scope()
        ctx_2(ext, PostWalk(partial(switch_node, branch=switch.if_true))(body))

        ctx_3 = ctx.scope()
        ctx_3(ext, PostWalk(partial(switch_node, branch=switch.if_false))(body))

        ctx.exec(
            asm.Block(
                (
                    asm.IfElse(
                        switch.cond, asm.Block(ctx_2.emit()), asm.Block(ctx_3.emit())
                    ),
                )
            )
        )


@dataclass
class Stepper(Looplet):
    preamble: Any = None
    stop: Callable = lambda ctx, ext: None
    chunk: Looplet | None = None
    next: Callable[[Any], Any] = lambda ctx: None
    body: Callable = lambda ctx, ext: None
    seek: Callable = lambda ctx, start: (_ for _ in ()).throw(
        NotImplementedError("seek not implemented error")
    )

    @property
    def pass_request(self):
        return StepperPass()


class StepperPass(LoopletPass):
    count: int = 1

    @property
    def priority(self):
        return 3

    def __call__(
        self,
        ctx: LoopletContext,
        idx: ntn.Variable,
        ext: SymbolicExtent,
        body,
    ) -> None:

        steppers: list[Stepper] = []

        for node in PostOrderDFS(body):
            match node:
                case Stepper() as st:
                    ctx.exec(st.seek(ctx, ext))
                    steppers.append(st)

        idx_start = ntn.Variable(ctx.freshen("i_start"), idx.result_type)
        ctx.exec(asm.Assign(ctx.ctx(idx_start), ctx.ctx(ext.get_start())))

        ctx_2 = ctx.scope()
        for st in steppers:
            ctx_2.exec(st.preamble(ctx_2))

        # Each iteration covers one phase, which ends at the nearest stepper
        # stop (or at the end of the extent).
        phase_stop = ntn.Variable(ctx.freshen("phase_stop"), idx.result_type)
        ctx_2.exec(
            asm.Assign(
                ctx_2.ctx(phase_stop),
                ctx_2.ctx(
                    ntn.Call(
                        ntn.L(ffuncs.min),
                        (
                            ext.get_end(),
                            *(
                                ntn.Call(
                                    ntn.L(ffuncs.add), (st.stop(ctx_2), ext.get_unit())
                                )
                                for st in steppers
                            ),
                        ),
                    )
                ),
            )
        )
        ext_2 = SymbolicExtent(idx_start, phase_stop)

        # Steppers which reach the end of the phase take their full chunk and
        # advance, the others are truncated to the phase.
        def stepper_body(node: ntn.NotationNode):
            match node:
                case ntn.Access(Stepper() as st, mode, (j, *idxs)) if j == idx:
                    stop = ntn.Call(ntn.L(ffuncs.add), (st.stop(ctx_2), ext.get_unit()))
                    full_chunk = Thunk(
                        body=lambda ctx, ext: st.chunk,
                        epilogue=lambda ctx, idx: st.next(ctx),
                    )
                    truncated_chunk = st.chunk.truncate(  # type: ignore[union-attr]
                        ctx_2,
                        idx,
                        SymbolicExtent(
                            idx_start,
                            ntn.Call(ntn.L(ffuncs.min), (phase_stop, st.stop(ctx_2))),
                        ),
                    )
                    return ntn.Access(
                        Switch(
                            asm.Call(
                                asm.L(ffuncs.eq),
                                (ctx_2.ctx(stop), ctx_2.ctx(phase_stop)),
                            ),
                            full_chunk,
                            truncated_chunk,
                        ),
                        mode,
                        (j, *idxs),
                    )

        ctx_2(ext_2, Rewrite(PostWalk(stepper_body))(body))
        ctx_2.exec(asm.Assign(ctx_2.ctx(idx_start), ctx_2.ctx(phase_stop)))

        ctx.exec(
            asm.WhileLoop(
                asm.Call(
                    asm.L(ffuncs.lt), (ctx.ctx(idx_start), ctx.ctx(ext.get_end()))
                ),
                asm.Block(ctx_2.emit()),
            )
        )


@dataclass
class Sequence(Looplet):
    head: Callable
    split: Callable
    tail: Callable

    @property
    def pass_request(self):
        return SequencePass()

    def truncate(self, ctx: LoopletContext, idx, ext: SymbolicExtent):
        split = self.split(ctx, ext)
        if prove(ntn.Call(ntn.L(ffuncs.ge), (split, ext.get_end()))):
            return self.head(ctx, idx)
        if prove(ntn.Call(ntn.L(ffuncs.le), (split, ext.get_start()))):
            return self.tail(ctx, idx)
        return self


class SequencePass(LoopletPass):
    """
    Lowers one Sequence looplet at the time.
    """

    @property
    def priority(self):
        return 4

    @classmethod
    def get_sequence_variations(
        cls, ctx, seqs: list[Sequence], heads, tails, ext: SymbolicExtent
    ) -> list[tuple]:
        ext_start, ext_end = ext.get_start(), ext.get_end()

        match seqs:
            case []:
                return [(heads, tails, ext_start, ext_end)]
            case [Sequence() as seq, *tail]:
                split = seq.split(ctx, ext)
                left = cls.get_sequence_variations(
                    ctx,
                    tail,
                    heads + [seq],
                    tails,
                    SymbolicExtent(
                        ext_start,
                        ntn.Call(ntn.L(ffuncs.min), (ext_end, split)),
                    ),
                )
                right = cls.get_sequence_variations(
                    ctx,
                    tail,
                    heads,
                    tails + [seq],
                    SymbolicExtent(
                        ntn.Call(ntn.L(ffuncs.max), (ext_start, split)),
                        ext_end,
                    ),
                )
                return left + right
            case other:
                raise Exception(f"Invalid sequence: {other}")

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        found_seqs: list[Sequence] = []

        for node in PostOrderDFS(body):
            match node:
                case Sequence() as seq:
                    found_seqs.append(seq)

        def sequence_node(
            node: ntn.NotationNode, heads: set[Sequence], tails: set[Sequence]
        ):
            match node:
                case ntn.Access(Sequence() as tns, mode, (j, *idxs)) if j == idx:
                    if tns in heads:
                        new_tns = tns.head(ctx, idx)  # type: ignore[call-arg]
                    elif tns in tails:
                        new_tns = tns.tail(ctx, idx)
                    else:
                        raise Exception(f"Seq: {tns} not present.")
                    return ntn.Access(new_tns, mode, (j, *idxs))

        variations = self.get_sequence_variations(ctx, found_seqs, [], [], ext)
        blocks = []

        for v in variations:
            heads, tails, subext_start, subext_end = v
            sub_body = PostWalk(partial(sequence_node, heads=heads, tails=tails))(body)
            subext = SymbolicExtent(subext_start, subext_end)
            sub_ctx = ctx.scope()
            sub_ctx(subext, self.looplet_simplify(sub_body))
            blocks += sub_ctx.emit()

        ctx.exec(asm.Block(tuple(blocks)))


@dataclass
class Run(Looplet):
    body: Any

    @property
    def pass_request(self):
        return RunPass()


class RunPass(LoopletPass):
    @property
    def priority(self):
        return 6

    def __call__(self, ctx, idx, ext, body):
        def run_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Run):
                        return ntn.Access(
                            Leaf(
                                lambda ctx: ntn.Value(
                                    asm.Literal(tns.body), tns.body.ftype
                                )
                            ),
                            mode,
                            (j, *idxs),
                        )
            return None

        body_2 = PostWalk(run_node)(body)
        ctx_2 = ctx.scope()
        ctx_2(ext, body_2)
        ctx.exec(asm.Block(ctx_2.emit()))


@dataclass
class AcceptRun(Looplet):
    body: Any

    @property
    def pass_request(self):
        return AcceptRunPass()


class AcceptRunPass(LoopletPass):
    @property
    def priority(self):
        return 1


@dataclass
class Lookup(Looplet):
    body: Callable

    @property
    def pass_request(self):
        return LookupPass()


class LookupPass(LoopletPass):
    @property
    def priority(self):
        return 2

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        def lookup_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Lookup):
                        tns_2 = tns.body(
                            ctx,
                            idx,
                        )
                        return ntn.Access(tns_2, mode, (j, *idxs))
            return None

        body_2 = PostWalk(lookup_node)(body)
        ctx_2 = ctx.scope()
        ext_2 = SymbolicExtent.point(idx)
        ctx_2(ext_2, body_2)
        body_3 = asm.Block(ctx_2.emit())

        if ext.is_sym_point():
            ctx.exec(
                asm.Block(
                    (asm.Assign(ctx.ctx(idx), ctx.ctx(ext.get_start())), *body_3.bodies)
                )
            )
        else:
            ctx.exec(
                asm.ForLoop(
                    ctx.ctx(idx),
                    ctx.ctx(ext.get_start()),
                    ctx.ctx(ext.get_end()),
                    body_3,
                )
            )


@dataclass
class Jumper(Looplet):
    preamble: Any = None
    stop: Callable = lambda ctx, ext: None
    chunk: Any = None
    next: Callable = lambda ctx, ext: None
    body: Callable = lambda ctx, ext: None
    seek: Callable = lambda ctx, start: (_ for _ in ()).throw(
        NotImplementedError("seek not implemented error")
    )

    @property
    def pass_request(self):
        return JumperPass()


class JumperPass(LoopletPass):
    @property
    def priority(self):
        return 0


@dataclass
class Leaf(Looplet):
    body: Callable

    @property
    def pass_request(self):
        return LeafPass()


class LeafPass(LoopletPass):
    @property
    def priority(self):
        return 0

    def __call__(self, ctx, idx, ext, body):
        def leaf_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Leaf):
                        return ntn.Access(tns.body(ctx), mode, tuple(idxs))
            return None

        body_2 = PostWalk(leaf_node)(body)
        ctx.ctx(body_2)  # calling AssemblyContext
