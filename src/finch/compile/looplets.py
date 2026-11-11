from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs
from finch.compile.lower import LoopletContext, LoopletPass, SymbolicExtent
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite


class Looplet(ABC):
    @property
    @abstractmethod
    def pass_request(self): ...


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
        thunks = {}

        def thunk_body(ctx, node: ntn.NotationNode):
            match node:
                case ntn.Access(
                    ntn.Looplet(Thunk() as thnk, type_), mode, (j, *idxs)
                ) if j == idx:
                    if id(thnk) in thunks:
                        return ntn.Access(thunks[id(thnk)], mode, (j, *idxs))
                    if (preamble := thnk.preamble) is not None:
                        match preamble(ctx, idx):
                            case asm.Block(bodies):
                                for statement in bodies:
                                    ctx.exec(statement)
                            case statement:
                                ctx.exec(statement)
                    if (epilogue := thnk.epilogue) is not None:
                        ctx.post(epilogue(ctx, idx))
                    thunks[id(thnk)] = ntn.Looplet(thnk.body(ctx, ext), type_)
                    return ntn.Access(thunks[id(thnk)], mode, (j, *idxs))

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
        # One switch is split at a time, so that each branch splits the others.
        for node in PostOrderDFS(body):
            match node:
                case ntn.Access(ntn.Looplet(Switch() as switch, _), _, (j, *_)) if (
                    j == idx
                ):
                    break

        def branch(choice):
            def switch_node(node):
                match node:
                    case ntn.Access(
                        ntn.Looplet(Switch() as tns, type_), mode, (j, *idxs)
                    ) if j == idx and tns is switch:
                        return ntn.Access(ntn.Looplet(choice, type_), mode, (j, *idxs))
                return None

            ctx_2 = ctx.scope()
            ctx_2(ext, PostWalk(switch_node)(body))
            return asm.Block(ctx_2.emit())

        ctx.exec(
            asm.IfElse(switch.cond, branch(switch.if_true), branch(switch.if_false))
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
        def stepper_body(ctx, node: ntn.NotationNode):
            match node:
                case ntn.Access(
                    ntn.Looplet(Stepper() as st, type_), mode, (j, *idxs)
                ) if j == idx:
                    assert st.chunk is not None
                    return ntn.Access(
                        ntn.Looplet(st.chunk, type_),
                        mode,
                        (j, *idxs),
                    )

        steppers: list[Stepper] = []

        for node in PostOrderDFS(body):
            match node:
                case ntn.Looplet(Stepper() as st, _):
                    steppers.append(st)

        steppers = list({id(stepper): stepper for stepper in steppers}.values())
        # A seek may move a stepper past the stop of its preamble.
        for stepper in steppers:
            ctx.exec(stepper.seek(ctx, ext))
            ctx.exec(stepper.preamble(ctx))

        full_body = Rewrite(PostWalk(lambda node: stepper_body(ctx, node)))(body)

        # Each phase ends after the nearest stop, or at the end of the loop, and
        # the steppers which stop there step past it.
        idx_start = ntn.Variable(ctx.freshen("i_start"), idx.result_type)
        phase_end = ntn.Variable(ctx.freshen("i_end"), idx.result_type)
        ctx.exec(asm.Assign(ctx.ctx(idx_start), ctx.ctx(ext.get_start())))
        ctx_2 = ctx.scope()
        stops = tuple(stepper.stop(ctx_2) for stepper in steppers)
        nearest = ntn.Call(
            ntn.L(ffuncs.add), (ntn.Call(ntn.L(ffuncs.min), stops), ext.get_unit())
        )
        ctx_2.exec(
            asm.Assign(
                ctx_2.ctx(phase_end),
                ctx_2.ctx(ntn.Call(ntn.L(ffuncs.min), (nearest, ext.get_end()))),
            )
        )
        ctx_2(SymbolicExtent(idx_start, phase_end), full_body)
        for stepper, stop in zip(steppers, stops, strict=True):
            cond = ntn.Call(
                ntn.L(ffuncs.eq),
                (ntn.Call(ntn.L(ffuncs.add), (stop, ext.get_unit())), phase_end),
            )
            ctx_2.exec(
                asm.If(
                    ctx_2.ctx(cond),
                    asm.Block((stepper.next(ctx_2), stepper.preamble(ctx_2))),
                )
            )
        ctx_2.exec(asm.Assign(ctx_2.ctx(idx_start), ctx_2.ctx(phase_end)))
        ctx.exec(
            asm.WhileLoop(
                asm.Call(
                    asm.L(ffuncs.lt), (ctx.ctx(idx_start), ctx.ctx(ext.get_end()))
                ),
                asm.Block(tuple(ctx_2.emit())),
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
                case ntn.Looplet(Sequence() as seq, _):
                    found_seqs.append(seq)

        found_seqs = list({id(seq): seq for seq in found_seqs}.values())

        def sequence_node(
            node: ntn.NotationNode, heads: set[Sequence], tails: set[Sequence]
        ):
            match node:
                case ntn.Access(
                    ntn.Looplet(Sequence() as tns, type_), mode, (j, *idxs)
                ) if j == idx:
                    if tns in heads:
                        new_tns = tns.head(ctx, idx)  # type: ignore[call-arg]
                    elif tns in tails:
                        new_tns = tns.tail(ctx, idx)
                    else:
                        raise Exception(f"Seq: {tns} not present.")
                    return ntn.Access(ntn.Looplet(new_tns, type_), mode, (j, *idxs))

        variations = self.get_sequence_variations(ctx, found_seqs, [], [], ext)
        blocks = []

        for v in variations:
            heads, tails, subext_start, subext_end = v
            sub_body = PostWalk(partial(sequence_node, heads=heads, tails=tails))(body)
            subext = SymbolicExtent(subext_start, subext_end)
            sub_ctx = ctx.scope()
            sub_ctx(subext, self.simplify_body(sub_body))
            blocks += sub_ctx.emit()

        ctx.exec(asm.Block(tuple(blocks)))


@dataclass
class Run(Looplet):
    body: ntn.NotationExpression

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
                case ntn.Access(ntn.Looplet(Run(body), _), mode, (j, *idxs)) if (
                    j == idx
                ):
                    # The protocol of j is spent with it.
                    match mode:
                        case ntn.Read((_, *protocols)):
                            mode = ntn.Read(tuple(protocols))
                    return ntn.Access(body, mode, tuple(idxs))
            return None

        body_2 = Rewrite(PostWalk(run_node))(body)
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
    body: Callable[[LoopletContext, ntn.Variable], Looplet]

    @property
    def pass_request(self):
        return LookupPass()


class LookupPass(LoopletPass):
    @property
    def priority(self):
        return 2

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        ctx_2 = ctx.scope()
        lookups = {}

        def lookup_node(node):
            match node:
                case ntn.Access(
                    ntn.Looplet(Lookup(lookup) as looplet, type_), mode, (j, *idxs)
                ) if j == idx:
                    if id(looplet) not in lookups:
                        lookups[id(looplet)] = ntn.Looplet(lookup(ctx_2, idx), type_)
                    return ntn.Access(lookups[id(looplet)], mode, (j, *idxs))
            return None

        body_2 = PostWalk(lookup_node)(body)
        ctx_2(SymbolicExtent.point(idx), body_2)
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
    """
    A stepper which skips ahead, as in Finch.jl. Each phase of its loop ends at
    the furthest stop of the jumpers, so that the jumper which stops there takes
    its chunk, and the others step to it.
    """

    preamble: Any = None
    stop: Callable = lambda ctx: None
    chunk: Looplet | None = None
    next: Callable[[Any], Any] = lambda ctx: None
    seek: Callable = lambda ctx, ext: (_ for _ in ()).throw(
        NotImplementedError("seek not implemented error")
    )

    @property
    def pass_request(self):
        return JumperPass()

    def stepper(self) -> Stepper:
        return Stepper(
            preamble=self.preamble,
            stop=self.stop,
            chunk=self.chunk,
            next=self.next,
            seek=self.seek,
        )


class JumperPass(LoopletPass):
    @property
    def priority(self):
        # A jumper takes over the steppers it meets, but a sequence splits it.
        return 3.5

    def __call__(
        self,
        ctx: LoopletContext,
        idx: ntn.Variable,
        ext: SymbolicExtent,
        body,
    ) -> None:
        jumpers: dict[int, Jumper] = {}
        for node in PostOrderDFS(body):
            match node:
                case ntn.Access(ntn.Looplet(Jumper() as jumper, _), _, (j, *_)) if (
                    j == idx
                ):
                    jumpers[id(jumper)] = jumper

        idx_start = ntn.Variable(ctx.freshen("i_start"), idx.result_type)
        phase_end = ntn.Variable(ctx.freshen("i_end"), idx.result_type)
        ctx.exec(asm.Assign(ctx.ctx(idx_start), ctx.ctx(ext.get_start())))
        ctx_2 = ctx.scope()
        rest = SymbolicExtent(idx_start, ext.get_end())
        for jumper in jumpers.values():
            ctx_2.exec(jumper.seek(ctx_2, rest))
            ctx_2.exec(jumper.preamble(ctx_2))
        stops = tuple(jumper.stop(ctx_2) for jumper in jumpers.values())
        furthest = ntn.Call(
            ntn.L(ffuncs.add), (ntn.Call(ntn.L(ffuncs.max), stops), ext.get_unit())
        )
        ctx_2.exec(
            asm.Assign(
                ctx_2.ctx(phase_end),
                ctx_2.ctx(ntn.Call(ntn.L(ffuncs.min), (furthest, ext.get_end()))),
            )
        )
        phase = SymbolicExtent(idx_start, phase_end)

        def lower_phase(leader: Jumper | None):
            def jumper_node(node):
                match node:
                    case ntn.Access(
                        ntn.Looplet(Jumper() as jumper, type_), mode, (j, *idxs)
                    ) if j == idx:
                        if jumper is leader:
                            looplet: Looplet = Thunk(
                                body=lambda ctx, ext: jumper.chunk,
                                epilogue=lambda ctx, idx: jumper.next(ctx),
                            )
                        else:
                            looplet = jumper.stepper()
                        return ntn.Access(ntn.Looplet(looplet, type_), mode, (j, *idxs))
                return None

            ctx_3 = ctx_2.scope()
            ctx_3(phase, Rewrite(PostWalk(jumper_node))(body))
            return asm.Block(tuple(ctx_3.emit()))

        # A jumper which doesn't stop at the end of the phase steps to it.
        phase_body = lower_phase(None)
        for leader, stop in reversed(tuple(zip(jumpers.values(), stops, strict=True))):
            cond = ntn.Call(
                ntn.L(ffuncs.eq),
                (ntn.Call(ntn.L(ffuncs.add), (stop, ext.get_unit())), phase_end),
            )
            phase_body = asm.Block(
                (asm.IfElse(ctx_2.ctx(cond), lower_phase(leader), phase_body),)
            )
        ctx_2.exec(phase_body)
        ctx_2.exec(asm.Assign(ctx_2.ctx(idx_start), ctx_2.ctx(phase_end)))
        ctx.exec(
            asm.WhileLoop(
                asm.Call(
                    asm.L(ffuncs.lt), (ctx.ctx(idx_start), ctx.ctx(ext.get_end()))
                ),
                asm.Block(tuple(ctx_2.emit())),
            )
        )
