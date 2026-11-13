from finch import finch_notation as ntn
from finch.algebra import FinchOperatorFType, is_identity
from finch.compile import looplets
from finch.compile.lower import only_resets_scalars
from finch.symbolic import PostOrderDFS, simplify_rules
from finch.symbolic.rewriters import Chain, Fixpoint, PostWalk, Rewrite
from finch.symbolic.stage import UnvalidatedForm

from .stages import NotationTransform


class LoopletSimplify(UnvalidatedForm, NotationTransform):
    def lower(self, term: ntn.Module) -> ntn.Module:
        rules = [*simplify_rules(), self.simplify]
        return Rewrite(Fixpoint(PostWalk(Chain(rules))))(term)

    @staticmethod
    def simplify(term: ntn.NotationNode):
        match term:
            # A run of a literal reads as that literal everywhere, so the read
            # is replaced by the literal for the other rules to fold.
            case ntn.Unwrap(
                ntn.Access(
                    ntn.Looplet(looplets.Run(ntn.Full(ntn.Literal(val))), _),
                    ntn.Read(),
                    _,
                )
            ):
                return ntn.Literal(val)
            case ntn.If(ntn.Literal(val), body):
                return body if val else ntn.Block(())
            # Incrementing by the identity of the update leaves the tensor as
            # it was, so the increment, and any write it would cause, is dropped.
            case ntn.Increment(
                ntn.Access(
                    _,
                    ntn.Update(
                        ntn.NotationExpression(
                            result_type=FinchOperatorFType() as op_type
                        )
                    ),
                    _,
                ),
                ntn.Literal(val),
            ) if is_identity(op_type, val):
                return ntn.Block(())
            case ntn.Loop(_, _, ntn.Block(())):
                return ntn.Block(())
            # A scalar which a loop body declares is a temporary of each
            # iteration, so a loop which only resets temporaries does nothing.
            case ntn.Loop(_, _, body) if only_resets_scalars(body):
                return ntn.Block(())
            case ntn.Block(bodies) if any(isinstance(b, ntn.Block) for b in bodies):
                return ntn.Block(
                    tuple(
                        stmt
                        for body in bodies
                        for stmt in (
                            body.bodies if isinstance(body, ntn.Block) else (body,)
                        )
                    )
                )
            # A scalar declared with a literal reads as the literal until it is
            # written again.
            case ntn.Block(bodies):
                for n, body in enumerate(bodies):
                    match body:
                        case ntn.Declare(tns, ntn.Literal() as val, _, ()):
                            rest = known_reads(tns, val, bodies[n + 1 :])
                            if rest != bodies[n + 1 :]:
                                return ntn.Block((*bodies[: n + 1], *rest))
        return None


def known_reads(
    tns: ntn.NotationExpression,
    val: ntn.Literal,
    stmts: tuple[ntn.NotationStatement, ...],
) -> tuple[ntn.NotationStatement, ...]:
    """stmts, whose reads of tns, before the first which may write it, are val."""

    def writes(stmt):
        for node in PostOrderDFS(stmt):
            match node:
                case (
                    ntn.Access(var, ntn.Update(), _)
                    | ntn.Declare(var, _, _, _)
                    | ntn.Thaw(var, _)
                ) if var == tns:
                    return True
        return False

    def read(node):
        match node:
            case ntn.Unwrap(ntn.Access(var, ntn.Read(), ())) if var == tns:
                return val
        return None

    n = next((n for n, stmt in enumerate(stmts) if writes(stmt)), len(stmts))
    return (*(Rewrite(PostWalk(read))(stmt) for stmt in stmts[:n]), *stmts[n:])
