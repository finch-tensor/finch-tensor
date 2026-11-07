from finch import finch_notation as ntn
from finch.algebra import FinchOperatorFType, is_identity
from finch.compile import looplets
from finch.symbolic import simplify_rules
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
        return None
