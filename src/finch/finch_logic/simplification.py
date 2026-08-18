from finch.algebra.tensor import TensorFType
from finch.symbolic import UnvalidatedForm
from finch.symbolic.simplification import annihilate, simplify, simplify_rules

from . import nodes as lgc
from .stages import LogicLoader
from .tensor_stats import StatsFactory, TensorStats


def unwrap_literal(node: lgc.LogicNode) -> lgc.LogicNode | None:
    """
    `Reorder(Literal(v), ())` => `Literal(v)`, and likewise for `Relabel`.

    A literal is zero-dimensional, so reordering or relabelling it by no fields
    is the identity. Without this the algebraic rules see nothing: `optimize`
    leaves every literal operand wrapped, so `LiteralTerm` never matches.
    """
    match node:
        case lgc.Reorder(lgc.Literal(_) as lit, ()) | lgc.Relabel(
            lgc.Literal(_) as lit, ()
        ):
            return lit
    return None


def simplify_logic(prgm: lgc.LogicStatement) -> lgc.LogicStatement:
    """
    Algebraically simplify a logic program.

    `annihilate` is deliberately left out. An operand of a MapJoin carries the
    fields that give the result its extent, so discarding one discards a
    dimension: annihilating `A * 0` here returns a tensor of shape `(1, 1)`.
    """
    rules = [rule for rule in simplify_rules() if rule is not annihilate]
    return simplify(prgm, [*rules, unwrap_literal])


class LogicSimplify(UnvalidatedForm, LogicLoader):
    """Simplify a logic program before it is lowered."""

    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        return self.ctx(simplify_logic(prgm), bindings, stats, stats_factory)
