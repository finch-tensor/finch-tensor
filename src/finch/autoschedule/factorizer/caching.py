from collections.abc import Callable
from typing import cast

from finch.algebra import (
    FinchOperator,
    ffuncs,
    is_commutative,
    is_distributive,
    is_identity,
)
from finch.algebra.tensor import Tensor
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import LoopDepthCostModel
from finch.autoschedule.executor import extract_tensors
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.util import push_relabels
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    FillTable,
    HardAlias,
    Literal,
    LogicEvaluator,
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    StatsFactory,
    Table,
)
from finch.finch_logic.nodes import TableValue
from finch.finch_logic.simplification import unwrap_literal
from finch.symbolic import (
    Chain,
    Fixpoint,
    PostOrderDFS,
    PostWalk,
    PreOrderDFS,
    Rewrite,
    UnvalidatedForm,
    simplify_rules,
)
from finch.symbolic.simplification import annihilate

from .galley_factorizer.galley_optimize import GalleyLogicFactorizer

# Of two operators which distribute over each other, the first is distributed
# over the second, so that the second is outermost, as and is in CNF.
_CNF_ORDER = [
    (ffuncs.or_, ffuncs.and_),
    (ffuncs.logical_or, ffuncs.logical_and),
]


def _distributes(f: FinchOperator, g: FinchOperator) -> bool:
    if is_distributive(g.ftype, f.ftype):
        return (f, g) in _CNF_ORDER
    return is_distributive(f.ftype, g.ftype)


def _shape(node: LogicNode) -> LogicNode:
    return Rewrite(PostWalk(lambda x: Field("_") if isinstance(x, Field) else None))(
        node
    )


def _pointwise(node: LogicNode) -> LogicNode | None:
    match node:
        case MapJoin(Literal(f), args):
            # Products of sums become sums of products, and or over and is CNF.
            for i, arg in enumerate(args):
                match arg:
                    case MapJoin(Literal(g), brgs) if _distributes(f, g):
                        return MapJoin(
                            Literal(g),
                            tuple(
                                MapJoin(Literal(f), (*args[:i], brg, *args[i + 1 :]))
                                for brg in brgs
                            ),
                        )
            # Literals stay in front, as `canonicalize_associative` keeps them,
            # and arguments are sorted by their shape before their fields, so
            # that the reduced fields can be named afterwards.
            if is_commutative(f.ftype):
                key = sorted(
                    args,
                    key=lambda a: (
                        not isinstance(a, Literal),
                        hash(_shape(a)),
                        hash(a),
                    ),
                )
                if tuple(key) != args:
                    return MapJoin(Literal(f), tuple(key))
        # An aggregate over no fields joins its init to its arg once.
        case Aggregate(Literal(ffuncs.overwrite), _, arg, ()):
            return arg
        case Aggregate(Literal(op), Literal(init), arg, ()) if is_identity(
            op.ftype, init
        ):
            return arg
        case Reorder(arg, idxs) if idxs == arg.fields():
            return arg
    return None


def query_key(rhs: LogicExpression, idxs: tuple[Field, ...]) -> LogicExpression:
    """
    A canonical form of the query which stores `rhs` in the order of `idxs`,
    where `rhs` has at most one aggregate. The fields of the result are named
    by position, the fields it reduces in order of appearance, and its
    pointwise expression is normalized.
    """
    names = {idx: Field(f"#{n}") for n, idx in enumerate(idxs)}
    expr = Rewrite(PostWalk(names.get))(Reorder(rhs, idxs))
    # An annihilated operand still gives the result its fields, as in
    # `simplify_logic`.
    rules = [rule for rule in simplify_rules() if rule is not annihilate]
    expr = Rewrite(Fixpoint(PostWalk(Chain([*rules, unwrap_literal, _pointwise]))))(
        expr
    )
    agg = next((x for x in PreOrderDFS(expr) if isinstance(x, Aggregate)), None)
    if agg is not None:
        seen = [x for x in PreOrderDFS(agg.arg) if x in agg.idxs]
        order = dict.fromkeys([*seen, *sorted(agg.idxs, key=lambda idx: idx.name)])
        reduced = {idx: Field(f"_{n}") for n, idx in enumerate(order)}

        def name(node: LogicNode) -> LogicNode | None:
            match node:
                case Field() if node in reduced:
                    return reduced[node]
                case Aggregate(op, init, arg, red):
                    red = tuple(sorted(red, key=lambda idx: idx.name))
                    return Aggregate(op, init, arg, red)
            return None

        expr = Rewrite(PostWalk(name))(expr)
    return cast(LogicExpression, expr)


class CachingFactorizer(UnvalidatedForm, LogicEvaluator):
    """
    Factors a program with a child factorizer, which defaults to Galley keeping
    loop nests shallow, then evaluates each query it produces with a child
    evaluator, caching the result of each query by the canonical form of what
    it computes. A query which reads cached results can hit the cache too.

    Results are cached by the identity of the tensors they read, so a tensor
    must not change while its results are cached, and they stay cached as long
    as the factorizer. The results of a program may be shared with the cache.
    """

    def __init__(
        self,
        evaluator: LogicEvaluator,
        factorizer: Callable[[LogicLoader], LogicLoader] | None = None,
        stats_factory: StatsFactory | None = None,
    ):
        self.evaluator = evaluator
        self.capture = LogicCapture()
        if factorizer is None:
            self.factorizer: LogicLoader = GalleyLogicFactorizer(
                self.capture, optimizer="greedy", cost_model=LoopDepthCostModel()
            )
        else:
            self.factorizer = factorizer(self.capture)
        self.stats_factory = (
            DenseStatsFactory() if stats_factory is None else stats_factory
        )
        self.cache: dict[LogicNode, Tensor] = {}

    def lower(self, prgm: LogicNode, bindings: dict[Alias, Tensor] | None = None):
        if isinstance(prgm, LogicExpression):
            out = HardAlias("result")
            stmt: LogicStatement = Plan(
                (Query(Table(out, prgm.fields()), prgm), Produces((out,)))
            )
        elif isinstance(prgm, Plan):
            stmt = prgm
        elif isinstance(prgm, LogicStatement):
            stmt = Plan((prgm,))
        else:
            raise ValueError(f"Invalid prgm type: {type(prgm)}")
        # Galley reads relabels only on tables.
        stmt, results = extract_tensors(push_relabels(stmt), bindings or {})
        stats = {
            var: self.stats_factory(
                tns, tuple(Field(f"d{i}") for i in range(len(tns.shape)))
            )
            for var, tns in results.items()
        }
        self.factorizer(
            stmt,
            {var: tns.ftype for var, tns in results.items()},
            stats,
            self.stats_factory,
        )

        def read(node: LogicNode) -> LogicNode | None:
            match node:
                case Table(Alias() as alias, idxs):
                    return Table(Literal(results[alias]), idxs)
                case FillTable(Alias() as alias):
                    return Literal(results[alias].fill_value)
            return None

        factored = self.capture.last_prgm
        assert isinstance(factored, Plan)
        for body in factored.bodies:
            match body:
                case Query(Table(Alias() as alias, idxs), rhs):
                    key = query_key(
                        cast(LogicExpression, Rewrite(PostWalk(read))(rhs)), idxs
                    )
                    if key not in self.cache:
                        reads = {
                            var: results[var]
                            for var in PostOrderDFS(rhs)
                            if isinstance(var, Alias)
                        }
                        (self.cache[key],) = self.evaluator(
                            Plan((body, Produces((alias,)))), reads
                        )
                    results[alias] = self.cache[key]
                case Produces(args):
                    if isinstance(prgm, LogicExpression):
                        return TableValue(results[cast(Alias, args[0])], prgm.fields())
                    return tuple(results[cast(Alias, arg)] for arg in args)
                case _:
                    raise ValueError(f"Unsupported statement: {body}")
        raise ValueError("The factorized program produces nothing")
