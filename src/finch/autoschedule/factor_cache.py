import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from finch.algebra import (
    FinchOperator,
    ffuncs,
    is_commutative,
    is_distributive,
    is_identity,
)
from finch.algebra.tensor import Tensor
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import LOOP_DEPTH_COST_MODEL
from finch.autoschedule.executor import extract_tensors
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.util import push_relabels
from finch.finch_logic import (
    Aggregate,
    Alias,
    DimensionTable,
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

from .factorizer.galley_factorizer.galley_optimize import GalleyLogicFactorizer


@dataclass(frozen=True)
class TensorRef:
    """A tensor in a query key, by identity, without keeping it alive."""

    id: int


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


class FactorCacheEvaluator(UnvalidatedForm, LogicEvaluator):
    """
    Factors a program with a child factorizer, which defaults to Galley keeping
    loop nests shallow, then evaluates each query it produces with a child
    evaluator, caching the result of each query by the canonical form of what
    it computes. A query which reads cached results can hit the cache too.

    Results are cached by the identity of the tensors they read, so a tensor
    must not change while its results are cached. A result is dropped when a
    tensor it read dies, so cached results live as long as their inputs. The
    results of a program may be shared with the cache.
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
                self.capture, optimizer="greedy", cost_model=LOOP_DEPTH_COST_MODEL
            )
        else:
            self.factorizer = factorizer(self.capture)
        self.stats_factory = (
            DenseStatsFactory() if stats_factory is None else stats_factory
        )
        self.cache: dict[LogicNode, Tensor] = {}
        # The keys which read each tensor, by id, to drop when it dies.
        self.readers: dict[int, list[LogicNode]] = {}

    def _ref(self, tns: Tensor) -> TensorRef:
        if id(tns) not in self.readers:
            self.readers[id(tns)] = []
            weakref.finalize(tns, self._forget, id(tns))
        return TensorRef(id(tns))

    def _forget(self, ref: int) -> None:
        for key in self.readers.pop(ref, ()):
            self.cache.pop(key, None)

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
                    return Table(Literal(self._ref(results[alias])), idxs)
                case FillTable(Alias() as alias):
                    return Literal(results[alias].fill_value)
                case DimensionTable(Alias() as alias, Literal(r)):
                    return Literal(np.intp(results[alias].shape[r]))
            return None

        factored = self.capture.last_prgm
        assert isinstance(factored, Plan)
        for body in factored.bodies:
            match body:
                # Galley starts a pointwise result from its fill, as the
                # evaluator does itself.
                case Query(lhs, Aggregate(Literal(ffuncs.overwrite), _, arg, ())):
                    body = Query(lhs, arg)
            match body:
                case Query(
                    Table(Alias() as alias, idxs), Table(Alias() as src, src_idxs)
                ) if idxs == src_idxs:
                    results[alias] = results[src]
                case Query(Table(Alias() as alias, idxs), rhs):
                    key = query_key(
                        cast(LogicExpression, Rewrite(PostWalk(read))(rhs)), idxs
                    )
                    if key in self.cache:
                        results[alias] = self.cache[key]
                        continue
                    reads = {
                        var: results[var]
                        for var in PostOrderDFS(rhs)
                        if isinstance(var, Alias)
                    }
                    (results[alias],) = self.evaluator(
                        Plan((body, Produces((alias,)))), reads
                    )
                    # A result which is one of its inputs would keep itself alive.
                    if all(results[alias] is not tns for tns in reads.values()):
                        self.cache[key] = results[alias]
                        for tns in reads.values():
                            self.readers[self._ref(tns).id].append(key)
                case Produces(args):
                    if isinstance(prgm, LogicExpression):
                        return TableValue(results[cast(Alias, args[0])], prgm.fields())
                    return tuple(results[cast(Alias, arg)] for arg in args)
                case _:
                    raise ValueError(f"Unsupported statement: {body}")
        raise ValueError("The factorized program produces nothing")
