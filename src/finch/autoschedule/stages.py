from abc import abstractmethod

from finch import finch_einsum as ein
from finch import finch_notation as ntn
from finch.algebra.algebra import is_identity
from finch.algebra.tensor import TensorFType
from finch.finch_assembly.stages import AssemblyLibrary
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    Table,
)
from finch.finch_logic.stages import LogicLoader
from finch.finch_logic.tensor_stats import StatsFactory, TensorStats
from finch.symbolic import Form, PreWalk, Rewrite, Stage
from finch.tensor.patterns import PatternTensorFType


class AliasedForm(Form):
    """
    AliasedForm requires that all aliases in the input are defined
    in the bindings or in previous queries and that all Tables
    are wrapping Aliases.
    """

    @classmethod
    def validate_inputs(
        cls,
        term: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        defined_aliases = set(bindings.keys())

        def validate(node):
            match node:
                case Query(Table(Alias() as lhs, _), _):
                    defined_aliases.add(lhs)
                case QueryInto(Table(Alias() as lhs, _), _, _):
                    if lhs not in defined_aliases:
                        raise ValueError(
                            f"QueryInto updates alias {lhs.name}, which is not defined."
                        )
                case Query(lhs, _) | QueryInto(lhs, _, _):
                    raise ValueError(f"Query must write to a Table of an Alias: {lhs}")
                case Alias(name):
                    if node not in defined_aliases:
                        raise ValueError(f"Alias {name} is not defined in bindings.")
                case Table(tns, _):
                    if not isinstance(tns, Alias):
                        raise ValueError("Table nodes must wrap an Alias.")
            return node

        Rewrite(PreWalk(validate))(term)


class SingleAggregateForm(AliasedForm):
    """
    SingleAggregateForm assumes that the fusion strategy has
    already been optimized for this query. There are four valid kinds of input query:
    1) transpose queries
        Query(Table(_, output_order), Table(_, _))
    2) aggregate queries
        Query(Table(_, output_order), Aggregate(_, _, arg, _))
    3) in-place aggregate queries
        QueryInto(Table(_, output_order), op, Aggregate(op, init, arg, _))
    (Here, the aggregate reduces with the update operator op, starting from an
    identity init, so each value can be folded directly into the output.)
    4) in-place pointwise queries
        QueryInto(Table(_, output_order), op, arg)
    (Here, arg has no aggregates.)
    """

    @classmethod
    def validate_inputs(
        cls,
        term: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        super().validate_inputs(term, bindings, stats, stats_factory)

        def validate(node, agg_allowed):
            match node:
                case Plan(bodies):
                    if not isinstance(bodies[-1], Produces):
                        raise ValueError(
                            "The last body of a plan must be a Produces node."
                        )
                    for body in bodies[:-1]:
                        validate(body, True)
                case Query(Table(), Table()):
                    return None
                case Query(Table(), Aggregate(_, _, arg, _)):
                    return validate(arg, False)
                case QueryInto(
                    Table(),
                    Literal(op1),
                    Aggregate(Literal(op2), Literal(init), arg, _),
                ):
                    if op2 != op1 or not is_identity(op1.ftype, init):
                        raise ValueError(
                            "The aggregate of an in-place query must reduce with "
                            "the update operator, starting from its identity."
                        )
                    return validate(arg, False)
                case QueryInto(Table(), _, arg):
                    return validate(arg, False)
                case Query(_, rhs):
                    raise ValueError(f"Unsupported query right-hand side: {rhs}")
                case Aggregate(_, _, arg, _):
                    if not agg_allowed:
                        raise ValueError("Nested aggregates are not supported.")
                    return validate(arg, False)
                case Reorder(arg, _):
                    return validate(arg, agg_allowed)
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg, agg_allowed)
                    return None
                case Literal() | Alias() | Table():
                    return None
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
            return None

        validate(term, True)


class LoopOrderedForm(SingleAggregateForm):
    """
    LoopOrderedForm assumes that the input query has had its loop order set.
    There are three valid forms for a query in LoopOrderedForm:
        1) transpose queries
            Query(Table(_, output_order), Table(_, _))
        2) aggregate queries
            Query(Table(_, output_order), Aggregate(_, _, Reorder(arg, loop_order), _))
        3) in-place aggregate queries
            QueryInto(Table(_, lhs_idxs), _,
                Aggregate(_, _, Reorder(arg, loop_order), _))
        4) in-place pointwise queries
            QueryInto(Table(_, lhs_idxs), _, Reorder(arg, loop_order))
    (Here, the loop order visits the fields of lhs_idxs in order.)
    """

    @staticmethod
    def _check_loop_order(idxs, loop_order):
        rel_loop_order = [idx for idx in loop_order if idx in idxs]
        return tuple(rel_loop_order) == tuple(idxs)

    @classmethod
    def validate_inputs(
        cls,
        term: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        super().validate_inputs(term, bindings, stats, stats_factory)

        def validate(node, loop_order):
            match node:
                case Plan(bodies):
                    for body in bodies[:-1]:
                        validate(body, loop_order)
                case Query(Table(), Table()):
                    return None
                case Query(Table(), Aggregate(_, _, Reorder(arg, idxs), _)):
                    return validate(arg, idxs)
                case Query(Table(), Aggregate(_, _, arg, _)):
                    raise ValueError(
                        "All aggregates must wrap a Reorder node specifying\
                             the loop order."
                    )
                case QueryInto(
                    Table(_, lhs_idxs), _, Aggregate(_, _, Reorder(arg, idxs_1), _)
                ):
                    if not cls._check_loop_order(lhs_idxs, idxs_1):
                        raise ValueError("Table index order does not match loop order.")
                    return validate(arg, idxs_1)
                case QueryInto(Table(), _, Aggregate()):
                    raise ValueError(
                        "In-place queries must have an interior loop order!"
                    )
                case QueryInto(Table(_, lhs_idxs), _, Reorder(arg, idxs_1)):
                    # A pointwise update has no aggregate to hold its loop order,
                    # so its right-hand side is wrapped in a Reorder instead.
                    if not cls._check_loop_order(lhs_idxs, idxs_1):
                        raise ValueError("Table index order does not match loop order.")
                    return validate(arg, idxs_1)
                case QueryInto():
                    raise ValueError("In-place queries must have a loop order!")
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg, loop_order)
                case Table(tns, idxs):
                    # Implicit patterns have no row-major storage to preserve.
                    match bindings.get(tns):
                        case PatternTensorFType():
                            return None
                    if not cls._check_loop_order(idxs, loop_order):
                        raise ValueError("Table index order does not match loop order.")
                case Reorder(arg, _):
                    raise ValueError("Reorder nodes should only appear in loop orders!")
                case Literal():
                    return None
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
            return None

        validate(term, None)


class FormattedForm(LoopOrderedForm):
    """
    FormattedForm requires that the input query has had its tensor formats
    set. Every alias must have a TensorFType in the bindings. The valid forms
    of a query are those of LoopOrderedForm.
    """

    @classmethod
    def validate_inputs(
        cls,
        term: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        super().validate_inputs(term, bindings, stats, stats_factory)

        def validate(node):
            match node:
                case Plan(bodies):
                    for body in bodies[:-1]:
                        validate(body)
                case Query(lhs, rhs) | QueryInto(lhs, _, rhs):
                    validate(lhs)
                    validate(rhs)
                case Aggregate(_, _, arg, _) | Reorder(arg, _):
                    validate(arg)
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg)
                case Table(tns, _):
                    if tns not in bindings:
                        raise ValueError(
                            f"Alias {tns.name} is not defined in bindings. All aliase\
                                 must have TensorFTypes specified at this stage."
                        )
                case Literal():
                    return
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
            return

        validate(term)


class LogicFactorizer(AliasedForm, LogicLoader):
    @abstractmethod
    def lower(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[Alias, TensorFType],
        dict[Alias, tuple[Field | None, ...]],
        LogicStatement,
    ]:
        """
        Optimize the aggregate structure of the given logic statement and
        make decisions about materialization.
        """


class LogicLoopOrderer(SingleAggregateForm, LogicLoader):
    @abstractmethod
    def lower(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[Alias, TensorFType],
        dict[Alias, tuple[Field | None, ...]],
        LogicStatement,
    ]:
        """
        Optimize the loop order of each query and add transposes where
        necessary.
        """


class LogicFormatter(LoopOrderedForm, LogicLoader):
    @abstractmethod
    def lower(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[Alias, TensorFType],
        dict[Alias, tuple[Field | None, ...]],
        LogicStatement,
    ]:
        """
        Optimize the tensor formats and output orders for each query.
        """


class LogicNotationLowerer(FormattedForm, Stage):
    @abstractmethod
    def lower(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> ntn.Module:
        """
        Generate Finch Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """


class LogicEinsumLowerer(FormattedForm, Stage):
    @abstractmethod
    def lower(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Generate Einsum Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """
