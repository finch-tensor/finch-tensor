from abc import abstractmethod

from finch import finch_einsum as ein
from finch import finch_notation as ntn
from finch.algebra import ffuncs
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
from finch.symbolic import Form, PostOrderDFS, PreWalk, Rewrite, Stage


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
    already been optimized for this query. There are three valid kinds of input query:
    1) transpose queries
        Query(Table(_, output_order), Table(_, _))
    2) aggregate queries
        Query(Table(_, output_order), Aggregate(_, Literal(), arg, _))
    3) in-place queries
        QueryInto(Table(_, output_order), op, arg)
    (Here, arg has no aggregates. The fields of arg which are not in
    output_order are reduced with op.)
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
                case Query(Table(), Aggregate(_, Literal(), arg, _)):
                    return validate(arg, False)
                case Query(Table(), Aggregate(_, init, _, _)):
                    raise ValueError(
                        f"Aggregate queries must start from a literal, not {init}. "
                        "Copy the init into the output and update it with a "
                        "QueryInto instead."
                    )
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
        3) in-place queries
            QueryInto(Table(_, lhs_idxs), _, Reorder(arg, loop_order))
    For aggregate and in-place queries, the loop order includes every lhs
    field and visits those fields in order.
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
                case Query(
                    Table(_, lhs_idxs), Aggregate(_, _, Reorder(arg, idxs), _)
                ) | QueryInto(Table(_, lhs_idxs), _, Reorder(arg, idxs)):
                    if not cls._check_loop_order(lhs_idxs, idxs):
                        raise ValueError("Table index order does not match loop order.")
                    return validate(arg, idxs)
                case Query(Table(), Aggregate(_, _, arg, _)):
                    raise ValueError(
                        "All aggregates must wrap a Reorder node specifying\
                             the loop order."
                    )
                case QueryInto():
                    raise ValueError("In-place queries must have a loop order!")
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg, loop_order)
                case Table(_, idxs):
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


class CompilerForm(AliasedForm):
    """
    CompilerForm is the input of the notation lowerer. Every statement but the
    final Produces is a QueryInto, and initialization is explicit. There are
    two valid kinds of statement:
    1) initializations
        QueryInto(Table(lhs, _), overwrite, Literal(init))
    (Every element of lhs is set to init.)
    2) folds
        QueryInto(Table(lhs, lhs_idxs), op, Reorder(arg, loop_order))
    (Here, arg is made of Tables, Literals, and MapJoins. The loop order
    contains each field once and visits all fields of lhs_idxs in order.
    Tables in a MapJoin must also follow loop order.
    A single Table argument may have a different storage order, representing a
    transpose; notation lowering inserts equality-constrained loops to read it
    in storage order. Fields absent from lhs_idxs are reduced with op.)
    A transpose or fold which overwrites lhs also initializes it, starting from
    the init of a preceding initialization, or else the fill value of lhs.
    Every alias must have a TensorFType in the bindings, and a statement can't
    read the alias it writes.
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

        def validate(node, loop_order):
            match node:
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg, loop_order)
                case Table(tns, idxs):
                    if tns not in bindings:
                        raise ValueError(f"Alias {tns} has no TensorFType.")
                    if not LoopOrderedForm._check_loop_order(idxs, loop_order):
                        raise ValueError("Table index order does not match loop order.")
                case Literal():
                    return
                case _:
                    raise ValueError(f"Unsupported expression in a fold: {node}")

        match term:
            case Plan((*bodies, Produces())):
                pass
            case _:
                raise ValueError("The last body of a plan must be a Produces node.")
        for body in bodies:
            match body:
                case QueryInto(Table(lhs, lhs_idxs), Literal(op), rhs):
                    if lhs not in bindings:
                        raise ValueError(f"Alias {lhs} has no TensorFType.")
                    if lhs in PostOrderDFS(rhs):
                        raise ValueError(f"QueryInto can't both read and write {lhs}.")
                    match rhs:
                        case Literal():
                            if op != ffuncs.overwrite:
                                raise ValueError(
                                    f"Initializing {lhs} must overwrite it, not {op}."
                                )
                        case Reorder(arg, loop_order):
                            if len(set(loop_order)) != len(loop_order):
                                raise ValueError("Loop order must not repeat fields.")
                            if not set(arg.fields()).issubset(loop_order):
                                raise ValueError(
                                    "Loop order must include every RHS field."
                                )
                            if not LoopOrderedForm._check_loop_order(
                                lhs_idxs, loop_order
                            ):
                                raise ValueError(
                                    "Table index order does not match loop order."
                                )
                            match arg:
                                case Table():
                                    validate(arg, arg.idxs)
                                case _:
                                    validate(arg, loop_order)
                        case _:
                            raise ValueError(f"Unsupported QueryInto: {body}")
                case _:
                    raise ValueError(f"CompilerForm only allows QueryInto, not {body}")


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


class LogicNotationLowerer(CompilerForm, Stage):
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
