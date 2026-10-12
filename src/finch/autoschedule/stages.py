from abc import abstractmethod

from finch import finch_einsum as ein
from finch import finch_notation as ntn
from finch.algebra import ffuncs
from finch.algebra.tensor import TensorFType
from finch.finch_assembly.stages import AssemblyLibrary
from finch.finch_logic import (
    Aggregate,
    Alias,
    FillTable,
    Fuse,
    FusedAlias,
    HardAlias,
    Literal,
    LogicExpression,
    LogicStatement,
    LogicTree,
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
from finch.symbolic import Form, PostOrderDFS, Stage


class AliasedForm(Form):
    """
    AliasedForm requires that all aliases in the input are defined
    in the bindings or in previous queries and that all Tables
    are wrapping Aliases. Defining an alias selects its view, which is either
    the HardAlias or a FusedAlias with some number of fused dimensions, and
    invalidates every other view of that alias. Statements may only read or
    update the current view of an alias. Bindings define the HardAlias view.
    """

    @classmethod
    def validate_inputs(
        cls,
        term: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        views: dict[Alias, Alias] = {var.unfused: var for var in bindings}

        def validate(node):
            match node:
                case Query(Table(Alias() as lhs, _), rhs):
                    # The rhs is read before the lhs is defined.
                    validate(rhs)
                    views[lhs.unfused] = lhs
                case QueryInto(Table(Alias() as lhs, _), _, rhs):
                    validate(rhs)
                    if lhs.unfused not in views:
                        raise ValueError(
                            f"QueryInto updates alias {lhs.name}, which is not defined."
                        )
                    validate(lhs)
                case Alias() as var:
                    view = views.get(var.unfused)
                    if view is None:
                        raise ValueError(
                            f"Alias {var.name} is not defined in bindings."
                        )
                    if view != var:
                        raise ValueError(
                            f"{var} was invalidated by the definition of {view}."
                        )
                case Table(Alias() as tns, _):
                    validate(tns)
                case Table():
                    raise ValueError("Table nodes must wrap an Alias.")
                case LogicTree():
                    for child in node.children:
                        validate(child)

        validate(term)


def _fused_statements(node: LogicStatement) -> tuple[LogicStatement, ...]:
    """The statements of `node`, looking through the bodies of Fuse nodes."""
    match node:
        case Fuse(_, body):
            return _fused_statements(body)
        case Plan(bodies):
            return tuple(stmt for body in bodies for stmt in _fused_statements(body))
        case _:
            return (node,)


def _is_init(init: LogicExpression) -> bool:
    # The init of an aggregate is a 0-D expression without aggregates or reorders.
    return init.fields() == () and not any(
        isinstance(node, Aggregate | Reorder) for node in PostOrderDFS(init)
    )


class SingleAggregateForm(AliasedForm):
    """
    SingleAggregateForm assumes that the fusion strategy has
    already been optimized for this query. There are three valid kinds of input query:
    1) transpose queries
        Query(Table(_, output_order), Table(_, _))
    2) aggregate queries
        Query(Table(_, output_order), Aggregate(_, init, arg, _))
    (init is a 0-D expression without aggregates or reorders.)
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
                case Fuse(_, body):
                    for stmt in _fused_statements(body):
                        validate(stmt, True)
                case Query(Table(), Table()):
                    return None
                case Query(Table(), Aggregate(_, init, arg, _)) if _is_init(init):
                    return validate(arg, False)
                case Query(Table(), Aggregate(_, init, _, _)):
                    raise ValueError(
                        f"Aggregate queries must start from a 0-D expression "
                        f"without aggregates or reorders, not {init}. "
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
                case Literal() | FillTable() | Alias() | Table():
                    return None
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
            return None

        validate(term, True)


class LoopOrderedForm(SingleAggregateForm):
    """
    LoopOrderedForm assumes that the input query has had its loop order set.
    The loop order of a query is the fields of its enclosing Fuse nodes, from
    outermost to innermost, which include every field of the query. There are
    four valid forms for a query in LoopOrderedForm:
        1) transpose queries
            Query(Table(_, output_order), Table(_, _))
        2) aggregate queries
            Query(Table(_, output_order), Aggregate(_, _, arg, _))
        3) in-place queries
            QueryInto(Table(_, lhs_idxs), _, arg)
        4) in-place initializations
            QueryInto(Table(_, _), _, Literal(_))
    Initializations need no loop order. Otherwise, the loop order visits the
    lhs fields in order, and the Tables of arg follow the loop order, except
    that a transpose query, or an in-place query of a single Table, may read
    its Table in another order. Queries have no Reorder nodes.
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

        def validate_query(lhs_idxs, arg, loop_order):
            unfused = {*lhs_idxs, *arg.fields()} - set(loop_order)
            if unfused:
                names = ", ".join(sorted(idx.name for idx in unfused))
                raise ValueError(
                    f"The loop order of a query is given by its enclosing Fuse "
                    f"nodes, which must include every field, so {names} need "
                    "Fuse nodes."
                )
            if not cls._check_loop_order(lhs_idxs, loop_order):
                raise ValueError("Table index order does not match loop order.")

        def validate(node, loop_order):
            match node:
                case Plan(bodies):
                    for body in bodies:
                        validate(body, loop_order)
                case Fuse(idx, body):
                    if idx in loop_order:
                        raise ValueError(
                            f"Fuse({idx}) is nested in another Fuse of {idx}."
                        )
                    validate(body, (*loop_order, idx))
                case Produces() | QueryInto(Table(), _, Literal()):
                    return
                case Query(Table(_, lhs_idxs), Table() as arg) | QueryInto(
                    Table(_, lhs_idxs), _, Table() as arg
                ):
                    validate_query(lhs_idxs, arg, loop_order)
                case Query(Table(_, lhs_idxs), Aggregate(_, _, arg, _)) | QueryInto(
                    Table(_, lhs_idxs), _, arg
                ):
                    validate_query(lhs_idxs, arg, loop_order)
                    validate(arg, loop_order)
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg, loop_order)
                case Table(_, idxs):
                    if not cls._check_loop_order(idxs, loop_order):
                        raise ValueError("Table index order does not match loop order.")
                case Reorder():
                    raise ValueError(
                        "Queries with loop orders can't have Reorder nodes, "
                        "since Fuse nodes give the loop order and the left-hand "
                        "tables order the results."
                    )
                case Literal() | FillTable():
                    return
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
            return

        validate(term, ())


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
                case Fuse(_, body):
                    for stmt in _fused_statements(body):
                        validate(stmt)
                case Query(lhs, rhs) | QueryInto(lhs, _, rhs):
                    validate(lhs)
                    validate(rhs)
                case Aggregate(_, _, arg, _):
                    validate(arg)
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg)
                case Table(Alias() as tns, _):
                    if tns.unfused not in bindings:
                        raise ValueError(
                            f"Alias {tns.name} is not defined in bindings. All aliase\
                                 must have TensorFTypes specified at this stage."
                        )
                case Literal() | FillTable():
                    return
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
            return

        validate(term)


class CompilerForm(FormattedForm):
    """
    CompilerForm is the input of the notation lowerer. It is a FormattedForm
    where every statement but the final Produces is a definition or an update.
    There are two valid kinds of definition:
    1) copies
        Query(Table(lhs, lhs_idxs), Table(arg, idxs))
    (lhs is reset to its fill value, and arg is written into it. A different
    storage order represents a transpose; notation lowering inserts
    equality-constrained loops to read arg in storage order. Fields absent from
    lhs_idxs are reduced by overwriting.)
    2) reductions
        Query(Table(lhs, lhs_idxs), Aggregate(op, init, arg, _))
    (lhs is reset to init, a 0-D expression without aggregates or reorders,
    and arg is folded into it with op.)
    There are two valid kinds of update:
    1) initializations
        QueryInto(Table(lhs, _), overwrite, Literal(init))
    (Every element of lhs is set to init.)
    2) folds
        QueryInto(Table(lhs, lhs_idxs), op, arg)
    (arg is folded into lhs with op. A fold which overwrites lhs resets it to
    its fill value first.)
    A statement may also be a Fuse, whose body is a Plan of these statements
    and Fuse nodes, without a Produces, in FusedForm. It runs its body once for
    each value of its field. Within Fuse nodes, a FusedAlias may store only the
    dimensions after its fused ones, as a workspace for each fused iteration.

    In reductions and folds, arg is made of Tables, Literals, and MapJoins. The
    loop order of every statement but an initialization is the fields of its
    enclosing Fuse nodes, as in LoopOrderedForm, which include every field of
    the statement and visit lhs_idxs in order. Tables in a MapJoin must also
    follow loop order, but a single Table argument may have a different storage
    order, as in a copy. Fields absent from lhs_idxs are reduced with op. Every
    alias must have a TensorFType in the bindings, and a statement can't read
    the alias it writes. Definitions may change dimension sizes but must
    preserve the allocated tensor's rank.
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

        # FormattedForm has checked the grammar of each fold, its bindings,
        # and that it visits the lhs and its Tables in loop order.
        match term:
            case Plan((*bodies, Produces())):
                pass
            case _:
                raise ValueError("The last body of a plan must be a Produces node.")
        # Deleting a Fuse doesn't change the meaning of a program, so fused
        # aliases outside of Fuse nodes are views of their whole tensors.
        for body in bodies:
            if isinstance(body, Fuse):
                FusedForm.validate_inputs(body, bindings, stats, stats_factory)
        stmts = tuple(stmt for body in bodies for stmt in _fused_statements(body))
        for body in stmts:
            match body:
                case Query(Table(Alias() as lhs, idxs), rhs) | QueryInto(
                    Table(Alias() as lhs, idxs), _, rhs
                ):
                    pass
                case _:
                    raise ValueError(f"CompilerForm only allows queries, not {body}")
            # A fused alias may store only the dimensions after its fused ones.
            ranks = {len(idxs)}
            if isinstance(lhs, FusedAlias):
                ranks.add(len(idxs) - lhs.n)
            if bindings[lhs.unfused].ndim not in ranks:
                raise ValueError(f"Cannot change the rank of allocated tensor {lhs}.")
            if any(
                node.unfused == lhs.unfused
                for node in PostOrderDFS(rhs)
                if isinstance(node, Alias)
            ):
                raise ValueError(
                    f"{type(body).__name__} can't both read and write {lhs}."
                )
            match body:
                case Query(_, Aggregate(_, init, _, _)) if not _is_init(init):
                    raise ValueError(
                        f"The init of an aggregate into {lhs} must be a 0-D "
                        "expression without aggregates or reorders."
                    )
                case QueryInto(_, Literal(op), Literal()) if op != ffuncs.overwrite:
                    raise ValueError(f"Initializing {lhs} must overwrite it, not {op}.")


class FusedForm(Form):
    """
    Each Fuse field occurs in a HardAlias table in its body. The leading n
    indices of every FusedAlias table match the first n enclosing Fuse fields,
    from outermost to innermost.

    Within each Fuse body, a tensor accessed by that Fuse's field in a stored
    dimension cannot occur on both a query lhs and a query rhs, even in separate
    queries. Different alias views refer to the same tensor for this check.
    These constraints apply independently of the query's lowering stage.
    """

    @classmethod
    def validate_inputs(
        cls,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        def validate(node, fused_idxs):
            match node:
                case Fuse(idx, body):
                    hard_idxs = set()
                    stored = set()
                    reads = set()
                    writes = set()
                    for child in PostOrderDFS(body):
                        match child:
                            case Table(HardAlias() as tns, idxs):
                                hard_idxs.update(idxs)
                                if idx in idxs:
                                    stored.add(tns)
                            case Table(FusedAlias(tns, n), idxs):
                                if idx in idxs[n:]:
                                    stored.add(tns)
                            case Query(Table(Alias() as lhs, _), rhs) | QueryInto(
                                Table(Alias() as lhs, _), _, rhs
                            ):
                                writes.add(lhs.unfused)
                                for arg in PostOrderDFS(rhs):
                                    match arg:
                                        case Alias() as tns:
                                            reads.add(tns.unfused)
                    if idx not in hard_idxs:
                        raise ValueError(
                            f"Fuse field {idx} must occur in a HardAlias table "
                            "in its body."
                        )
                    conflicts = stored & reads & writes
                    if conflicts:
                        names = ", ".join(sorted(tns.name for tns in conflicts))
                        raise ValueError(
                            f"Fuse field {idx} accesses a non-fused dimension of "
                            f"{names}, which cannot occur on both lhs and rhs "
                            "in its body."
                        )
                    validate(body, (*fused_idxs, idx))
                case Table(FusedAlias(tns, n), idxs):
                    if n > len(fused_idxs) or idxs[:n] != fused_idxs[:n]:
                        raise ValueError(
                            f"Fused indices of {tns} must match the enclosing "
                            "Fuse fields in order."
                        )
                case LogicTree():
                    for child in node.children:
                        validate(child, fused_idxs)

        validate(term, ())


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
