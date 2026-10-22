from abc import abstractmethod
from itertools import count

from finch import finch_einsum as ein
from finch import finch_notation as ntn
from finch.algebra import AbstractFillFType, ffuncs, is_static
from finch.algebra.tensor import TensorFType
from finch.finch_assembly.stages import AssemblyLibrary
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
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
    fill_expression,
    is_fill_expression,
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

    Every Fuse node is in FusedForm. A FusedAlias defined within Fuse nodes
    stores a slice for each value of its fused fields, so it can only be read
    within the Fuse nodes of those fields which define it. The reset of a
    definition or overwrite runs before the outermost Fuse over a field which
    its lhs stores or which is reduced into it, so it can't read a dynamic fill
    of a tensor which that Fuse defines.
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
        fills: dict[Alias, AbstractFillFType] = {
            var.unfused: val.fill_type for var, val in bindings.items()
        }
        # The Fuse nodes enclosing the latest definition of each tensor.
        scopes: dict[Alias, tuple] = {}
        fuse_ids = count()

        def check_reset(stmt, fuses):
            if not fuses:
                return
            match stmt:
                case Query(Table(Alias() as lhs, idxs), Aggregate(_, init, arg, _)):
                    pass
                case Query(Table(Alias() as lhs, idxs), arg):
                    init = fill_expression(arg)
                case QueryInto(
                    Table(Alias() as lhs, idxs), Literal(ffuncs.overwrite), arg
                ) if not is_static(fills[lhs.unfused]):
                    init = fill_expression(arg)
                case _:
                    return
            if is_fill_expression(init) and is_static(init.fill_type(fills)):
                return
            n = lhs.nfused
            spanned = {*idxs[n:], *arg.fields()} - set(idxs[:n])
            for level, (idx, _) in enumerate(fuses):
                if idx in spanned:
                    scope = fuses[: level + 1]
                    for node in PostOrderDFS(init):
                        match node:
                            case Alias() as var if (
                                scopes.get(var.unfused, ())[: level + 1] == scope
                            ):
                                raise ValueError(
                                    f"The reset of {lhs} runs before the Fuse of "
                                    f"{idx}, so it can't read the fill of "
                                    f"{var.name}, which that Fuse defines."
                                )
                    return

        def validate(node, fuses):
            match node:
                case Fuse(idx, body):
                    validate(body, (*fuses, (idx, next(fuse_ids))))
                    if not fuses:
                        FusedForm.validate_inputs(node, bindings, stats, stats_factory)
                case Query(Table(Alias() as lhs, _), rhs):
                    # The rhs is read before the lhs is defined.
                    validate(rhs, fuses)
                    check_reset(node, fuses)
                    views[lhs.unfused] = lhs
                    fills.setdefault(lhs.unfused, rhs.fill_type(fills))
                    scopes[lhs.unfused] = fuses
                case QueryInto(Table(Alias() as lhs, _), _, rhs):
                    validate(rhs, fuses)
                    if lhs.unfused not in views:
                        raise ValueError(
                            f"QueryInto updates alias {lhs.name}, which is not defined."
                        )
                    validate(lhs, fuses)
                    check_reset(node, fuses)
                case Query(lhs, _) | QueryInto(lhs, _, _):
                    raise ValueError(f"Query must write to a Table of an Alias: {lhs}")
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
                    match var:
                        case FusedAlias(_, n) if (
                            fuses[:n] != scopes.get(var.unfused, ())[:n]
                        ):
                            raise ValueError(
                                f"{var} stores a slice for each value of its fused "
                                "fields, so it can't be read outside the Fuse "
                                "nodes which define it."
                            )
                case Table(Alias() as tns, _):
                    validate(tns, fuses)
                case Table():
                    raise ValueError("Table nodes must wrap an Alias.")
                case LogicTree():
                    for child in node.children:
                        validate(child, fuses)

        validate(term, ())


def _fused_statements(node: LogicStatement) -> tuple[LogicStatement, ...]:
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
    of a query are those of LoopOrderedForm. The rank of each TensorFType is
    the number of fields which the views of its alias store.
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
                    # A fused alias may store only the fields after its fused ones.
                    match lhs:
                        case Table(FusedAlias(var, n), idxs):
                            ranks = {len(idxs), len(idxs) - n}
                        case Table(Alias() as var, idxs):
                            ranks = {len(idxs)}
                    if bindings[var.unfused].ndim not in ranks:
                        raise ValueError(
                            f"Cannot change the rank of allocated tensor {lhs.tns}."
                        )
                case Aggregate(_, _, arg, _) | Reorder(arg, _):
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
    loop order contains each field once and visits all fields of lhs_idxs in
    order. Tables in a MapJoin must also follow loop order, but a single Table
    argument may have a different storage order, as in a copy. Fields absent
    from lhs_idxs are reduced with op. Every alias must have a TensorFType in
    the bindings, and a statement can't read the alias it writes. The loop
    order is the fields of the enclosing Fuse nodes.
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
        stmts = tuple(stmt for body in bodies for stmt in _fused_statements(body))
        for body in stmts:
            match body:
                case Query(Table(Alias() as lhs, _), rhs) | QueryInto(
                    Table(Alias() as lhs, _), _, rhs
                ):
                    pass
                case _:
                    raise ValueError(f"CompilerForm only allows queries, not {body}")
            for node in PostOrderDFS(rhs):
                match node:
                    case Alias() as var if var.unfused == lhs.unfused:
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
    A Fuse runs its body once for each value of its field, and deleting it
    doesn't change the meaning of a program. So a Fuse field isn't fused again
    within its body, every statement of its body uses it, and it occurs in a
    HardAlias table of its body, which gives its extent. The leading n indices
    of every FusedAlias table match the first n enclosing Fuse fields, from
    outermost to innermost.

    Every iteration of a Fuse shares the tensors which its body stores along
    its field, or reduces it into, so they cannot occur on both lhs and rhs in
    its body, and only one statement of its body may reduce into a tensor. A
    definition which broadcasts the Fuse field, which its rhs lacks, has extent
    1 in it, so it is the only statement of its body. A statement which reads a
    Table out of loop order revisits its fields, so it shares no Fuse node
    with other statements. Different alias views refer to the same tensor for
    these checks, which apply independently of the query's lowering stage.
    """

    @classmethod
    def validate_inputs(
        cls,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> None:
        def validate_body(idx: Field, stmts: tuple[LogicStatement, ...]):
            extent = False
            shared: set[Alias] = set()
            reads: set[Alias] = set()
            writes: list[Alias] = []
            reductions: list[Alias] = []
            broadcasts: list[Alias] = []
            for stmt in stmts:
                match stmt:
                    case Query(Table(Alias() as lhs, idxs), rhs) | QueryInto(
                        Table(Alias() as lhs, idxs), _, rhs
                    ):
                        pass
                    case _:
                        continue
                writes.append(lhs.unfused)
                match lhs:
                    case FusedAlias(_, n):
                        stores = idx in idxs[n:]
                    case _:
                        stores = idx in idxs
                        extent = extent or stores
                if stores:
                    shared.add(lhs.unfused)
                reads_idx = False
                for node in PostOrderDFS(rhs):
                    match node:
                        case Alias() as var:
                            reads.add(var.unfused)
                        case Table(HardAlias() as tns, arg_idxs) if idx in arg_idxs:
                            reads_idx = extent = True
                            shared.add(tns)
                        case Table(FusedAlias(tns, n), arg_idxs) if idx in arg_idxs:
                            reads_idx = True
                            if idx in arg_idxs[n:]:
                                shared.add(tns)
                if idx not in idxs:
                    if not reads_idx:
                        raise ValueError(
                            f"{stmt} doesn't use the field {idx} of its enclosing "
                            "Fuse, so it would run once for each of its values."
                        )
                    shared.add(lhs.unfused)
                    reductions.append(lhs.unfused)
                match stmt:
                    case Query() if stores and not reads_idx:
                        broadcasts.append(lhs)
            if not extent:
                raise ValueError(
                    f"Fuse field {idx} must occur in a HardAlias table in its body."
                )
            conflicts = shared & reads & set(writes)
            if conflicts:
                names = ", ".join(sorted(tns.name for tns in conflicts))
                raise ValueError(
                    f"The iterations of Fuse field {idx} share {names}, which "
                    "cannot occur on both lhs and rhs in its body."
                )
            for tns in reductions:
                if writes.count(tns) > 1:
                    raise ValueError(
                        f"Fuse field {idx} is reduced into {tns.name}, which no "
                        "other statement of its body may write."
                    )
            if broadcasts and len(stmts) > 1:
                raise ValueError(
                    f"{broadcasts[0]} broadcasts Fuse field {idx}, so it has "
                    "extent 1 and needs a Fuse of its own."
                )

        def validate(node, fused_idxs, alone):
            """Validate node within Fuse nodes of fused_idxs, whose bodies have
            only one statement when alone is true."""
            match node:
                case Fuse(idx, body):
                    if idx in fused_idxs:
                        raise ValueError(
                            f"Fuse({idx}) is nested in another Fuse of {idx}."
                        )
                    stmts = _fused_statements(body)
                    validate_body(idx, stmts)
                    validate(body, (*fused_idxs, idx), alone and len(stmts) == 1)
                case Query(_, Table(_, arg_idxs) as rhs) | QueryInto(
                    _, _, Table(_, arg_idxs) as rhs
                ) if not alone and tuple(
                    idx for idx in arg_idxs if idx in fused_idxs
                ) != tuple(idx for idx in fused_idxs if idx in arg_idxs):
                    raise ValueError(
                        f"{node} reads {rhs} out of loop order, which revisits "
                        "its fields, so it needs Fuse nodes of its own."
                    )
                case Table(FusedAlias(tns, n), idxs):
                    if n > len(fused_idxs) or idxs[:n] != fused_idxs[:n]:
                        raise ValueError(
                            f"Fused indices of {tns} must match the enclosing "
                            "Fuse fields in order."
                        )
                case LogicTree():
                    for child in node.children:
                        validate(child, fused_idxs, alone)

        validate(term, (), True)


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
