from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from finch.algebra import (
    cansplitpush,
    ffuncs,
    is_associative,
    is_commutative,
    is_distributive,
    repeat_operator,
)
from finch.algebra.algebra import FinchOperator
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicNode,
    LogicTree,
    MapJoin,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import gensym

from .logic_to_stats import insert_statistics

# A location in an expression tree. Unlike a node value, a path identifies one occurrence.
# So, rewrites addressed by path cannot confuse structurally equal subexpressions
# (e.g. two `Literal(2.0)` constants, repeated tables, ...).
Path = tuple[int, ...]


@dataclass
class AnnotatedQuery:
    stats_factory: StatsFactory
    output_name: Alias
    reduce_idxs: list[Field]
    point_expr: LogicExpression
    # Locations are stored as paths into `point_expr`, never as node values:
    # equal nodes at different positions are distinct locations.
    idx_lowest_path: OrderedDict[Field, Path]
    idx_op: OrderedDict[Field, Any]
    idx_init: OrderedDict[Field, Any]
    parent_idxs: OrderedDict[Field, list[Field]]
    original_idx: OrderedDict[Field, Field]
    connected_components: list[list[Field]]
    connected_idxs: OrderedDict[Field, set[Field]]
    bindings: OrderedDict[Alias, TensorStats]
    output_order: list[Field] | None = None

    def __init__(
        self,
        stats_factory: StatsFactory,
        q: Query,
        bindings: OrderedDict[Alias, TensorStats] | None = None,
    ):
        """
        Build an `AnnotatedQuery` from a logical `Query`, extracting reduction
        structure and precomputing tensor statistics.

        Parameters
        ----------
        stats_factory : StatsFactory
            Concrete stats factory used to create statistics.
        q : Query
            Logical query of the form `Query(name, rhs)` whose `rhs` may contain
            `Aggregate` nodes.
        bindings : OrderedDict[Alias, TensorStats], optional
            Existing alias→stats environment to seed the analysis.
        """
        assert isinstance(q, Query), (
            "Annotated Queries can only be built from queries of the form: "
            "Query(lhs, rhs)"
        )
        self.stats_factory = stats_factory
        if bindings is None:
            bindings = OrderedDict()
        self.bindings = bindings
        cache: dict[object, TensorStats] = {}
        insert_statistics(
            self.stats_factory,
            q,
            bindings=bindings,
            replace=False,
            cache=cache,
        )
        self.cache = cache
        output_name = q.lhs
        expr = q.rhs
        output_order: None | list[Field] = []
        if isinstance(expr, Reorder):
            output_order = list(expr.idxs)
            expr = expr.arg
        else:
            output_order = None
        starting_reduce_idxs: list[Field] = []
        idx_starting_path: OrderedDict[Field, Path] = OrderedDict()
        idx_top_order: OrderedDict[Field, int] = OrderedDict()
        top_counter = 1
        idx_op: OrderedDict[Field, FinchOperator] = OrderedDict()
        idx_init: OrderedDict[Field, Any] = OrderedDict()

        def strip_aggregates(
            node: LogicNode,
        ) -> tuple[LogicNode, list[tuple[Field, Path, Literal, Literal, LogicNode]]]:
            """Remove Aggregate nodes bottom-up, recording each reduction index
            together with the path of the aggregate's argument in the stripped
            tree. An aggregate collapses to its argument, so that path is the
            aggregate's own position."""
            match node:
                case Aggregate(Literal() as op, Literal() as init, arg, idxs):
                    new_arg, records = strip_aggregates(arg)
                    for idx in idxs:
                        records.append((idx, (), op, init, new_arg))
                    return new_arg, records
                case LogicTree():
                    children = node.children
                    new_children = []
                    records = []
                    for i, child in enumerate(children):
                        new_child, child_records = strip_aggregates(child)
                        new_children.append(new_child)
                        records.extend(
                            (idx, (i, *path), op, init, arg)
                            for idx, path, op, init, arg in child_records
                        )
                    if new_children != children:
                        node = type(node).from_children(*new_children)
                    return node, records
                case _:
                    return node, []

        point_expr, agg_records = strip_aggregates(expr)
        for idx, agg_path, op_lit, init_lit, agg_arg in agg_records:
            idx_starting_path[idx] = agg_path
            idx_top_order[idx] = top_counter
            top_counter += 1

            if op_lit.val is None:
                idx_op[idx] = ffuncs.init_write(cache[agg_arg].fill_value)
                idx_init[idx] = cache[agg_arg].fill_value
            else:
                idx_op[idx] = op_lit.val
                idx_init[idx] = init_lit.val

            starting_reduce_idxs.append(idx)

        cache_point: dict[object, TensorStats] = {}
        insert_statistics(
            self.stats_factory,
            point_expr,
            bindings=bindings,
            replace=False,
            cache=cache_point,
        )
        self.cache_point = cache_point

        reduce_idxs: list[Field] = []
        original_idx: OrderedDict[Field, Field] = OrderedDict(
            (idx, idx) for idx in cache[q.rhs].index_order
        )
        idx_lowest_path: OrderedDict[Field, Path] = OrderedDict()
        # Repeat rewrites are keyed by the path of the node being wrapped, so
        # each occurrence is its own rewrite even when several occurrences are
        # structurally equal, and by reduction index within a path so each
        # index contributes its domain size exactly once. They are applied in
        # one pass after the loop; rewriting incrementally would let a later
        # rewrite match inside a subtree an earlier one inserted.
        repeats: dict[Path, OrderedDict[Field, tuple[FinchOperator, Any]]] = {}
        stats_point = cache_point[point_expr]
        for idx in starting_reduce_idxs:
            agg_op = idx_op[idx]
            idx_dim_size = stats_point.dim_sizes[idx]
            starting_path = idx_starting_path[idx]
            lowest_paths = AnnotatedQuery.find_lowest_roots(
                agg_op,
                idx,
                cast(
                    LogicExpression,
                    AnnotatedQuery.node_at(point_expr, starting_path),
                ),
                base=starting_path,
            )
            original_idx[idx] = idx
            if len(lowest_paths) == 1:
                idx_lowest_path[idx] = lowest_paths[0]
                reduce_idxs.append(idx)
            else:
                new_idxs = [
                    Field(f"{idx.name}_{i}")
                    for i, _ in enumerate(lowest_paths, start=1)
                ]
                for i, path in enumerate(lowest_paths):
                    node = AnnotatedQuery.node_at(point_expr, path)
                    if idx not in cache_point[node].index_order:
                        # If the lowest root doesn't contain the reduction index, we
                        # attempt to remove the reduction via a repeat_operator, i.e.
                        # ∑_i B_j = B_j*|Dom(i)|
                        f = repeat_operator(agg_op)
                        if f is None:
                            continue
                        repeats.setdefault(path, OrderedDict())[idx] = (
                            f,
                            idx_dim_size,
                        )
                        continue
                    new_idx = new_idxs[i]
                    idx_op[new_idx] = agg_op
                    idx_init[new_idx] = idx_init[idx]
                    idx_lowest_path[new_idx] = path
                    idx_starting_path[new_idx] = starting_path
                    original_idx[new_idx] = idx
                    reduce_idxs.append(new_idx)

        if repeats:
            # Repeats over several indices with the same operator compose by
            # multiplying their domain sizes, since the repeat operators are
            # power-like: (x*d1)*d2 == x*(d1*d2) and (x**d1)**d2 == x**(d1*d2).
            wrap_groups: dict[Path, list[tuple[FinchOperator, Any]]] = {}
            for path, per_idx in repeats.items():
                groups: list[tuple[FinchOperator, Any]] = []
                for f, size in per_idx.values():
                    if groups and groups[-1][0] == f:
                        groups[-1] = (f, groups[-1][1] * size)
                    else:
                        groups.append((f, size))
                wrap_groups[path] = groups

            def make_wrap(
                groups: list[tuple[FinchOperator, Any]],
            ) -> Callable[[LogicNode], LogicNode]:
                def wrap(node: LogicNode) -> LogicNode:
                    for f, factor in groups:
                        node = MapJoin(
                            Literal(f),
                            (cast(LogicExpression, node), Literal(factor)),
                        )
                    return node

                return wrap

            point_expr = cast(
                LogicExpression,
                AnnotatedQuery.replace_at(
                    point_expr,
                    {path: make_wrap(g) for path, g in wrap_groups.items()},
                ),
            )
            # Wrapping inserts MapJoin levels above the wrapped nodes, so every
            # stored path running through a wrapped position must be remapped.
            # The wrapped node sits at child 1 of each inserted MapJoin
            # (children are [op, node, factor]).
            wrap_paths = sorted(wrap_groups, key=len, reverse=True)

            def remap_through_wraps(path: Path) -> Path:
                for p in wrap_paths:
                    if path[: len(p)] == p:
                        levels = (1,) * len(wrap_groups[p])
                        path = (*p, *levels, *path[len(p) :])
                return path

            for idx, path in idx_lowest_path.items():
                idx_lowest_path[idx] = remap_through_wraps(path)
            for idx, path in idx_starting_path.items():
                idx_starting_path[idx] = remap_through_wraps(path)
            insert_statistics(
                self.stats_factory,
                point_expr,
                bindings=bindings,
                replace=False,
                cache=cache_point,
            )

        parent_idxs: OrderedDict[Field, list[Field]] = OrderedDict(
            (i, []) for i in reduce_idxs
        )
        connected_idxs: OrderedDict[Field, set[Field]] = OrderedDict(
            (i, set()) for i in reduce_idxs
        )
        for idx1 in reduce_idxs:
            idx1_op = idx_op[idx1]
            idx1_bottom = idx_lowest_path[idx1]
            for idx2 in reduce_idxs:
                idx2_op = idx_op[idx2]
                idx2_top = idx_starting_path[idx2]
                idx2_bottom = idx_lowest_path[idx2]
                # idx2's lowest root lies within idx1's lowest root subtree.
                if idx2_bottom[: len(idx1_bottom)] == idx1_bottom:
                    connected_idxs[idx1].add(idx2)
                mergeable_agg_op = (
                    idx1_op == idx2_op
                    and is_associative(idx1_op)
                    and is_commutative(idx1_op)
                )
                # If idx1 isn't a parent of idx2, then idx2 can't restrict the
                # summation of idx1. idx2 is a parent when its starting root
                # lies strictly within idx1's lowest root subtree.
                if (
                    len(idx2_top) > len(idx1_bottom)
                    and idx2_top[: len(idx1_bottom)] == idx1_bottom
                ) or (
                    not mergeable_agg_op
                    and idx_top_order[original_idx[idx2]]
                    < idx_top_order[original_idx[idx1]]
                ):
                    parent_idxs[idx1].append(idx2)

        connected_components = self.get_idx_connected_components(
            parent_idxs, connected_idxs
        )

        self.output_name = output_name
        self.reduce_idxs = reduce_idxs
        self.point_expr = cast(LogicExpression, point_expr)
        self.idx_lowest_path = idx_lowest_path
        self.idx_op = idx_op
        self.idx_init = idx_init
        self.parent_idxs = parent_idxs
        self.original_idx = original_idx
        self.connected_components = connected_components
        self.connected_idxs = connected_idxs
        self.output_order = output_order

    def copy(self) -> "AnnotatedQuery":
        """
        Make a structured copy of an AnnotatedQuery.
        """
        new = object.__new__(AnnotatedQuery)
        new.stats_factory = self.stats_factory
        new.output_name = self.output_name
        new.point_expr = self.point_expr
        new.reduce_idxs = list(self.reduce_idxs)
        new.idx_lowest_path = OrderedDict(self.idx_lowest_path.items())
        new.idx_op = OrderedDict(self.idx_op.items())
        new.idx_init = OrderedDict(self.idx_init.items())
        new.parent_idxs = OrderedDict((m, list(n)) for m, n in self.parent_idxs.items())
        new.original_idx = OrderedDict(self.original_idx.items())
        new.connected_components = [list(n) for n in self.connected_components]
        new.connected_idxs = OrderedDict(
            (m, set(n)) for m, n in self.connected_idxs.items()
        )
        new.output_order = (
            None if self.output_order is None else list(self.output_order)
        )
        new.bindings = OrderedDict(self.bindings.items())
        new.cache = OrderedDict(self.cache.items())
        new.cache_point = OrderedDict(self.cache_point.items())

        return new

    def get_reducible_idxs(self) -> list[Field]:
        """
        Indices eligible to be reduced immediately (no parents).

        Parameters
        ----------
        aq : AnnotatedQuery
            Query containing the candidate reduction indices and their parent map.

        Returns
        -------
        list[Field]
            Field objects in `aq.reduce_idxs` with zero parents.
        """
        return [
            idx for idx in self.reduce_idxs if len(self.parent_idxs.get(idx, [])) == 0
        ]

    def get_reducible_idxs_for_component(self, component: list[Field]) -> list[Field]:
        """
        Indices in this component that have no parents (reducible now).

        Parameters
        ----------
        component : list[Field]
            A connected component of reduction indices.

        Returns
        -------
        list[Field]
            Field objects in the component that are reducible (zero parents).
        """
        return sorted(
            set(component).intersection(self.get_reducible_idxs()),
            key=lambda field: field.name,
        )

    @staticmethod
    def get_idx_connected_components(
        parent_idxs: Mapping[Field, Iterable[Field]],
        connected_idxs: Mapping[Field, Iterable[Field]],
    ) -> list[list[Field]]:
        """
        Compute connected components of indices (Field objects) and order those
        components by parent/child constraints.

        Parameters
        ----------
        parent_idxs : Dict[Field, Iterable[Field]]
            Mapping from an index to the iterable of its parent indices.
        connected_idxs : Dict[Field, Iterable[Field]]
            Mapping from an index to the iterable of indices considered
            "connected" to it (undirected neighbors). Only connections between
            non-parent pairs are used to form components.

        Returns
        -------
        List[List[Field]]
            A list of components, each a list of Field objects. Components are
            ordered so that any component containing a parent appears before any
            component containing its child.
        """
        parent_map: dict[Field, set[Field]] = {
            k: set(v) for k, v in parent_idxs.items()
        }
        conn_map: OrderedDict[Field, set[Field]] = OrderedDict(
            (k, set(v)) for k, v in connected_idxs.items()
        )

        component_ids: OrderedDict[Field, int] = OrderedDict(
            (x, i) for i, x in enumerate(conn_map.keys())
        )

        finished = False
        while not finished:
            finished = True
            for idx1, neighbours in conn_map.items():
                for idx2 in neighbours:
                    if idx2 in parent_map.get(idx1, set()) or idx1 in parent_map.get(
                        idx2, set()
                    ):
                        continue
                    if component_ids[idx2] != component_ids[idx1]:
                        finished = False
                    component_ids[idx2] = min(component_ids[idx2], component_ids[idx1])
                    component_ids[idx1] = min(component_ids[idx2], component_ids[idx1])

        unique_ids = list(OrderedDict.fromkeys(component_ids[idx] for idx in conn_map))
        components: list[list[Field]] = []
        for id in unique_ids:
            members = [idx for idx in conn_map if component_ids[idx] == id]
            components.append(members)

        component_order: OrderedDict[tuple[Field, ...], int] = OrderedDict(
            (tuple(c), i) for i, c in enumerate(components)
        )

        finished = False
        while not finished:
            finished = True
            for component1 in components:
                for component2 in components:
                    is_parent_of_1 = False
                    for idx1 in component1:
                        for idx2 in component2:
                            if idx2 in parent_map.get(idx1, set()):
                                is_parent_of_1 = True
                                break
                        if is_parent_of_1:
                            break

                    if (
                        is_parent_of_1
                        and component_order[tuple(component2)]
                        > component_order[tuple(component1)]
                    ):
                        max_pos = max(
                            component_order[tuple(component1)],
                            component_order[tuple(component2)],
                        )
                        min_pos = min(
                            component_order[tuple(component1)],
                            component_order[tuple(component2)],
                        )
                        component_order[tuple(component1)] = max_pos
                        component_order[tuple(component2)] = min_pos
                        finished = False

        components.sort(key=lambda c: component_order[tuple(c)])
        return components

    @staticmethod
    def node_at(root: LogicNode, path: Path) -> LogicNode:
        """
        Return the node at `path`, a sequence of child indices from `root`.
        """
        node = root
        for i in path:
            node = cast(LogicTree, node).children[i]
        return node

    @staticmethod
    def replace_at(
        root: LogicNode,
        transforms: Mapping[Path, Callable[[LogicNode], LogicNode | None]],
    ) -> LogicNode:
        """
        Rebuild `root`, applying each transform to the node at its path.

        Parameters
        ----------
        root : LogicNode
            The expression to transform.
        transforms : Mapping[Path, Callable[[LogicNode], LogicNode | None]]
            For each path, a function from the node at that position to its
            replacement. Returning None removes the node, which is only
            meaningful for MapJoin arguments. All paths address positions in
            the original tree.

        Returns
        -------
        LogicNode
            `root` with every transform applied.

        Notes
        -----
        Children are rebuilt before a node's own transform runs, so nested
        transform paths compose (an outer transform receives the node with
        inner transforms already applied), and the single traversal never
        descends into subtrees a transform inserted -- an inserted node can
        never itself be matched, which is what makes wrapping a node in an
        expression that contains it safe.
        """

        def rebuild(node: LogicNode, path: Path) -> LogicNode | None:
            if isinstance(node, LogicTree) and any(
                len(p) > len(path) and p[: len(path)] == path for p in transforms
            ):
                children = node.children
                new_children = []
                for i, child in enumerate(children):
                    new_child = rebuild(child, (*path, i))
                    if new_child is None:
                        if not isinstance(node, MapJoin):
                            raise ValueError(
                                "Only MapJoin arguments can be removed, not "
                                f"children of {type(node).__name__}."
                            )
                        continue
                    new_children.append(new_child)
                if new_children != children:
                    node = type(node).from_children(*new_children)
            transform = transforms.get(path)
            if transform is not None:
                return transform(node)
            return node

        result = rebuild(root, ())
        if result is None:
            raise ValueError("Cannot remove the root of an expression.")
        return result

    @staticmethod
    def splice_paths(
        path: Path, replace_path: Path, removal_paths: Iterable[Path]
    ) -> Path:
        """
        Remap `path` through a splice: the node at `replace_path` was replaced
        and the MapJoin arguments at `removal_paths` (siblings of
        `replace_path`) were removed, shifting the positions of the arguments
        after them. Paths inside the replaced or removed subtrees map to the
        replacement's position, since the kernel they addressed now lives
        behind it.
        """
        removal_paths = list(removal_paths)
        if path[: len(replace_path)] == replace_path or any(
            path[: len(r)] == r for r in removal_paths
        ):
            path = replace_path
        for parent in {r[:-1] for r in removal_paths}:
            if len(path) > len(parent) and path[: len(parent)] == parent:
                child = path[len(parent)]
                child -= sum(
                    1 for r in removal_paths if r[:-1] == parent and r[-1] < child
                )
                path = (*parent, child, *path[len(parent) + 1 :])
        return path

    @staticmethod
    def find_lowest_roots(
        op: FinchOperator, idx: Field, root: LogicExpression, base: Path = ()
    ) -> list[Path]:
        """
        Compute the lowest MapJoin / leaf positions that a reduction over `idx`
        can be safely pushed down to in a logical expression.

        Parameters
        ----------
        op : FinchOperator
            The reduction operator (e.g., ffuncs.add) that we are trying to
            push down.
        idx : Field
            The index (dimension) being reduced over.
        root : LogicExpression
            The root logical expression under which we search for the lowest
            pushdown positions for the reduction.
        base : Path
            The path of `root` in the enclosing expression; returned paths
            extend it.

        Returns
        -------
        list[Path]
            The paths of the lowest positions in the expression tree where the
            reduction over `idx` with operator `op` can be safely pushed down.
            Paths address occurrences, so structurally equal subexpressions at
            different positions are reported separately.
        """
        match root:
            case MapJoin(Literal(FinchOperator() as mj_op), args):
                # A MapJoin's children are [op, *args]: argument i is child
                # i + 1.
                with_idx = [
                    (i, arg) for i, arg in enumerate(args) if idx in arg.fields()
                ]
                without_idx = [
                    (i, arg) for i, arg in enumerate(args) if idx not in arg.fields()
                ]

                if len(with_idx) == 1 and is_distributive(mj_op, op):
                    i, arg = with_idx[0]
                    return AnnotatedQuery.find_lowest_roots(
                        op, idx, arg, (*base, i + 1)
                    )

                if cansplitpush(op, mj_op):
                    roots_without = [(*base, i + 1) for i, _ in without_idx]
                    roots_with: list[Path] = []
                    for i, arg in with_idx:
                        roots_with.extend(
                            AnnotatedQuery.find_lowest_roots(
                                op, idx, arg, (*base, i + 1)
                            )
                        )
                    return roots_without + roots_with

                return [base]
            # A Literal is a zero-dimensional constant, so it is a leaf that a
            # reduction can be pushed down to just like a table.
            case Alias(_) | Table(_, _) | Reorder(_, _) | Literal(_):
                return [base]
            case _:
                raise ValueError(
                    f"There shouldn't be nodes of type {type(root).__name__} "
                    "during root pushdown."
                )

    def get_reduce_query(
        self, reduce_idx: Field
    ) -> tuple[Query, Path, list[Path], list[Field]]:
        """
        Extract the maximal kernel that depends on `reduce_idx` into a standalone
        reduction query, and return the information needed to splice the result
        back into the main expression.

        Parameters
        ----------
        reduce_idx : Field
            The index being reduced.

        Returns
        -------
        query : Query
            A new Query whose RHS is an Aggregate over the kernel that depends on
            `reduce_idx`.
        replace_path : Path
            The position in `self.point_expr` that will be replaced with the
            alias produced by `query`.
        removal_paths : list[Path]
            Positions of MapJoin arguments (siblings of `replace_path`) that
            become redundant after the replacement.
        reduced_idxs : list[Field]
            The list of indices actually reduced in `query`.
        """
        original_idx = self.original_idx[reduce_idx]
        reduce_op = self.idx_op[reduce_idx]
        root_path = self.idx_lowest_path[reduce_idx]
        root_node = cast(
            LogicExpression, AnnotatedQuery.node_at(self.point_expr, root_path)
        )
        query_expr: LogicExpression
        idxs_to_be_reduced: set[Field] = set()
        replace_path: Path = root_path
        removal_paths: list[Path] = []
        reducible_idxs = self.get_reducible_idxs()
        stats_cache = self.cache_point

        use_root = False
        match root_node:
            case MapJoin(Literal(FinchOperator() as op), args) if is_distributive(
                op, reduce_op
            ):
                # If you're already reducing one index, then it may
                # make sense to reduce others as well.
                # E.g. when you reduce one vertex of a triangle, you should
                # do the other two as well.
                args_with_reduce_idx = [
                    arg for arg in args if original_idx in stats_cache[arg].index_order
                ]
                kernel_idxs = set().union(
                    *(stats_cache[arg].index_order for arg in args_with_reduce_idx)
                )
                # Positions, not values: equal arguments at different positions
                # are distinct.
                relevant_pos = [
                    i
                    for i, arg in enumerate(args)
                    if set(stats_cache[arg].index_order).issubset(kernel_idxs)
                ]
                relevant_args = [args[i] for i in relevant_pos]
                if len(relevant_pos) == len(args):
                    replace_path = root_path
                else:
                    # A MapJoin's children are [op, *args]: argument i is child
                    # i + 1.
                    replace_path = (*root_path, relevant_pos[0] + 1)
                    removal_paths = [(*root_path, i + 1) for i in relevant_pos[1:]]
                query_expr = MapJoin(Literal(op), tuple(relevant_args))
                stats_cache[query_expr] = self.stats_factory.mapjoin(
                    op, *[stats_cache[arg] for arg in relevant_args]
                )
                relevant_pos_set = set(relevant_pos)
                for idx in reducible_idxs:
                    if self.idx_op[idx] != self.idx_op[reduce_idx]:
                        continue

                    pos_with_idx = {
                        i
                        for i, arg in enumerate(args)
                        if self.original_idx[idx] in stats_cache[arg].index_order
                    }
                    if idx in self.connected_idxs[
                        reduce_idx
                    ] and relevant_pos_set.issuperset(pos_with_idx):
                        idxs_to_be_reduced.add(idx)
            case _:
                use_root = True

        if use_root:
            query_expr = root_node
            replace_path = root_path
            reducible_idxs = self.get_reducible_idxs()
            for idx in reducible_idxs:
                if self.idx_op[idx] != self.idx_op[reduce_idx]:
                    continue
                if (
                    idx in self.connected_idxs[reduce_idx]
                    or self.idx_lowest_path[idx] == root_path
                ):
                    idxs_to_be_reduced.add(idx)

        final_idxs_to_be_reduced: list[Field] = []
        for idx in idxs_to_be_reduced:
            orig = self.original_idx[idx]
            if orig not in final_idxs_to_be_reduced:
                final_idxs_to_be_reduced.append(orig)
        reduced_idxs = list(idxs_to_be_reduced)
        final_idxs_to_be_reduced.sort(key=lambda f: f.name)

        agg_op = self.idx_op[self.original_idx[reduce_idx]]
        agg_init = self.idx_init[self.original_idx[reduce_idx]]

        query_expr = Aggregate(
            Literal(agg_op),
            Literal(agg_init),
            query_expr,
            tuple(final_idxs_to_be_reduced),
        )

        stats_cache[query_expr] = self.stats_factory.aggregate(
            agg_op,
            agg_init,
            tuple(final_idxs_to_be_reduced),
            stats_cache[query_expr.arg],
        )

        query = Query(Alias(gensym("A")), query_expr)
        return query, replace_path, removal_paths, reduced_idxs

    def reduce_idx(self, reduce_idx: Field, do_condense: bool = False) -> Query:
        """
        Perform a single reduction rewrite over `reduce_idx`, restructuring `aq`
        so that the portion of the expression dependent on `reduce_idx` becomes
        a standalone subquery.

        Steps:
        1. Use `get_reduce_query` to extract the maximal subexpression that
            depends on `reduce_idx` and package it into a new `Query`.
        2. Create a fresh `Alias` for this subquery and register its statistics.
        3. Replace the extracted kernel in `aq.point_expr` with that alias and
            remove any nodes that are no longer reachable.
        4. Update all index-related metadata in `aq`—roots, ops, inits, parent
            structure, connectivity, components, and the remaining reduction set.

        Parameters
        ----------
        reduce_idx : Field
            The index being reduced.
        aq : AnnotatedQuery
            The annotated query to rewrite in place.
        do_condense : bool.

        Returns
        -------
        Query
            The newly created `Query` whose RHS computes the reduced kernel; its
            alias is used in the updated `aq.point_expr`.
        """
        query, replace_path, removal_paths, reduced_idxs = self.get_reduce_query(
            reduce_idx
        )

        alias_expr = Alias(query.lhs.name)
        stats_cache = self.cache_point
        insert_statistics(
            self.stats_factory,
            query,
            self.bindings,
            replace=False,
            cache=stats_cache,
        )
        alias_idxs = list(self.bindings[alias_expr].index_order)

        alias_table = Table(alias_expr, tuple(alias_idxs))
        transforms: dict[Path, Callable[[LogicNode], LogicNode | None]] = {
            replace_path: lambda _node: alias_table
        }
        for removal_path in removal_paths:
            transforms[removal_path] = lambda _node: None
        new_point_expr = cast(
            LogicExpression,
            AnnotatedQuery.replace_at(self.point_expr, transforms),
        )
        new_reduce_idxs = [x for x in self.reduce_idxs if x not in reduced_idxs]
        new_idx_lowest_path: OrderedDict[Field, Path] = OrderedDict()
        new_idx_op: OrderedDict[Field, Any] = OrderedDict()
        new_idx_init: OrderedDict[Field, Any] = OrderedDict()
        new_parent_idxs: OrderedDict[Field, list[Field]] = OrderedDict()
        new_connected_idxs: OrderedDict[Field, set[Field]] = OrderedDict()
        for idx in self.idx_lowest_path:
            if idx in reduced_idxs:
                continue
            new_idx_lowest_path[idx] = AnnotatedQuery.splice_paths(
                self.idx_lowest_path[idx], replace_path, removal_paths
            )
            new_idx_op[idx] = self.idx_op[idx]
            new_idx_init[idx] = self.idx_init[idx]
            new_idx_op[self.original_idx[idx]] = self.idx_op[idx]
            new_idx_init[self.original_idx[idx]] = self.idx_init[idx]
            new_parent_idxs[idx] = [
                x for x in self.parent_idxs.get(idx, []) if x not in reduced_idxs
            ]
            new_connected_idxs[idx] = {
                x for x in self.connected_idxs.get(idx, set()) if x not in reduced_idxs
            }

        for idx in new_idx_lowest_path:
            for idx2 in new_idx_lowest_path:
                if new_idx_lowest_path[idx] == new_idx_lowest_path[idx2]:
                    new_connected_idxs[idx].add(idx2)
                    new_connected_idxs[idx2].add(idx)

        new_components = AnnotatedQuery.get_idx_connected_components(
            new_parent_idxs, new_connected_idxs
        )

        insert_statistics(
            self.stats_factory,
            new_point_expr,
            self.bindings,
            replace=True,
            cache=stats_cache,
        )

        self.reduce_idxs = new_reduce_idxs
        self.point_expr = new_point_expr
        self.idx_lowest_path = new_idx_lowest_path
        self.idx_op = new_idx_op
        self.idx_init = new_idx_init
        self.parent_idxs = new_parent_idxs
        self.connected_idxs = new_connected_idxs
        self.connected_components = new_components
        return query

    def get_remaining_query(self) -> Query:
        """
        Build a final `Query` from the remaining pointwise expression in `aq`.

        Always returns a `Query` binding ``self.output_name``.
        """
        expr = self.point_expr
        output_order = tuple(self.output_order or expr.fields())
        if not isinstance(expr, Table):
            expr = Aggregate(
                Literal(ffuncs.overwrite),
                Literal(self.cache_point[expr].fill_value),
                cast(LogicExpression, expr),
                (),
            )
        if self.output_order is not None:
            expr = Reorder(cast(LogicExpression, expr), output_order)
        return Query(self.output_name, expr)

    def get_cost_of_reduce_idx(self, reduce_idx: Field) -> float:
        """
        Get the estimated cost of reducing `reduce_idx` in the current `aq`.

        Parameters
        ----------
        reduce_idx : Field
            The index for which to estimate the reduction cost.

        Returns
        -------
        float
            The estimated cost of performing the reduction over `reduce_idx`
            in the current state of `aq`.
        """
        query, _, _, _ = self.get_reduce_query(reduce_idx)
        stats_cache = self.cache_point
        insert_statistics(
            self.stats_factory,
            query.rhs,
            self.bindings,
            replace=False,
            cache=stats_cache,
        )
        match query.rhs:
            case Aggregate() as agg:
                mat_stats = stats_cache[agg]
                comp_stats = stats_cache[agg.arg]
                if isinstance(mat_stats, NumericStats) and isinstance(
                    comp_stats, NumericStats
                ):
                    return (
                        10 * mat_stats.estimate_non_fill_values()
                        + comp_stats.estimate_non_fill_values()
                    )
                raise TypeError("Stats Class must be inherit from NumericStats")
        raise ValueError(
            "The root of the reduction query should always be an Aggregate node."
        )
