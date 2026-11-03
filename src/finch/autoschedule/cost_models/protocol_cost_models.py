from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from finch.algebra import ftypes, np_dtype
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Extrude,
    Field,
    Follow,
    Fuse,
    Gallop,
    Literal,
    LogicStatement,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS
from .format_cost_models import (
    ElementLevelOption,
    FormatBasis,
    LevelChoice,
    LevelOption,
    alias_blocks,
    write_protocol,
)
from .loop_cost_models import LoopSpaces, Prefix


class LevelBasis(FormatBasis, LoopOrderedForm):
    """
    The basis of the levels of one type of option. Each level contributes
    an allocation row for each block of its alias, and an access row for each
    query loop prefix that uses it. The features are the storage features of
    the option for the allocation, followed by walks, lookups, sequential
    writes, and scatter writes for the accesses.
    """

    def __init__(self, option_type: type[LevelOption]):
        self.option_type = option_type

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *self.option_type.feature_names,
            "walks",
            "lookups",
            "sequential_writes",
            "scatter_writes",
        )

    def to_json(self) -> dict[str, Any]:
        return {**super().to_json(), "option": self.option_type.kind}

    @classmethod
    def _from_json(cls, data: Mapping[str, Any]) -> Self:
        return cls(LevelOption.kinds[data["option"]])

    def get_access_features(
        self,
        alias: Alias,
        level: int,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the accesses to alias at a zero-based stored depth in each query
        of term: walks, lookups, sequential writes, and scatter writes, with one
        row per loop prefix that accesses the level in each query.
        A level is reached at the last loop over a field of the levels from the
        root to it, since it is reached only once the positions of its parents
        are known. The loops are the fields of the enclosing Fuse nodes. A read
        which walks the field of that loop steps through the level, and any
        other read looks it up wherever the loop visits, as does a write. A
        write is sequential where it extrudes the level, and scatters where it
        laminates it, as write_protocol infers. Every
        view of alias accesses its levels. The element level is accessed with
        its parent, and looks up its value.
        """
        features = []
        stats_interpreter = StatsMachine(
            machine.stats_factory, OrderedDict(machine.bindings)
        )

        def visit_access(table, loop_order, rhs, lhs_idxs, is_write, prefixes):
            tns, idxs = table.tns, table.idxs
            if not idxs:
                return
            # The levels of a fused alias follow its fused fields.
            depth = min(tns.nfused + level, len(idxs) - 1)
            order = tuple(idxs[: depth + 1])
            loops = loop_order[: max(map(loop_order.index, order)) + 1]
            spaces = LoopSpaces(Prefix(loops, rhs, lhs_idxs), stats_interpreter)
            features = prefixes.setdefault(loops, np.zeros(4))
            if is_write:
                visits = spaces.count(spaces.visited())
                extruded = write_protocol(tns, idxs, level, loop_order) == Extrude()
                features[2 if extruded else 3] += visits
            elif (
                self.option_type is not ElementLevelOption
                and loops[-1] == idxs[depth]
                and spaces.protocol(table) not in (Follow(), Gallop())
            ):
                features[0] += spaces.walk(table)
            else:
                features[1] += spaces.count(spaces.visited())

        def visit_stmt(stmt: LogicStatement, loop_order: tuple[Field, ...]):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body, loop_order)
                    return
                case Fuse(idx, body):
                    visit_stmt(body, (*loop_order, idx))
                    return
                case QueryInto(Table(), _, Literal()):
                    stats_interpreter(stmt)
                    return
                case Produces():
                    return
                case Query(Table(Alias() as lhs, lhs_idxs), Table() as rhs):
                    arg = rhs
                case Query(
                    Table(Alias() as lhs, lhs_idxs), Aggregate(_, _, arg, _) as rhs
                ):
                    pass
                case QueryInto(Table(Alias() as lhs, lhs_idxs), _, arg):
                    rhs = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            prefixes = {}
            for node in PostOrderDFS(arg):
                match node:
                    case Table(Alias() as tns, _) if tns.unfused == alias.unfused:
                        visit_access(node, loop_order, rhs, lhs_idxs, False, prefixes)
            if lhs.unfused == alias.unfused:
                table = Table(lhs, lhs_idxs)
                visit_access(table, loop_order, rhs, lhs_idxs, True, prefixes)
            features.extend(prefixes.values())
            stats_interpreter(stmt)

        visit_stmt(term, ())
        return np.asarray(features, dtype=float).reshape(-1, 4)

    def get_component_features(
        self,
        component: LevelChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the allocation row of the level in each block of its alias,
        followed by its access rows in the whole program.
        """
        blocks = alias_blocks(component.alias, component.term, machine)
        access_features = self.get_access_features(
            component.alias, component.depth, component.term, machine
        )
        n_storage = len(self.option_type.feature_names)
        features = np.zeros((len(blocks) + len(access_features), self.n_features))
        for row, ((stats, over), num_pos) in enumerate(
            zip(blocks, component.num_pos, strict=True)
        ):
            depth = len(over) + component.depth
            shape_type = stats.shape_type
            # The element level has no dimension, and a broadcast dimension is
            # intp.
            idx_size = 0
            if depth < len(shape_type):
                idx_size = np_dtype(shape_type[depth] or ftypes.intp).itemsize
            features[row, :n_storage] = component.option.get_storage_features(
                stats,
                machine.stats_factory,
                depth,
                num_pos,
                idx_size,
                component.pos_size,
                np_dtype(stats.element_type).itemsize,
                over,
            )
        features[len(blocks) :, n_storage:] = access_features
        return features
