class LevelOption(ABC):
    """
    A level the formatter may choose for a dimension. Its coefficients weigh
    the level features of a FormatCostModel.
    """

    def __init__(self, coefficients):
        self._coefficients = np.array(coefficients, dtype=float)

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @abstractmethod
    def next_num_pos(self, num_pos, n_l, nnz_l): ...


class DenseLevelOption(LevelOption):
    def next_num_pos(self, num_pos, n_l, nnz_l):
        return num_pos * n_l


class SparseHashLevelOption(LevelOption):
    def next_num_pos(self, num_pos, n_l, nnz_l):
        return nnz_l


class FormatCostModel(CostModel, LoopOrderedForm):
    """
    Each tensor level contributes an allocation component and an access component
    for each query loop prefix that uses it. Each entry of formats names a level
    option, whose coefficients supply the default linear weights for that format.
    """

    def __init__(
        self,
        formats: Mapping[str, LevelOption],
        element_coefficients,
        predictor: Callable[[np.ndarray], float] | None = None,
    ):
        self.formats = formats
        self.element_coefficients = np.array(element_coefficients, dtype=float)
        coefficients = np.concatenate(
            [
                *(format.coefficients for format in self.formats.values()),
                self.element_coefficients,
            ]
        )
        super().__init__(
            predictor if predictor is not None else LinearPredictor(coefficients)
        )

    @property
    def n_features(self) -> int:
        return 10 * (len(self.formats) + 1)

    def nnz_after(self, stats: NS, stats_factory: StatsFactory[NS], level: int):
        """Return the non-fill entries of the first level + 1 levels of stats."""
        reduce_fields = tuple(stats.index_order[level + 1 :])
        if reduce_fields:
            stats = stats_factory.aggregate(ffuncs.or_, False, reduce_fields, stats)
        return stats.estimate_non_fill_values()

    def get_level_features(self, num_pos, n_l, nnz_l, val_size, pos_size):
        """
        Return the features of a level with num_pos positions entering it, a
        dimension of n_l, and nnz_l non-fill entries: positions, slots, entries,
        position bytes, index bytes, and value bytes.
        """
        return np.array(
            [
                num_pos,
                num_pos * n_l,
                nnz_l,
                (num_pos + 1) * pos_size,
                nnz_l * pos_size,
                num_pos * val_size,
            ]
        )

    def get_access_features(
        self,
        alias: Alias,
        level: int,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the accesses to alias at a zero-based depth in each query of term:
        sequential reads, scatter reads, sequential writes, and scatter writes,
        with one row per loop prefix that accesses the level in each query.
        Each access is counted by the PigeonCostModel iterations of the loops
        from the root to the current level. The element level is accessed with
        its parent.
        """
        # A sequential write is when the prefix of loops from root to current
        # level matches the ordering of the access exactly.
        # A sequential read is when the prefix of loops from root to current
        # level is a supersequence of the ordering of the access.
        features = []
        stats_interpreter = StatsMachine(
            machine.stats_factory, OrderedDict(machine.bindings)
        )
        pigeon = PigeonCostModel()

        def visit_access(idxs, loop_order, rhs, lhs_idxs, is_write, prefixes):
            if not idxs:
                return
            depth = min(level, len(idxs) - 1)
            order = tuple(idxs[: depth + 1])
            loops = tuple(loop_order[: loop_order.index(idxs[depth]) + 1])
            iterations = pigeon.get_prefix_features(
                loops, rhs, stats_interpreter, lhs_idxs
            )[0]
            features = prefixes.setdefault(loops, np.zeros(4))
            if is_write:
                features[2 if loops == order else 3] += iterations
            else:
                features[0 if is_subsequence(order, loops) else 1] += iterations

        def visit_stmt(stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                    return
                case QueryInto(Table(), _, Literal()):
                    stats_interpreter(stmt)
                    return
                case Produces():
                    return
                case Query(Table(lhs, lhs_idxs), Table() as rhs):
                    arg = rhs
                    loop_order = lhs_idxs
                case Query(
                    Table(lhs, lhs_idxs),
                    Aggregate(_, _, Reorder(arg, loop_order), _) as rhs,
                ):
                    pass
                case QueryInto(Table(lhs, lhs_idxs), _, Reorder(arg, loop_order)):
                    rhs = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            prefixes = {}
            for node in PostOrderDFS(arg):
                match node:
                    case Table(tns, idxs) if tns == alias:
                        visit_access(idxs, loop_order, rhs, lhs_idxs, False, prefixes)
            if lhs == alias:
                visit_access(lhs_idxs, loop_order, rhs, lhs_idxs, True, prefixes)
            features.extend(prefixes.values())
            stats_interpreter(stmt)

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, 4)

    def get_format_features(
        self,
        alias: Alias,
        level: int,
        num_pos: float,
        format_key: str | None,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """Return the features at a zero-based depth with num_pos input positions.

        format_key selects a level option; None selects the element level at
        the tensor's rank. term is the LoopOrderedForm program which reads and
        writes alias. Each format has a block of ten columns: six allocation
        features followed by four access features. The first row describes the
        allocation; subsequent rows describe accesses at individual loop prefixes.
        """
        stats = machine(alias)
        fields = stats.index_order
        fill_ftype = ftype(stats.fill_value)
        leaf = element(stats.fill_value, fill_ftype)
        assert leaf.position_type is not None
        val_size = np_dtype(fill_ftype).itemsize
        pos_size = np_dtype(leaf.position_type).itemsize

        access_features = self.get_access_features(alias, level, term, machine)
        features = np.zeros((1 + len(access_features), self.n_features))
        if format_key is None:
            # The element level stores one value for each position.
            block = len(self.formats)
            n_l, nnz_l = 1, num_pos
        else:
            block = list(self.formats).index(format_key)
            n_l = stats.get_dim_size(fields[level])
            nnz_l = self.nnz_after(stats, machine.stats_factory, level)
        features[0, 10 * block : 10 * block + 6] = self.get_level_features(
            num_pos, n_l, nnz_l, val_size, pos_size
        )
        features[1:, 10 * block + 6 : 10 * block + 10] = access_features
        return features

    def predict_format_cost(
        self,
        alias: Alias,
        level: int,
        num_pos: float,
        format_key: str | None,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> float:
        features = self.get_format_features(
            alias, level, num_pos, format_key, term, machine
        )
        return self.predictor(features)

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, list[str]],
        stats: Mapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        """
        Return the features of the formats in bindings, which name the level
        option of each level of an alias, outermost first.
        """
        features = []
        stats_interpreter = StatsMachine(stats_factory, dict(stats))
        visited: set[Alias] = set()

        def visit_stmt(stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                case Query(Table(Alias() as lhs, _), _) | QueryInto(
                    Table(Alias() as lhs, _), _, _
                ):
                    stats_interpreter(stmt)
                    # Each alias is formatted once, when it is first written.
                    match bindings.get(lhs):
                        case list() as format_keys if lhs not in visited:
                            lhs_stats = stats_interpreter(lhs)
                            num_pos = 1.0
                            for level, key in enumerate((*format_keys, None)):
                                features.append(
                                    self.get_format_features(
                                        lhs,
                                        level,
                                        num_pos,
                                        key,
                                        term,
                                        stats_interpreter,
                                    )
                                )
                                if key is not None:
                                    n_l = lhs_stats.get_dim_size(
                                        lhs_stats.index_order[level]
                                    )
                                    nnz_l = self.nnz_after(
                                        lhs_stats, stats_factory, level
                                    )
                                    num_pos = self.formats[key].next_num_pos(
                                        num_pos, n_l, nnz_l
                                    )
                    visited.add(lhs)
                case Produces():
                    return
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        visit_stmt(term)
        if not features:
            return np.empty((0, self.n_features))
        return np.concatenate(features, axis=0)
