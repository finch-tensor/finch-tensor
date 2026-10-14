class LoopCostModel(CostModel):
    """
    A cost model where each loop prefix, including the empty prefix, is a component.
    get_prefix_features returns one feature vector; get_features stacks these
    vectors in query order, from the shortest prefix to the longest. Predictors used
    with branch-and-bound loop search must predict nonnegative component costs.
    """

    @abstractmethod
    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray: ...

    def predict_prefix_cost(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> float:
        features = self.get_prefix_features(prefix, expr, machine, output_vars)
        return self.predictor(features[None, :])

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        stats = dict(stats)
        features = []
        stats_interpreter = StatsMachine(stats_factory, stats)

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
                case Query(Table(_, lhs_idxs), Table() as out):
                    idxs = lhs_idxs
                case Query(
                    Table(_, lhs_idxs),
                    Aggregate(_, _, Reorder(_, idxs), _) as out,
                ):
                    pass
                case QueryInto(Table(_, lhs_idxs), _, Reorder(_, idxs)):
                    out = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            features.extend(
                self.get_prefix_features(idxs[:n], out, stats_interpreter, lhs_idxs)
                for n in range(len(idxs) + 1)
            )
            stats_interpreter(stmt)

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, self.n_features)


class MultiLoopCostModel(LoopCostModel, MultiCostModel):
    """Combine feature columns for matching loop prefixes."""

    def __init__(
        self,
        *submodels: LoopCostModel,
        predictor: Callable[[np.ndarray], float] | None = None,
    ):
        super().__init__(*submodels, predictor=predictor)
        self.submodels: tuple[LoopCostModel, ...] = submodels

    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray:
        return np.concatenate(
            [
                model.get_prefix_features(prefix, expr, machine, output_vars)
                for model in self.submodels
            ]
        )



class PigeonCostModel(LoopCostModel, LoopOrderedForm):
    """
    PigeonCostModel counts iteration, operation, and output costs across loop
    prefixes. By default, allocations are weighted 10x reads and operations.
    """

    @property
    def n_features(self) -> int:
        return 3

    def __init__(self, predictor: Callable[[np.ndarray], float] | None = None):
        super().__init__(
            predictor
            if predictor is not None
            else LinearPredictor(np.array([1.0, 1.0, 10.0]))
        )

    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray:
        """Return the incremental features for an outermost-first loop prefix.

        A nonempty prefix charges its last index, projecting execution and
        output onto prefix[:-1], as in the inside-out calculation. The empty
        prefix charges the output's non-fill values once. Summing all prefixes,
        including the empty one, gives the query's features.
        """
        stats_factory = machine.stats_factory
        match expr:
            case Aggregate(arg=arg):
                pass
            case _:
                arg = expr
        out_space = machine(expr)
        if output_vars is not None:
            reduce_idxs = tuple(
                idx for idx in out_space.index_order if idx not in output_vars
            )
            if reduce_idxs:
                out_space = stats_factory.copy(out_space)
                out_space.fill_value = StaticFill(False)
                out_space = stats_factory.aggregate(
                    ffuncs.or_, False, reduce_idxs, out_space
                )
            out_space = stats_factory.reorder(out_space, output_vars)
        if not prefix:
            return np.array([0.0, 0.0, out_space.estimate_non_fill_values()])

        def support(stat: NS) -> NS:
            stat = stats_factory.copy(stat)
            stat.fill_value = StaticFill(False)
            return stat

        def project(stat: NS) -> NS:
            reduce_idxs = tuple(idx for idx in stat.index_order if idx not in parent)
            return stats_factory.aggregate(
                ffuncs.or_, False, reduce_idxs, support(stat)
            )

        parent = set(prefix[:-1])
        idx = prefix[-1]
        exec_space = project(machine(arg))
        inputs = []
        for node in PostOrderDFS(arg):
            match node:
                case Table(_, idxs) if idx in idxs:
                    inputs.append(support(machine(node)))
        iter_space = exec_space
        if inputs:
            iter_space = stats_factory.mapjoin(
                ffuncs.and_, exec_space, stats_factory.mapjoin(ffuncs.or_, *inputs)
            )
        out_cost = 0.0
        if idx in out_space.index_order:
            out_cost = project(out_space).estimate_non_fill_values()
        return np.array(
            [
                iter_space.estimate_non_fill_values(),
                exec_space.estimate_non_fill_values() * len(inputs),
                out_cost,
            ]
        )


class TransposeCostModel(LoopCostModel, LoopOrderedForm):
    """
    Count dense and sparse transposed entries, charging each input once per query.
    """

    @property
    def n_features(self) -> int:
        return 2

    def __init__(self, predictor: Callable[[np.ndarray], float] | None = None):
        super().__init__(
            predictor
            if predictor is not None
            else LinearPredictor(np.array([0.5, 60.0]))
        )

    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray:
        """Charge only the input transpositions newly forced by this prefix.

        Deduplicate input tables within this prefix. Comparing with the parent
        prefix avoids charging them again at deeper loop levels. Empty prefixes
        have no transposition cost.
        """
        features = np.zeros(self.n_features)
        if not prefix:
            return features

        match expr:
            case Aggregate(arg=arg):
                pass
            case _:
                arg = expr
        for node in set(PostOrderDFS(arg)):
            match node:
                case Table(_, idxs) if prefix[-1] in idxs:
                    stat = machine(node)
                case _:
                    continue
            order = stat.index_order
            parent = tuple(idx for idx in prefix[:-1] if idx in order)
            if parent == order[: len(parent)] and prefix[-1] != order[len(parent)]:
                nnz = stat.estimate_non_fill_values()
                space = stat.get_dim_space_size(order)
                dense = 0 < space < float("inf") and nnz / space > 0.1
                features[0 if dense else 1] += nnz
        return features