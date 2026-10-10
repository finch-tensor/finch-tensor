class CostModel(ABC, Form):
    @property
    @abstractmethod
    def coefficients(self):
        raise NotImplementedError()

    @abstractmethod
    def get_features(self, stmt::LogicStatement, factory:StatsFactory, binding_ftypes:dict[Alias, FType], binding_stats:dict[Alias, TensorStats]):
        raise NotImplementedError()

    @abstractmethod
    def predict_cost(self, stmt::LogicStatement, factory:StatsFactory, binding_ftypes:dict[Alias, FType], binding_stats:dict[Alias, TensorStats]):
        return np.dot(self.coefficients, self.get_features(stmt, factory, binding_ftypes, binding_stats))



class FlopsCostModel(CostModel,AliasedForm):
    def __init__(self):
        pass

    @property
    def coefficients(self):
        # Implementation for getting coefficients
        pass

    def get_features(self, stmt: LogicStatement, factory: StatsFactory, binding_ftypes: dict[Alias, FType], binding_stats: dict[Alias, TensorStats]):
        binding_stats = copy(binding_stats)
        features = np.zeros(len(self.coefficients))
        def collect_iter_space(self, expr: LogicExpression):
            match expr:
                case Aggregate(op, init, rhs, idxs):
                    stats = collect_iter_space(rhs)
                    stats = factory.mapjoin(op, init, rhs)
                    return stats
                case Mapjoin(op, *args):
                    stats = factory.mapjoin(op, *args)
                case Reorder(op, *args):
                    stats = factory.reorder(op, *args)
                    return stats
                case Relabel(op, *args):
                    stats = factory.relabel(op, *args)
                    return stats

        def visit_stmt(self, stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for stmt in bodies:
                        features += self.predict_cost(stmt, factory, binding_ftypes, binding_stats)
                    return
                case Query(lhs, rhs):
                case QueryInto(lhs, op, rhs):
                    features += collect_iter_space(rhs).estimate_nnz()