import finch.finch_einsum as ein
import finch.finch_logic as lgc
from finch.algebra import ffuncs, ftype, init_value
from finch.algebra.tensor import TensorFType
from finch.autoschedule.tensor_stats import TensorStats
from finch.finch_assembly.stages import AssemblyLibrary
from finch.finch_einsum import EinsumLoader, MockEinsumLoader
from finch.finch_logic import Alias, LogicStatement, StatsFactory
from finch.finch_logic.stages import LogicLoader
from finch.symbolic import PreOrderDFS

from .stages import FormattedForm, LogicEinsumLowerer


def generate_einsum_stmt(node: LogicStatement) -> ein.EinsumStatement:
    match node:
        case lgc.Plan(bodies):
            return ein.Plan(tuple(generate_einsum_stmt(body) for body in bodies))
        case lgc.Fuse(_, body):
            # Einsums don't have loop orders.
            return generate_einsum_stmt(body)
        case lgc.Query(
            lgc.Table(lgc.Alias(name), output_idxs),
            lgc.Aggregate(lgc.Literal(operation), lgc.Literal(init), arg, _),
        ):
            einidxs = tuple(ein.Index(field.name) for field in output_idxs)
            body = ein.Einsum(
                op=ein.Literal(operation),
                tns=ein.Alias(name),
                idxs=einidxs,
                arg=generate_einsum_expr(arg),
            )
            if operation != ffuncs.overwrite and init != init_value(
                operation.ftype, ftype(init)
            ):
                return ein.Plan(
                    (
                        ein.Einsum(
                            op=ein.Literal(ffuncs.overwrite),
                            tns=ein.Alias(name),
                            idxs=einidxs,
                            arg=ein.Literal(init),
                        ),
                        body,
                    )
                )
            return body
        case lgc.Query(
            lgc.Table(lgc.Alias(name), output_idxs),
            lgc.Aggregate(lgc.Literal(ffuncs.overwrite), _, arg, _),
        ):
            # An overwrite needs no initializing einsum. Einsum can't read fills,
            # so other reductions require a literal init.
            return ein.Einsum(
                op=ein.Literal(ffuncs.overwrite),
                tns=ein.Alias(name),
                idxs=tuple(ein.Index(field.name) for field in output_idxs),
                arg=generate_einsum_expr(arg),
            )
        case lgc.QueryInto(
            lgc.Table(lgc.Alias(name), output_idxs),
            lgc.Literal(operation),
            rhs,
        ):
            return ein.Einsum(
                op=ein.Literal(operation),
                tns=ein.Alias(name),
                idxs=tuple(ein.Index(field.name) for field in output_idxs),
                arg=generate_einsum_expr(rhs),
            )
        case lgc.Query(lgc.Table(lgc.Alias(name), output_idxs), rhs):
            assert isinstance(rhs, lgc.LogicExpression)
            einarg = generate_einsum_expr(rhs)
            return ein.Einsum(
                op=ein.Literal(ffuncs.overwrite),
                tns=ein.Alias(name),
                idxs=tuple(ein.Index(field.name) for field in output_idxs),
                arg=einarg,
            )
        case lgc.Produces(args):
            names = []
            for arg in args:
                assert isinstance(arg, lgc.Alias)
                names.append(ein.Alias(arg.name))
            return ein.Produces(tuple(names))
        case _:
            raise Exception(f"Unrecognized logic: {node}")


def generate_einsum_expr(
    ex: lgc.LogicExpression,
) -> ein.EinsumExpression:
    match ex:
        case lgc.MapJoin(lgc.Literal(operation), lgcargs):
            args = tuple([generate_einsum_expr(arg) for arg in lgcargs])
            return ein.Call(ein.Literal(operation), args)
        case lgc.Table(lgc.Alias(name), idxs):
            return ein.Access(
                tns=ein.Alias(name),
                idxs=tuple(ein.Index(idx.name) for idx in idxs),
            )
        case lgc.Literal(value):
            return ein.Literal(val=value)
        case _:
            raise Exception(f"Unrecognized logic: {ex}")


class EinsumGenerator(LogicEinsumLowerer):
    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        bindings_2 = {ein.Alias(var.name): val for var, val in bindings.items()}
        return (generate_einsum_stmt(prgm), bindings_2)


class LogicEinsumLoader(FormattedForm, LogicLoader):
    def __init__(
        self,
        ctx_lower: LogicEinsumLowerer | None = None,
        ctx_load: EinsumLoader | None = None,
    ):
        if ctx_lower is None:
            ctx_lower = EinsumGenerator()
        self.ctx_lower: LogicEinsumLowerer = ctx_lower
        if ctx_load is None:
            ctx_load = MockEinsumLoader()
        self.ctx_load: EinsumLoader = ctx_load

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, "TensorStats"],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
    ]:
        ein_prgm, ein_bindings = self.ctx_lower(prgm, bindings, stats, stats_factory)
        # Aliases which are defined before any other access are storage
        # placeholders, not initialized einsum inputs. Statements come in
        # program order, and a query's lhs before its rhs.
        accessed = set()
        defined = set()
        for node in PreOrderDFS(prgm):
            match node:
                case lgc.Query(lgc.Table(lgc.Alias(name), _), _) if (
                    name not in accessed
                ):
                    defined.add(ein.Alias(name))
                case lgc.Alias(name):
                    accessed.add(name)
        mod, ein_bindings = self.ctx_load(ein_prgm, ein_bindings, defined=defined)
        lgc_bindings: dict[lgc.Alias, TensorFType] = {
            lgc.HardAlias(var.name): val for var, val in ein_bindings.items()
        }
        return mod, lgc_bindings
