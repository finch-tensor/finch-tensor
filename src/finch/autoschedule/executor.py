from collections import OrderedDict
from typing import Any

import numpy as np

from finch import finch_logic as lgc
from finch.algebra import is_dynamic, np_dtype
from finch.algebra.tensor import Tensor, TensorFType
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.finch_logic import LogicEvaluator, LogicLoader, LogicNode, StatsFactory
from finch.finch_logic.nodes import TableValue
from finch.symbolic import Namespace, PostWalk, Rewrite, UnvalidatedForm

from .formatter.formatter import BufferizedNDArrayFormatter


def extract_tensors(
    root: lgc.LogicStatement,
    bindings: dict[lgc.Alias, Tensor],
) -> tuple[lgc.LogicStatement, dict[lgc.Alias, Tensor]]:
    """
    Extracts tensors from logic plan, replacing them with aliases.
    """
    bindings = bindings.copy()
    # ids is a dictionary that has key value as memory_address : Alias
    ids: dict[int, lgc.Alias] = {id(val): key for key, val in bindings.items()}
    spc = Namespace(root)
    for alias in bindings:
        # Reserving the Alias names that already exist
        spc.freshen(alias.name)

    def rule_0(node):
        match node:
            # Case where we have table with actual tensor
            case lgc.Table(lgc.Literal(tns), idxs):
                if id(tns) in ids:
                    var = ids[id(tns)]
                    return lgc.Table(var, idxs)
                # If we don't have an Alias for the tensor we just found we create one
                var = lgc.HardAlias(spc.freshen("A"))
                # Updating the ids and bindings
                ids[id(tns)] = var
                bindings[var] = tns
                return lgc.Table(var, idxs)

    root = Rewrite(PostWalk(rule_0))(root)
    return root, bindings


class LogicExecutor(UnvalidatedForm, LogicEvaluator):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        stats_factory: StatsFactory | None = None,
        cache: bool = False,
    ):
        if ctx is None:
            ctx = BufferizedNDArrayFormatter()
        if stats_factory is None:
            stats_factory = DenseStatsFactory()
        self.ctx: LogicLoader = ctx
        self.stats_factory = stats_factory
        self.cache = cache
        self.cached_kernels: dict[tuple[Any, Any], Any] = {}

    def lower(
        self,
        prgm: LogicNode,
        bindings: dict[lgc.Alias, Tensor] | None = None,
    ):
        if bindings is None:
            bindings = {}
        if isinstance(prgm, lgc.LogicExpression):
            var = lgc.HardAlias("result")
            stmt: lgc.LogicStatement = lgc.Plan(
                (lgc.Query(lgc.Table(var, prgm.fields()), prgm), lgc.Produces((var,)))
            )
        elif isinstance(prgm, lgc.LogicStatement):
            stmt = prgm
        else:
            raise ValueError(f"Invalid prgm type: {type(prgm)}")
        if not isinstance(stmt, lgc.Plan):
            stmt = lgc.Plan((stmt,))

        stmt, bindings = extract_tensors(stmt, bindings)

        # A tensor's ftype says whether its fill may be specialized on, so a
        # dynamic fill already keeps the value out of the cache key and one
        # kernel serves many values. A backend that cannot express a runtime
        # fill must raise DynamicFillError and handle its own recovery.
        arg_ftypes: dict[lgc.Alias, TensorFType] = {
            var: val.ftype for var, val in bindings.items()
        }
        mod, binding_ftypes = self._load_cached(stmt, arg_ftypes, bindings)

        bindings = dict(zip(binding_ftypes.keys(), bindings.values(), strict=False))

        for var, tns_ftype in binding_ftypes.items():
            if var not in bindings:
                shape = (0,) * tns_ftype.ndim
                if is_dynamic(tns_ftype.fill_type):
                    # Declare replaces this typed placeholder inside the kernel.
                    value = np.zeros((), dtype=np_dtype(tns_ftype.element_type))[()]
                    bindings[var] = tns_ftype.construct(
                        shape, fill_value=tns_ftype.fill_type(value)
                    )
                else:
                    bindings[var] = tns_ftype.construct(shape)

        args = list(bindings.values())

        res = mod.main(*args)

        if isinstance(prgm, lgc.LogicExpression):
            return TableValue(res[0], prgm.fields())
        return tuple(res)

    def _load_cached(
        self,
        stmt: lgc.LogicStatement,
        binding_ftypes: dict[lgc.Alias, TensorFType],
        bindings: dict[lgc.Alias, Tensor],
    ):
        key = (stmt, tuple(binding_ftypes.items()))
        if self.cache and key in self.cached_kernels:
            return self.cached_kernels[key]

        stats_bindings = OrderedDict()
        for var, T in bindings.items():
            shape = T.shape
            fields = tuple(lgc.Field(f"d{i}") for i in range(len(shape)))
            stats_bindings[var] = self.stats_factory(T, fields)

        entry = self.ctx(
            stmt,
            binding_ftypes,
            stats_bindings,
            self.stats_factory,
        )

        if self.cache:
            self.cached_kernels[key] = entry
        return entry
