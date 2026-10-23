from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

import numpy as np

from finch.algebra import FType, SingletonOperatorFType, StructFType, ffuncs
from finch.algebra.ftypes import FDType
from finch.symbolic import (
    DataFlowAnalysis,
    Namespace,
    PostOrderDFS,
    PostWalk,
    Rewrite,
)

from .cfg_builder import (
    NumberedStatement,
    assembly_build_cfg,
    assembly_dataflow_postprocess,
    assembly_dataflow_preprocess,
)
from .nodes import (
    AssemblyNode,
    AssemblyStatement,
    Assign,
    Block,
    BufferLoop,
    Call,
    ForLoop,
    Function,
    GetAttr,
    If,
    IfElse,
    Literal,
    Module,
    SetAttr,
    Variable,
    WhileLoop,
)
from .utils import get_vars_in_expr

# Type variable for dataflow analysis context,
# bound to AbstractAssemblyDataflow or its subclasses.
AnalysisT = TypeVar("AnalysisT", bound="AbstractAssemblyDataflow")


def literal_key(val) -> tuple | None:
    """exact key of a scalar literal value, since `Literal` compares by value only"""
    match val:
        case bool() | int():
            return ("const", type(val), val)
        case float() | complex() | np.number() | np.bool_():
            return ("const", type(val), np.asarray(val).tobytes())
    return None


def pure_op_key(op) -> object | None:
    """key of a pure operator, or None if calling `op` may have effects"""
    # user-defined finch operators can mutate their arguments, `ffuncs` ones don't
    if type(op).__module__ != ffuncs.__name__:
        return None
    match op.ftype:
        case SingletonOperatorFType() as op_type:
            return op_type
    # e.g. `choose(0.0) == choose(-0.0)`, so key by identity
    return ("id", id(op))


def is_scalar_type(type_: FType | None) -> bool:
    match type_:
        case StructFType():
            return False
        case FDType():
            return True
    return False


def has_effects(stmt: AssemblyNode) -> bool:
    for node in PostOrderDFS(stmt):
        match node:
            case Call(Literal(op), _) if pure_op_key(op) is not None:
                pass
            case Call():
                return True
    return False


def variables_read(stmt: AssemblyNode) -> list[str]:
    match stmt:
        case NumberedStatement(stmt, _):
            pass
    match stmt:
        case Assign(_, rhs):
            stmt = rhs
    return list(get_vars_in_expr(stmt))


def root_of(node: AssemblyNode) -> AssemblyNode:
    """the variable or slot under a getattr chain"""
    match node:
        case GetAttr(obj, _):
            return root_of(obj)
    return node


class ScopedRewriter:
    """
    walks a numbered program tracking the variables in scope, and rewrites each
    numbered statement with `rewrite`. scopes follow the strictest backend: the
    type checker scopes every block and the interpreter every body, and a
    variable is declared by its first assignment.
    """

    def __init__(self) -> None:
        self.scopes: list[dict[str, Variable]] = []

    def __call__(self, node: AssemblyNode) -> AssemblyNode:
        match node:
            case Module(funcs):
                return Module(tuple(self.function(func) for func in funcs))
            case Function():
                return self.function(node)
            case AssemblyStatement():
                return self.statement(node)
        return node

    def function(self, func: Function) -> Function:
        return Function(func.name, func.args, self.scoped(func.body, func.args))

    def scoped(
        self, body: AssemblyStatement, decls: Sequence[Variable] = ()
    ) -> AssemblyStatement:
        self.scopes.append({var.name: var for var in decls})
        try:
            match body:
                case Block(bodies):
                    return Block(self.block(bodies))
            kept = self.block((body,))
            match kept:
                case (AssemblyStatement() as stmt,):
                    return stmt
            return Block(kept)
        finally:
            self.scopes.pop()

    def block(self, bodies: Sequence) -> tuple:
        return tuple(self.statement(stmt) for stmt in bodies)

    def statement(self, node):
        match node:
            case Block():
                return self.scoped(node)
            case If(cond, body):
                return If(cond, self.scoped(body))
            case IfElse(cond, body, else_body):
                return IfElse(cond, self.scoped(body), self.scoped(else_body))
            case WhileLoop(cond, body):
                return WhileLoop(cond, self.scoped(body))
            case ForLoop(var, start, end, body):
                return ForLoop(var, start, end, self.scoped(body, (var,)))
            case BufferLoop(buf, var, body):
                return BufferLoop(buf, var, self.scoped(body, (var,)))
            case NumberedStatement(stmt, sid):
                result = NumberedStatement(self.rewrite(stmt, sid), sid)
                # an assignment declares its variable only after it runs
                match stmt:
                    case Assign(Variable(name, _) as var, _) if not self.visible(name):
                        self.scopes[-1][name] = var
                return result
        return node

    def visible(self, name: str) -> bool:
        return any(name in scope for scope in self.scopes)

    def rewrite(self, stmt: AssemblyStatement, sid: int) -> AssemblyStatement:
        return stmt


def assembly_dataflow_analyze(
    node: AssemblyNode, analysis_cls: type[AnalysisT]
) -> tuple[AnalysisT, AssemblyNode]:
    """
    Run preprocessing + CFG build + analysis for a dataflow pass.

    Returns:
        (analysis_ctx, preprocessed_node)
    """
    namespace = Namespace(node)
    pre_node, sid = assembly_dataflow_preprocess(node)
    ctx = analysis_cls(assembly_build_cfg(pre_node, sid=sid, namespace=namespace))
    ctx.analyze()
    return ctx, pre_node


def assembly_dataflow_run(
    node: AssemblyNode,
    analysis_cls: type[AnalysisT],
    apply: Callable[[AssemblyNode, AnalysisT], AssemblyNode],
) -> AssemblyNode:
    """
    Run a full dataflow pass (preprocess -> analyze -> apply -> postprocess).
    """
    ctx, pre_node = assembly_dataflow_analyze(node, analysis_cls)
    updated = apply(pre_node, ctx)
    return assembly_dataflow_postprocess(updated)


def assembly_copy_propagation(node: AssemblyNode) -> AssemblyNode:
    """
    Apply copy-propagation to a FinchAssembly node.
    Args:
        node: Root FinchAssembly node to optimize.
    Returns:
        AssemblyNode: The optimized FinchAssembly node.
    """

    return assembly_dataflow_run(
        node,
        AssemblyCopyPropagation,
        lambda pre_node, ctx: _CopyRewriter(ctx)(pre_node),
    )


class _CopyRewriter(ScopedRewriter):
    def __init__(self, ctx: "AssemblyCopyPropagation"):
        super().__init__()
        self.replacements = ctx.collect_copy_replacements()

    def rewrite(self, stmt: AssemblyStatement, sid: int) -> AssemblyStatement:
        def rename(node: AssemblyNode) -> AssemblyNode | None:
            match node:
                # the source of a copy may be declared in a scope that ended
                case Variable(name, type_) if self.visible(
                    new := self.replacements.get((sid, name), "")
                ):
                    return Variable(new, type_)
            return None

        rename_all = Rewrite(PostWalk(rename))
        match stmt:
            case Assign(lhs, rhs):
                return Assign(lhs, rename_all(rhs))
            case SetAttr(obj, attr, val):
                return SetAttr(obj, attr, rename_all(val))
        return rename_all(stmt)


class AbstractAssemblyDataflow(DataFlowAnalysis):
    """Assembly-specific base for dataflow analyses."""

    def stmt_str(self, stmt, state: dict) -> str:
        """Annotate a statement with lattice values.

        Delegates expression traversal and collection of (name,value) pairs to
        ``get_lattice_value`` which now returns the annotation list directly.
        """
        annotations = self.get_lattice_value(state, stmt)
        if annotations:
            annostr = ", ".join(f"{name} = {str(val)}" for name, val in annotations)
            return f"{stmt} \t# {annostr}"
        return str(stmt)

    def statement_states(self) -> Iterator[tuple[NumberedStatement, dict]]:
        """
        yield each numbered statement of the reached blocks with the state
        flowing into its transfer (before it for forward, after it for backward).
        """
        forward = self.direction() == "forward"
        for block in self.cfg.blocks.values():
            if block.id not in self.reached:
                continue
            state = self.input_states[block.id]
            for stmt in block.statements if forward else block.statements[::-1]:
                match stmt:
                    case NumberedStatement():
                        yield stmt, state
                state = self.transfer([stmt], state)

    @abstractmethod
    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        """Return list of (var_instance_name, lattice_value) pairs for a stmt/expr."""
        ...


class AssemblyCopyPropagation(AbstractAssemblyDataflow):
    """Copy propagation for FinchAssembly.

    Lattice:
    - defs: mapping ``{ var_name: sid | None }`` describing a unique reaching
        definition id for each variable (None means "not uniquely defined").
    - copies: mapping ``{ dst_var: (src_var, src_def_id) }`` describing a direct
        copy ``dst_var = src_var`` that is valid only if ``src_var`` still has the
        same unique reaching definition ``src_def_id``.
    """

    def direction(self) -> str:
        """Copy propagation is a forward analysis."""
        return "forward"

    def boundary_state(self) -> dict:
        return {"defs": {}, "copies": {}}

    def collect_copy_replacements(self) -> dict[tuple[int, str], str]:
        """Collect per-statement copy replacements.

        Returns:
            dict: Mapping ``(stmt_id, old_var_name) -> new_var_name`` indicating
                where ``old_var_name`` can be replaced by ``new_var_name`` at
                statement ``stmt_id`` based on copy-propagation facts.
        """
        replacements: dict[tuple[int, str], str] = {}
        for stmt, state in self.statement_states():
            for name, value in self.get_lattice_value(state, stmt):
                if isinstance(value, tuple) and value:
                    replacements[(stmt.sid, name)] = value[0]
        return replacements

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        """Collect lattice annotations for variables used in a stmt or expr."""
        annotated: list[tuple[str, object]] = []
        target = stmt

        if isinstance(target, NumberedStatement):
            target = target.stmt

        match target:
            case Assign(_, rhs) | SetAttr(_, _, rhs):
                target = rhs

        copies = state["copies"]

        for node in PostOrderDFS(target):
            match node:
                case Variable(name, _):
                    if name in copies:
                        annotated.append((name, copies[name]))
                case _:
                    continue
        return annotated

    def transfer(self, stmts: Iterable[Any], state: dict) -> dict:
        """Transfer function over a sequence of statements.

        Applies copy-propagation effects of each statement in order, returning
        the updated lattice mapping. Only copies of variables are recorded; any
        existing mappings that point to a variable being reassigned are
        invalidated first.

        Args:
            stmts: Iterable of Assembly statements in a basic block.
            state: Incoming lattice mapping.

        Returns:
            dict: The outgoing lattice mapping after processing ``stmts``.
        """
        new_state = {"defs": state["defs"].copy(), "copies": state["copies"].copy()}

        defs: dict[str, int | None] = new_state["defs"]
        copies: dict[str, tuple[str, int | None]] = new_state["copies"]

        for wrapped in stmts:
            sid, stmt = self._unpack_stmt(wrapped)
            match stmt:
                case Assign(Variable(lhs_name, _), rhs):
                    # Any assignment kills previous copy info involving lhs.
                    copies.pop(lhs_name, None)

                    # If some other variable was known to be a copy of lhs, kill it
                    # because lhs's value changed.
                    to_remove = [
                        dst for dst, (src, _) in copies.items() if src == lhs_name
                    ]
                    for dst in to_remove:
                        copies.pop(dst, None)

                    # Update reaching definition for lhs.
                    defs[lhs_name] = sid

                    # If rhs is a variable with a unique reaching def, record a copy.
                    if isinstance(rhs, Variable):
                        rhs_name = rhs.name
                        rhs_def = defs.get(rhs_name)
                        if rhs_def is not None:
                            copies[lhs_name] = (rhs_name, rhs_def)

                    # Ensure all recorded copies remain consistent with current defs.
                    new_state["copies"] = self._prune_inconsistent_copies(defs, copies)
                    copies = new_state["copies"]
                case SetAttr(obj, _, _):
                    # mlir rebinds the root of the target, so it gets a new value
                    match root_of(obj):
                        case Variable(name, _):
                            defs[name] = sid
                            new_state["copies"] = copies = {
                                dst: copy
                                for dst, copy in copies.items()
                                if name not in (dst, copy[0])
                            }
                case _:
                    continue

        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        """Meet operator for must copy-propagation.

        - defs join: keep a def id only if both agree, else None.
        - copies join: keep a copy only if both agree exactly, and it remains
          consistent with the joined defs.
        """
        defs_1: dict[str, int | None] = state_1["defs"]
        defs_2: dict[str, int | None] = state_2["defs"]
        copies_1: dict[str, tuple[str, int | None]] = state_1["copies"]
        copies_2: dict[str, tuple[str, int | None]] = state_2["copies"]

        joined_defs: dict[str, int | None] = {}
        for name in set(defs_1) | set(defs_2):
            v1 = defs_1.get(name)
            v2 = defs_2.get(name)
            joined_defs[name] = v1 if v1 == v2 else None

        joined_copies: dict[str, tuple[str, int | None]] = {
            dst: val
            for dst, val in copies_1.items()
            if dst in copies_2 and copies_2[dst] == val
        }

        joined_copies = self._prune_inconsistent_copies(joined_defs, joined_copies)

        return {"defs": joined_defs, "copies": joined_copies}

    def _unpack_stmt(self, stmt):
        if isinstance(stmt, NumberedStatement):
            return stmt.sid, stmt.stmt
        return None, stmt

    def _prune_inconsistent_copies(self, defs: dict, copies: dict) -> dict:
        pruned: dict[str, tuple[str, int | None]] = {}
        for dst, (src, src_def) in copies.items():
            if src_def is None:
                continue
            if defs.get(src) != src_def:
                continue
            pruned[dst] = (src, src_def)
        return pruned
