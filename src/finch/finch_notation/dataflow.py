from collections.abc import Callable, Sequence
from typing import TypeVar

from finch.symbolic import (
    DataFlowAnalysis,
    PostOrderDFS,
    PostWalk,
    Rewrite,
    UnvalidatedForm,
)

from .cfg_builder import (
    NumberedStatement,
    notation_build_cfg,
    notation_dataflow_postprocess,
    notation_dataflow_preprocess,
)
from .nodes import (
    Assign,
    Module,
    NotationExpression,
    NotationNode,
    NotationStatement,
    Repack,
    Variable,
)
from .stages import NotationTransform

NotationCFGEntry = NumberedStatement | NotationExpression

# Type variable for dataflow analysis context,
# bound to DataFlowAnalysis or its subclasses.
AnalysisT = TypeVar("AnalysisT", bound="DataFlowAnalysis")


def notation_dataflow_analyze(
    node: NotationNode, analysis_cls: type[AnalysisT]
) -> tuple[AnalysisT, NotationNode]:
    """
    Run preprocessing + CFG build + analysis for a dataflow pass.

    Returns:
        (analysis_ctx, preprocessed_node)
    """
    pre_node, sid = notation_dataflow_preprocess(node)
    ctx = analysis_cls(notation_build_cfg(pre_node, sid=sid))
    ctx.analyze()
    return ctx, pre_node


def notation_dataflow_run(
    node: NotationNode,
    analysis_cls: type[AnalysisT],
    apply: Callable[[NotationNode, AnalysisT], NotationNode],
) -> NotationNode:
    """
    Run a full dataflow pass (preprocess -> analyze -> apply -> postprocess).
    """
    ctx, pre_node = notation_dataflow_analyze(node, analysis_cls)
    updated = apply(pre_node, ctx)
    return notation_dataflow_postprocess(updated)


def notation_copy_propagation(node: NotationNode) -> NotationNode:
    """
    Apply copy propagation to a FinchNotation node.

    Args:
        node: Root FinchNotation node to optimize.
    Returns:
        NotationNode: The optimized FinchNotation node.
    """

    def apply(node: NotationNode, ctx: NotationCopyPropagation) -> NotationNode:
        replacements = ctx.collect_copy_replacements()

        def replace_vars(target: NotationNode, sid: int) -> NotationNode:
            def rw_var(n: NotationNode):
                match n:
                    case Variable(name, type_) if (sid, name) in replacements:
                        return Variable(replacements[(sid, name)], type_)
                return None

            return Rewrite(PostWalk(rw_var))(target)

        def rw(x: NotationNode):
            match x:
                case NumberedStatement(Assign(lhs, rhs), sid):
                    new_rhs = replace_vars(rhs, sid)
                    assert isinstance(new_rhs, NotationExpression)
                    return NumberedStatement(Assign(lhs, new_rhs), sid)
                case NumberedStatement(Repack(), _):
                    # Repack(slot, obj) defines obj, it does not read it
                    return None
                case NumberedStatement(stmt, sid):
                    new_stmt = replace_vars(stmt, sid)
                    assert isinstance(new_stmt, NotationStatement)
                    return NumberedStatement(new_stmt, sid)
            return None

        return Rewrite(PostWalk(rw))(node)

    return notation_dataflow_run(node, NotationCopyPropagation, apply)


class NotationCopyPropagationTransform(UnvalidatedForm, NotationTransform):
    def lower(self, term: Module) -> Module:
        result = notation_copy_propagation(term)
        assert isinstance(result, Module)
        return result


class NotationCopyPropagation(DataFlowAnalysis):
    """
    A dataflow analysis for copy propagation in finch notation.
    """

    def direction(self) -> str:
        """
        Copy propagation is a forward dataflow analysis
        """
        return "forward"

    def collect_copy_replacements(self) -> dict[tuple[int, str], str]:
        """Collect per-statement copy replacements.

        Returns:
            dict: Mapping ``(stmt_id, old_var_name) -> new_var_name`` indicating
                where ``old_var_name`` can be replaced by ``new_var_name`` at
                statement ``stmt_id``.
        """
        replacements: dict[tuple[int, str], str] = {}

        for block in self.cfg.blocks.values():
            state = self.input_states.get(block.id, {})

            for entry in block.statements:
                if isinstance(entry, NumberedStatement):
                    copies: dict[str, str] = state.get("copies", {})
                    # walk the AST of the statment to find all variables
                    # that can be replaced
                    for node in PostOrderDFS(entry.stmt):
                        match node:
                            case Variable(name, _) if name in copies:
                                src = copies[name]
                                # transitively follow the copy chain to find the source
                                while src in copies and src != name:
                                    src = copies[src]
                                replacements[(entry.sid, name)] = src

                # advance state to the next statement in the block
                state = self.transfer([entry], state)

        return replacements

    @staticmethod
    def _effect(stmt: NotationNode) -> tuple[str | None, str | None]:
        """Determine what variables are defined / killed in a NotationStatement.

        Args:
            stmt: A NotationStatement to analyze.

        Returns:
            A tuple (defined_name, copy_source_name) where:
                - defined_name: The name of the variable that is defined (written)
                by the statement, or None if no variable is defined.
                - copy_source_name: Source of the copy
        """
        match stmt:
            case Assign(Variable(name, _), Variable(src, _)):
                return name, src
            case Assign(Variable(name, _), _):
                return name, None
            case Repack(_, Variable(name, _)):
                # `Repack(C_, C)` writes the slot back into the variable `C`,
                # so any fact mentioning `C` is stale afterwards.
                return name, None
            case _:
                return None, None

    def transfer(self, stmts: Sequence[NotationCFGEntry], state: dict) -> dict:
        """Apply the block transfer function

        Args:
            stmts: Iterable of statements in the current block
            state: Input lattice state for this block

        Returns:
            dict: Output lattice state for this block after processing stmts
        """
        in_copies: dict[str, str] = state.get("copies", {})
        gen: dict[str, str] = {}
        kill: set[str] = set()

        for entry in stmts:
            # Unwrap the finch notation statement from the NumberedStatement wrapper
            stmt = entry.stmt if isinstance(entry, NumberedStatement) else entry

            # determine what variables this statement defines
            defined, source = self._effect(stmt)
            if defined is None:
                continue

            kill.add(defined)

            # Overwrite any copy previously generated in this block
            gen.pop(defined, None)

            # Drop any existing copy of the variable since it is now stale
            for dst in [d for d, src in gen.items() if src == defined]:
                gen.pop(dst)

            # Skip self-assignment
            if source is not None and source != defined:
                gen[defined] = source

        out_copies = {
            dst: src
            for dst, src in in_copies.items()
            if dst not in kill and src not in kill
        }
        out_copies.update(gen)
        return {"copies": out_copies}

    def join(self, state_1: dict, state_2: dict) -> dict:
        """
        Merge two predecessor states by intersecting their copy facts
        """
        # Unevaluated predecessor contributes nothing and leaves the other side intact
        if "copies" not in state_1:
            return state_2
        if "copies" not in state_2:
            return state_1

        copies_1: dict[str, str] = state_1["copies"]
        copies_2: dict[str, str] = state_2["copies"]

        # Return the intersection of the two predessor states
        return {
            "copies": {
                dst: src for dst, src in copies_1.items() if copies_2.get(dst) == src
            }
        }

    def stmt_str(self, stmt: NotationCFGEntry, state: dict) -> str:
        """
        Format a single statement given the current lattice state
        """
        str_state = ", ".join(
            f"{dst} -> {src}" for dst, src in state.get("copies", {}).items()
        )
        return f"Copies: {{{str_state}}} | Stmt: {stmt}"
