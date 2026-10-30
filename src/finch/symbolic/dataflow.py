import copy
import heapq
from abc import ABC, abstractmethod
from collections.abc import Callable


class BasicBlock:
    """Linear sequence of statements with a single entry and exit.

    Successor/predecessor lists encode the CFG edges. Statements execute in
    order; no internal branching occurs within a BasicBlock.
    """

    def __init__(self, id: str) -> None:
        self.id = id
        self.statements: list = []
        self.successors: list[BasicBlock] = []
        self.predecessors: list[BasicBlock] = []

    def add_statement(self, statement) -> None:
        self.statements.append(statement)

    def add_successor(self, successor: "BasicBlock") -> None:
        if successor not in self.successors:
            self.successors.append(successor)

        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def __str__(self) -> str:
        """String representation of BasicBlock in LLVM style."""
        lines = []

        if self.successors:
            succ_names = [succ.id for succ in self.successors]
            succ_str = f" #succs=[{', '.join(succ_names)}]"
        else:
            succ_str = " #succs=[]"

        # Block header
        lines.append(f"{self.id}:{succ_str}")

        # Block statements
        lines.extend(f"    {stmt}" for stmt in self.statements)

        return "\n".join(lines)


class ControlFlowGraph:
    """Collection of BasicBlocks plus explicit ENTRY and EXIT nodes.

    Provides helpers to allocate uniquely named blocks and holds the global
    block mapping used by dataflow analyses.
    """

    def __init__(self) -> None:
        self.block_counter = 0
        self.block_name = ""
        self.blocks: dict[str, BasicBlock] = {}

        self.entry_block = self.new_block_custom("ENTRY")
        self.exit_block = self.new_block_custom("EXIT")

    def new_block(self) -> BasicBlock:
        bid = f"{self.block_name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def new_block_custom(self, name: str) -> BasicBlock:
        bid = f"{name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def __str__(self) -> str:
        """Print the CFG in LLVM style format."""
        blocks = list(self.blocks.values())

        # Use list comprehension with join for better performance
        block_strings = [str(block) for block in blocks]
        return "\n\n".join(block_strings)


class DataFlowAnalysis(ABC):
    """
    Base class for performing data flow analyses over a ControlFlowGraph.

    This abstract class implements a generic work-list algorithm that can run either
    in a "forward" or "backward" direction. Subclasses must provide IR-specific
    semantics by implementing the abstract methods: transfer, join, stmt_str and
    direction, and may override boundary_state.

    only blocks reachable from the boundary block (ENTRY for forward, EXIT for
    backward) are visited, and joins skip neighbors that weren't processed yet.
    so loops start optimistic and dead blocks (e.g. after a `return`) never get
    joined in. `transfer` must not mutate its input.

    blocks are visited in reverse postorder, so a block is joined after the
    neighbors before it.

    Key attributes
    - cfg: the ControlFlowGraph to analyze.
    - input_states: a mapping from basic block id to the state flowing into the
        block's transfer (its entry for forward, its exit for backward).
    - output_states: a mapping from basic block id to the state the block's
        transfer produces.
    - reached: ids of the blocks visited by `analyze`.
    """

    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.input_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }
        self.output_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }
        self.reached: set[str] = set()

    def __str__(self) -> str:
        """
        Return a string representation describing the current dataflow
        analysis state over the control-flow graph (CFG).

        The output lists each basic block in the order
        produced by self.cfg.blocks.values().
        For every block the printed form is:

            <block.id>: #succs=[<succ_id>, ...]
                <stmt_representation_for_first_stmt>
                <stmt_representation_for_second_stmt>
                ...

        statements are annotated with the state flowing into their transfer
        (before them for forward, after them for backward), statements of
        unreached blocks without annotations.
        """
        lines: list[str] = []
        forward = self.direction() == "forward"

        for block in self.cfg.blocks.values():
            succ_names = [succ.id for succ in block.successors]
            lines.append(f"{block.id}: #succs=[{', '.join(succ_names)}]")

            if block.id not in self.reached:
                lines.extend(f"    {stmt}" for stmt in block.statements)
            else:
                # deep copy, so printing never changes the stored states
                state = copy.deepcopy(self.input_states[block.id])
                stmts = block.statements if forward else block.statements[::-1]
                annotated = []
                for stmt in stmts:
                    annotated.append(f"    {self.stmt_str(stmt, state)}")
                    state = self.transfer([stmt], state)
                lines.extend(annotated if forward else annotated[::-1])

            lines.append("")

        return "\n".join(lines)

    @abstractmethod
    def stmt_str(self, stmt, state: dict) -> str:
        """
        Format a single statement given the current lattice state.
        Implement in subclasses or IR-specific abstract bases.
        """
        ...

    @abstractmethod
    def transfer(self, stmts, state: dict) -> dict:
        """
        Transfer function for the data flow analysis.
        This should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def join(self, state_1: dict, state_2: dict) -> dict:
        """
        Join function for the data flow analysis.
        This should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def direction(self) -> str:
        """
        Return the direction of the data flow analysis, either "forward" or "backward".
        This should be implemented by subclasses.
        """
        ...

    def boundary_state(self) -> dict:
        """state at the cfg boundary: out of ENTRY (forward) or into EXIT (backward)"""
        return {}

    def analyze(self) -> None:
        """run the work list until the states of all reached blocks are stable"""
        forward = self.direction() == "forward"
        start = self.cfg.entry_block if forward else self.cfg.exit_block

        def upstream(block: BasicBlock) -> list[BasicBlock]:
            return block.predecessors if forward else block.successors

        def downstream(block: BasicBlock) -> list[BasicBlock]:
            return block.successors if forward else block.predecessors

        rank = _depth_first(start, downstream)

        work_list = [(rank[start.id], start.id)]
        queued = {start.id}
        while work_list:
            _, block_id = heapq.heappop(work_list)
            queued.remove(block_id)
            block = self.cfg.blocks[block_id]

            # join the states of processed upstream neighbors
            states = [
                self.output_states[b.id]
                for b in upstream(block)
                if b.id in self.reached
            ]
            if block is start:
                states.insert(0, self.boundary_state())
            input_state = states[0].copy()
            for other in states[1:]:
                input_state = self.join(input_state, other)
            self.input_states[block.id] = input_state

            output_state = self.transfer(block.statements, input_state)
            if (
                block.id in self.reached
                and output_state == self.output_states[block.id]
            ):
                continue
            self.reached.add(block.id)
            self.output_states[block.id] = output_state

            for neighbor in downstream(block):
                if neighbor.id not in queued:
                    queued.add(neighbor.id)
                    heapq.heappush(work_list, (rank[neighbor.id], neighbor.id))


def _depth_first(
    start: BasicBlock, downstream: Callable[[BasicBlock], list[BasicBlock]]
) -> dict[str, int]:
    """reverse postorder rank of the blocks reachable from `start`"""
    postorder: list[str] = []
    visited = {start.id}
    # neighbors go last to first, so a loop body is ranked before the code after
    # the loop, since the cfg builder links the body first
    stack = [(start, reversed(downstream(start)))]
    while stack:
        block, neighbors = stack[-1]
        for neighbor in neighbors:
            if neighbor.id not in visited:
                visited.add(neighbor.id)
                stack.append((neighbor, reversed(downstream(neighbor))))
                break
        else:
            stack.pop()
            postorder.append(block.id)
    return {block_id: i for i, block_id in enumerate(reversed(postorder))}
