# Notation global value numbering

`notation_global_value_numbering(module, assume_pure_arithmetic=True,
assume_value_semantics=True)` replaces redundant scalar assignment RHS calls with
available variables.
`NotationGlobalValueNumberingTransform` exposes the same pass to the compiler.
Rewriting is opt-in. The caller must guarantee pure arithmetic/runtime hooks and
that scalar object identity is unobservable, including in opaque consumers and
returned values. Without both flags, the pass leaves the IR unchanged. The default
compiler has no Notation transforms; enable GVN only at a boundary that provides
both contracts:

```python
compiler = LogicCompiler(ctx_transforms=(
    NotationGlobalValueNumberingTransform(
        assume_pure_arithmetic=True,
        assume_value_semantics=True,
    ),
))
```

For example, under those contracts:

```text
x = a + b
if condition:
    y = a + b
else:
    y = a + b
```

Both assignments to `y` can become `y = x`. The pass keeps the assignments, including
their binding creation. This branch retains copy assignments and unused statements;
it does not add copy propagation or dead code elimination.

## Domain and convergence

Each function gets a fixed universe of supported variables, exact literals and
source call subexpressions. These values are hash-consed: variables use their name,
literals use their exact runtime class and encoded bits, and calls use operator
identity plus canonical ordered operand identities. Call hashing and equality stay
shallow even for deeply nested expressions. A state is an equivalence partition of
that universe plus definitely bound variable names. Integer class labels are
canonicalized to the smallest member index; their allocation history has no
semantic significance.

The order is inclusion of equality facts and definite bindings. Confluence retains
facts common to every incoming state. It intersects partitions by pairing each
member's incoming class labels, then closes congruence: equal operators applied to
equal ordered operands produce equal results. An empty state dictionary denotes
the separate optimistic TOP sentinel; entry instead has singleton classes and the
function's parameter bindings. Only entry-reachable blocks participate.

For `x = rhs`, transfer saves the old RHS class, detaches `x` and every source term
that reads `x`, and attaches the new `x` to surviving members of the RHS class.
Congruence closure follows. This reads the RHS before overwriting the destination;
ordinary `x = x` preserves the partition. Unsupported effects reset equalities.
Input states are not mutated. Stable block inputs are replayed to select rewrites.

The universe never grows during solving. Its finite equality and binding relations
give a finite-height domain; kill, generation and congruence are monotone. Starting
at TOP with the fixed entry boundary yields descending worklist iteration, so loop
analysis terminates without an iteration cap or newly allocated value numbers.

This is a finite-source-expression GVN adaptation, **not the full expression-pool
algorithm** in Saleena and Paleri's [A Simple Algorithm for Global Value Numbering](https://arxiv.org/html/1303.1880v2).
It can miss relationships requiring expressions absent from the source universe.
It does not perform SSA conversion, introduce phi nodes, fold constants, commute
operands or reassociate arithmetic.

## Purity, scope and availability

The supported arithmetic operators are Finch `add`, `sub` and `mul`. Expression
identity does not depend on Finch types or type inference. The pass does not contain
Python, NumPy or Finch scalar-type allowlists.

`assume_pure_arithmetic=True` is a caller contract that these operations are pure,
deterministic and well-typed for the values they receive. It also covers implicit
runtime hooks encountered by this IR: assignment validation, variable type checks,
condition truth conversion, and built-in extent/index conversion. This matters for
overloaded arithmetic, NumPy warning/error callbacks and custom Finch types.

Enable GVN together with the separate value-semantics contract:

```python
transform = NotationGlobalValueNumberingTransform(
    assume_pure_arithmetic=True,
    assume_value_semantics=True,
)
```

Pure arithmetic alone is insufficient to preserve Python object identity. Two
equal integer additions can allocate distinct objects; replacing the second with
a copy makes `x is y` true in an opaque observer or after returning both values.
An effect barrier at that observer cannot undo the earlier identity change. This
is why the value-semantics contract is separate from the purity contract.

Literal keys preserve exact runtime types and floating/complex bit patterns,
including signed zero. Arbitrary Python calls, tensor accesses, opaque host values,
repacking and tensor mutation are conservative barriers. GVN does not remove tensor
loads/stores or authorize shared output storage. Custom scalar annotations can be
numbered under the explicit contracts without GVN hashing, comparing or inferring
those annotations.

The flat CFG does not model lexical scope exit. Names first assigned in nested
scopes and loop-index names are therefore excluded globally from the expression
domain. This sacrifices precision while preventing references to expired bindings
and confusion between successive loop-index values and ordinary copies.
Functions containing unsupported loop extents are left unchanged: only the built-in
`make_extent` shape is modeled. Custom extents can run arbitrary code between
iterations outside the explicit CFG. Built-in index conversion is covered by the
purity contract.

A replacement must be a definitely bound variable equal to the entire RHS call.
The caller's contract includes well-typed supported arithmetic and assignments.
All RHS variable reads must be definitely bound. The pass does not rewrite
conditions, loop extents, returns or arbitrary subexpressions. Functions are
analyzed independently.

Functions with early or nested returns are left unchanged. The current interpreter
continues executing after `Return`, whereas the CFG treats it as an exit; limiting
optimization to terminal returns prevents those execution models from disagreeing.

## Cost and verification

Let N be the number of tracked terms and A the total call-operand edges. A partition
uses O(N) space; storing block states costs O(BN) for B blocks (plus binding sets).
Congruence closure repeatedly scans call signatures, with at most N-1 successful
class merges per closure. Its cost includes union-find operations and operand
edges; whole-pass cost multiplies this by statement transfers, predecessor merges
and worklist revisits. There is no claim of linear or constant-time GVN.

`test_notation_gvn.py` covers observable rewrites, execution preservation, branch
confluence, reassignment, loops, source scope, scalar identity, effect barriers and
pipeline integration, preservation of scalar identity without opt-in, and
preservation of the interpreter's execution after early or nested returns.
