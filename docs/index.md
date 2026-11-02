# Finch: Flexible, Optimized Sparse Array Programming Suite


!!! warning "Stability Notice"
    Finch is *pre-alpha software*. Sparse array optimization is an active
    field of research, and sharp corners are present which may be confusing
    to users unfamiliar with theory of sparse array optimizations.


## Sparse Array Definition.

A sparse array is an ordered sequence of values, where it is advantageous to reason about the sequence as being primarily composed of a singular value at most positions, which is called the fill value, and other values elsewhere. A very common matrix that fits this definition is an identity matrix.


## Finch is special.
There have been many sparse array projects prior to this project, but this project is special for a couple of reasons:

| Solution      | Maximum Dimensions        | Optimizing | Extensible with custom sparse format |
| :---          | :---                      | :---:      | ---:                                 |
| Finch         | word size  (usually 2^64) | YES        | YES                                  |
| Scipy Sparse  | 2                         | NO         | NO                                   |
| Pydata Sparse | 2                         | NO         | NO                                   |




## Dimensionality

Previous work in the python ecosystem often restricts the user to only two dimensions worth of sparsity. Finch has no such restriction. This makes finch extremely powerful for a much broader class of problems.

### An optimizing runtime.


Finch is meant to handle large sparse programming problems, and has methods of optimizing operations
such as re-ordering iteration orders and matrix multiplications minimizing the use of work and resources.
Other runtimes often to do not optimize sparse problems, and are therefore slower than Finch.

### Extensibility

While many sparse array formats already exist, some computations may require a novel format, as certain problem may have attributes that can be more informative when represented in an uncommon way, not yet available in Finch's codebase. In other sparse array programming suites, this is quite limiting, as a new implementation for every operator must be provided, not only for the format with itself, but also with every other format it interfaces with. Finch, however, requires a couple definitions per type, which the finch library the generalizes to all the common operations, both with the novel data format, and with all existing data formats. This is an asymptotic reduction in the amount of work required to fully integrate and test a novel format.

Finch allows end-users to create their own sparse array data-types, and then automatically optimizes those sparse arrays to interface with all prior and future novel sparse array data structures.

## Multiple evaluation runtimes

Finch does not run purely in Python, or purely in a C context compiled for your operating system.

Finch can:

 - JIT a calculation to execute it much faster than standard python.
 - Generate, compile, and execute sparse programs in C code.
 - Generate and run MLIR, an intermediate code representation, which can take advantage of hardware specifics to optimize problem runtime.
 - Execute the calculation in Python, for debugging and when speed is not a priority.







### Background
Finch is a pure-Python rewrite of [`Finch.jl`](https://github.com/finch-tensor/Finch.jl).
