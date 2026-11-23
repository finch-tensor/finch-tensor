# Compilation Targets

Most traditional python numerical computing tools operate by exposing functions to python which then call low level functions (often written in C and Fortran) to perform high-performance numerical calculations. Finch eschews this common approach and instead of simply instructing computations to be performed in the low level language, it implements a full compiler in Python, which transforms the high level python expressions into a low level language, called finch assembly which is then compiled to the target platform in a just-in-time fashion.


This means that when the user is writing finch, they have the option to choose which engine is executing their python programs which are using finch formats and datatypes.

Currently, there are a couple of targets that Finch uses. The most well supported target at the curent time is the C backend, which takes finch assembly, transpiles it to C, compiles the C code into a dynamic library, loads aforementioned dynamic library and calls it with the data provided in Python.

!!! info "Gold Standard"
    Finch's gold standard compiler is the C compiler. Status on the C compiler scheduler is described in [Issue #803](https://github.com/finch-tensor/finch-tensor/issues/803).


To set a finch compilation target for the duration of execution use:

```py
import finch

# set the finch compilation target
finch.autoschedule.set_default_scheduler(finch.autoschedule.INTERPRET_ASSEMBLY)
```


To temporarily set a compilation target, execution:
```py
import finch

with finch.autoschedule.with_default_scheduler(finch.autoschedule.INTERPRET_ASSEMLY):
    # code here will use interpret assembly by default
    pass
```


# Terminology

!!! info "Sharp Terminology"
    The terminology around schedulers is obfuscated by conflations and elisions. Discussions often assume collaborators can disambiguate the intentioned meaning through their knowledge of standard background compiler theory.

**finch logic**: high level, declarative, query based language describing computation to be performed on tensors. Lowers to finch notation. Usually derived by analying a computation written in Python syntax.
**finch notation**: declarative, control-flow based language describing computation to be performed on tensors. Lowers from finch logic. Lowers to finch assembly.
**finch assembly**: a typed, detailed, limited low level language representation, independent of target. Lowers from finch notation, lowers to system programming language backends.
**lowering**: the transformation from a higher level of implied detail to a lower level of realized detail, while preserving the semantics of the higher level.
**pass**: a traversal over a program which rewrites the program to a semantically equivalent program, usually used to optimize the program either for semantic interpretability or performance.
**target**: an environment for describing a numerical computation, often a low level, high performance language. Due to the low level nature of a target, often, targets are hardware specific.
**scheduler**:  A series of compiler lowerings and passes to reach a target format.

# The C Compilation Target.


Finch's flagship runtime evaluation target. Finch traces and compiles high level programs from python to finch-assembly, which it then transpiles to C. This backend supports all the functionality of Finch, and is an excellent first choice of evaluation context. The C compilation backend is notably different than libraries like numpy, because finch is not calling pre-defined procedures for numerical calculations implemented in C. Instead, finch itself is generating those procedures and compiling them with the system C compiler.

The C compiler is implemented in the `CCompiler()` which ingests finch assembly, compiles and executes the resulting C library. There is not currently a pre-defined autoscheduler which utilizes the C compiler, but work is in progress on implementing this.


# Multi-Level intermediate representation: MLIR

MLIR is implemented more as a JIT, where the compilation is done in-process after finch-assembly is transpiled to MLIR. The C compilation also does work just in time, but it does it out of process. MLIR is a very capable backend, but it may not support all the features that the C backend supports. Still, because it has a great deal of access to higher level linear algebra concepts, it may be able to outperform the C backend on certain workloads.

The MLIR backend is available within: `finch.autoschedule.COMPILE_MLIR`.

# Logic Interpreter

The logic interpreter is mostly used for debugging and is not meant for serious work regarding sparse tensor procedures. For the most part, the logic interpreter is too high level to effectively execute general finch programs. In general, finch logic _a la carte_ is not sufficient to compute programs because it is unaware of the data formats that it's describing operations over. This can, of course, be augmented and generated at runtime either dynamically (as with interpreters) or by JIT compiling a routine, but either way, more lowering has to be done for the expression to be computable.


The Logic Interpreter backend is avaiable by using: `finch.autoschedule.INTERPRET_LOGIC`.

# Notation Interpreter

Much like the logic interpreter, the notation interpeter also exists primarily to facilitate testing of the logic lowering layer as opposed to being a serious backend for computation. While finch notation has undergone enough lowering to have visibility into the formats that it is operating on, again, it is still a little bit too high level to be executed in this form, and it would not be executed more efficiently than the next phase. In addition, due to the stateful design of looplets, currently, a finch notation interpreter alone is not sufficient to traverse arbitrary sparse tensor formats, as looplets express their computations in the domain of finch assembly embedded inside of notation, rather than in pure notation.

The notation interpeter is available at `finch.autoschedule.INTERPRET_NOTATION`.


# Assembly Interpreter

Finch assembly is the last universal compilation target for finch, everything after finch assembly is specialized for the backend and eventually often specialized for the hardware. Finch assembly is either generated from the compiler lowering from notation, which itself usually comes from logic, or from handwritten routines in finch assembly such as looplet state variables and related procedures. The finch assembly interpreter is a perfectly workable, albeit very slow interpreter, but it is meant to be very simple and mostly serve the purpose of making debugging possible at a higher level. Most of the time, for novel operations and implementations, provided that the data is small, finch assembly suites the purpose of testing for correctness of generated programs.

The assembly interpeter is avaiable at `finch.autoschedule.INTERPRET_ASSEMBLY`

# Numba Compiler

[Numba](https://numba.pydata.org/) is an embedded JIT for a python-like language embedded within python. It essentially makes some semantic assumptions about how python code will be used in a way that is generally true of most python numerical programming routines, and JIT compiles them instead of having python perform its usual interpreting step. Technically, python always compiles to python bytecode, but this bytecode is then interpreted, rather than compiled to machine code. Numba differs here in that it can compile to machine code. Overall, Numba is relatively straightforward to use, and just requires generating a function with an `@njit` decorator and invoking said function. Finch does exactly this, it generates finch-assembly, transpiles that finch-assembly back to python and decorates it with `@njit`. Compiling python to python might seem slightly strange, but, in this case, the finch compiler is doing real work by expanding out and realizing the process for performing operations on sparse arrays. The numba compiler alone is unsuited for this task and the dynamic, duck-typed nature of python would make a JIT alone less performant compared to the looplets based lowering that python does.

Numba is a first class citizen of the finch compiler and most routines and numerical computations require no modification to execute in this context. There are two numba based schedulers avaiable by default in finch. One of which is `COMPILE_NUMBA` and then there is a slightly more advanced optimizing compiler named `COMPILE_NUMBA_GALLEY` which can express some computations in a much more effecient manner compared to `COMPILE_NUMBA`, albeit at _slightly_ higher compilation cost and complexity.

The Numba scheduler can be invoked either from `finch.autoschedule.COMPILE_NUMBA` or `finch.autoschedule.COMPILE_NUMBA_GALLEY`.


# The Julia Targets

!!! warning "Julia Targets"
    Julia is deprecated.

Julia is a programming language that proposes that scientific programs can be written in a high level feeling language, and then compiled JIT via [LLVM](https://llvm.org/) to achieve high performance and solve the [two language problem](https://discourse.julialang.org/t/two-language-problem-what-is-it/82925/2). Finch was originally written as a Julia library called [Finch.jl](https://github.com/), but Julia has not been successful at overtaking python in general usage (as of late 2026). There is a litany of reasons why this is the case, from less ecosystem support, Python's momentum, and Julia's high startup latency.




Julia is still technically supported via `COMPILE_JULIA` and `COMPILE_JULIA_GALLEY`, but efforts are underway to fully remove these options and migrate fully to a exclusively python based compiler.

# Looking Ahead

!!! quote "Finch should support that platform"

While finch covers a small scope of targets that it currently aspires to, engineering work is being performed to expand it to more computation environments.

## GPUs

GPUs can often perform matrix operations very quickly, but their architecture makes them less amenable to arbitrary computations compared to CPUs. Finch aspires to someday have sparse tensor routines work on the GPU. Details surrounding this effort are cataloged in [Issue #638](https://github.com/finch-tensor/finch-tensor/issues/638).


## On Disk

Some computations are not effectively able to be stored in RAM as sparse arrays due to their large size, and some computations will generate a large intermediate result, which makes them largely intractible under current sparse regimes as modern operating systems are not well optimized or designed to efficiently transfer intermediate results of sparse array formas between memory and disk as needed. Therefore, like many database implementations, finch intends to one day support full sparse matrix operations on disk in a memory constrained manner.

As sparse arrays are tautologically required by their existence to implement some operations more efficiently than dense arrays, expressing intermediate results, which may be of much higher cardinality than either input or output results complicates the procedure of storing routines on disk, and complicates tiling.
