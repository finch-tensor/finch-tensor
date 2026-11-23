# What Really Powers All Of This?



!!! warning "Intended Audience"
    Finch is based on active areas of research, subject to change. Do not depend on any of the functionality described here. Users generally will not need to understand exactly what powers Finch. If you would like to contribute to Finch, then these sections will hopefully be enlightening.


## The Fiber Tensor

The data structure that everyone interfaces with the most is called the FiberTensor. A Fiber Tensor exposes an API and structure over a recursive nesting of Levels. This recursive structure is what allows Finch to store arbitrarily many dimensions of data.


## Nesting Levels

While multi-dimensional array formats vary in how they are physically represented in memory, logically, multi-dimensional arrays are arrays of arrays. They are not necessarily required to be stored that way, but an array-of-arrays is sufficient to represent any arbitrarily high dimensional multi-dimensional array.

Each one of these arrays, is what Finch calls a "Level". Perhaps somewhat counter-intuitively, a level can be constructed such that it represents multiple dimensions, but the nesting structure is the same. Levels are nested within levels recursively until they reach the value level.


Because we don't expect this concept to be intuitive immediately, some examples below are provided.

### Corollary with standard python arrays

When we index standard python (not-Numpy) arrays, we use the array syntax. If we have a two-dimensional array, we have `[row][col]` indexing, and then beyond that, we have `[a][b][c]...]` indexing.

If we decompose this, we can see that at every indexing operation, we either return a list of lists, a list of values, or a value directly. So for an indexing expression like `X[a][b][c]`, we can decompose like so:

```py
# the following V1
V1 = X[a][b][c]

# is equivalent to V2 computed this way
F1 = X[a]
F2 = F1[b]
V2 = F2[c]


assert V1 == V2
```

At each step, we have produced a new smaller array with one less dimension.


Let's do a worked example,

```py
V = [[[1, 2], [2, 3]], [[4], [5, 2, 3]]]

V1 = V[1][1][2] # => 3

F1 = V[1] # => [[4], [5, 2, 3]]
F2 = F1[1] # [5, 2, 3]
V2 = F2[2] # => 3

assert V1 == V2
```

If we wanted to describe this, we have a list of lists of lists of values. We can implement this _exact_ behavior in another, interesting way.


```py
V = [[{0: 1, 1: 2}, {0: 2, 1: 3}], [{0: 4}, {0: 5, 1: 2, 2: 3}]]


V1 = V[1][1][2] # => 3

F1 = V[1] # => [{0: 4}, {0: 5, 1: 2, 2: 3}]
F2 = F1[1] # {0: 5, 1: 2, 2: 3}
V2 = F2[2] # => 3

assert V1 == V2
```


We have replaced the very last level of the array with a dictionary, and yet the actual program behavior in computing V1 and V2 is exactly the same. While in the above example, no efficiency is to be gained, one could imagine a case where the inner-most array of values has only two values, one at index 0, and one at index 1,000,000. In this case, the dictionary approach would be more efficient compared to the list of lists approach. So, in this case, we have a list of lists of dictionaries of values, as opposed to a list-of-lists-of-lists-of-values. This is exactly the methodology of how Finch nests levels. Each one of the "list-of" levels corresponds to a "level" in Finch. This allows finch to represent sparse numerical tensors, while allowing each individual level to have a different format for efficiency.


!!! aside "Aside"
    _Technically_, for efficiency and usability, Finch allows multiple dimensions of the array to be represented by a single level. This slight deviation from "one-level-per-dimension" allows for Finch to execute much faster and have a much cleaner user-experience.

# Indexing is not enough.

Unfortunately, indexing is not quite enough to define all the array operations. Imagine something like pair-wise addition of two arrays (so the first element of Array A, with the first element of Array B, the second element of Array A, and the second element of B, and so on, and so forth). If the indexing is "dense", that is, every value of the array must participate in the addition, then iterating from 0 up to the length of the array and indexing both A and B is sufficient to implement addition. This is a sparse-tensor library, where that invariant notably does not hold. Worse, finch has no guarantee that it's even adding two arrays of the same format. Element-wise multiplication has the same issue, as do other matrix operations, like matrix multiplication.

So how do we solve this? Well, prior works solved this in a very limited way,
either by manually implementing the operations, which was never done for every combination of sparse array formats and operators. You can imagine, if you have fifteen operators and ten sparse formats, that's 1500 method implementations. Naturally, there are more than 15 operators, and there are more than ten sparse formats.

Finch can efficiently solve all of these problems with only a couple of definitions per level format, thanks to work done by Willow Marie Ahrens (who is a major contributor to Finch) in developing [Looplets](https://arxiv.org/abs/2209.05250).

## Compiler Lowering in Finch

Finch executes programs first by walking the tree of computations (`lazy`) and then computing values (`compute`). After the tree of computations is walked, prior to execution, finch analyzes the structure of the data and formats and automatically selects the best order of operations and format required to perform the user's calculation. While programs are written in imperative-fashion in finch, they are lowered to a more declarative format (called finch logic), which only describes which calculations are required. Finch notation is a thin-lowering over finch logic which translates finch logic to loops over dense formats (called looplets). Once finch notation is generated. Next, the compiler generates finch assembly, which a fully typed, simplistic imperative language designed to express fundamental control flow. Finch assembly is then lowered to a  variety of backends, with the C backend being the default and most capable, but interpreters for the format exist, and JIT compilation via [Numba](https://numba.pydata.org/) and [MLIR](https://mlir.llvm.org/). Finch's goal is to be a sparse tensor compiler that efficiently executes sparse tensor routines everywhere. While not as many platforms are supported as we would like, extending finch to also work on disk and on GPUs are active areas of research.



### What is a Looplet?

A Looplet is a data structure that describes iteration and seeking over a level. This is actually sufficient to implement the matrix operations the project concerns itself with. To people not familiar with efficient low-level math, the majority of the computation time is usually spent fetching values from memory. The average processor can do a _lot_ of multiplications and additions in the time it takes to fetch a value from memory.

## Schedulers

A scheduler determines the intermediate formats and order of operations that will performed in the target runtime. Currently, schedulers can be found in `finch.autoschedule`.
