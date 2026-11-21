# Guide for Numpy Users

!!! warning
    This section relies on API functionality that is _not yet_ released in the form of [PR #763](https://github.com/finch-tensor/finch-tensor/pull/763). Until that PR, or an equivalent PR is merged in, these docs may not represent valid ways of using finch.

[Numpy](https://numpy.org/) is the most popular array programming library in Python. You are not expected to know Numpy to be able to use Finch; however, as many members of the community do use Numpy, this guide exists for them.

## What is a sparse array?

In efforts not to bore with tediously long definitions, we describe increasingly precise definitions of sparse arrays, to the extent that they are implemented by this package.

!!! info "Definition 1: An Array"
    An array is a contiguous, ordered sequence of values of finite length, which can be indexed and iterated over.
!!! info "Definition 2: Sparse Arrays (introduction)"
    A sparse array is an array where most values are zero. Some notable examples include arrays of the identity transform (I, aka `np.eye`) and triangular matrices.

!!! info "Definition 3: Sparse arrays _slightly_ more generally"
    A sparse array is a representation of an array when an appreciable number of entries in that array are zero.

!!! info "Definition 4: Sparse arrays. Zeros are not special."
    A sparse array, practically, is a useful method of representing an array when an appreciable number of entries is some-non-zero value, called the fill. A matrix which represents the pairwise distance of two groups may have unreachable entries represented using infinity as a value, for instance.

!!! info "Definition 5: Extending to sparse tensors"
    A sparse matrix is a sparse array of sparse arrays of values. A sparse tensor is a sparse array of sparse matrices. Finch allows this nesting pattern to continue, allowing for the construction of arbitrarily "nested" (high-dimensional) sparse tensors.

## Installation

While finch remains in active development, install it directly from GitHub:

=== "pip"

    ```{ .sh, .copy }
    pip install git+https://github.com/finch-tensor/finch-tensor.git
    ```

=== "uv"

    ```{ .sh, .copy }
    uv add git+https://github.com/finch-tensor/finch-tensor
    ```



## Sparse Arrays

The most used abstraction of Finch is the `FiberTensor`, as it abstracts over the various sparse array formats. While more detail will be provided on `FiberTensor` more generally, it is the most important format to concern yourself with at this point.

Because a FiberTensor allows for different formats of arrays to be stored, we can trivially construct one from a numpy matrix, in fully dense form. All numpy matrices are dense in their memory representation, as they literally store one value for every index in the array. To convert a numpy array to a FiberTensor, call the `from_numpy` function from the module `finch.interface`.

```py
# include the required libraries
import numpy as np
import finch

example_array_numpy = np.array([42, 45, 87])
example_array_finch = finch.from_numpy(example_array_numpy)
```


## Operations on Finch Tensors

Intuitively, the first thing that you _might_ try to do is _index_ into a sparse tensor. This is _not_ possible generally in Finch, but there are ways to make it work. It is not possible to read nor assign an entry in a Finch array at the current time. There is work being done to optimize some of these restrictions away. For now, for matrices of sufficient size, converting to a fully dense numpy array `finch.to_numpy` will restore indexing capability. Progress on indexing into fiber tensors is documented in [Issue #622](https://github.com/finch-tensor/finch-tensor/issues/622).


## Defining a truly sparse array

A single entry in a sparse array can be represented as a pair. The left hand side of the pair is a tuple of _n_ many entries, representing the index. The right hand side of the pair is value at that index. Repeating this process can describe all entries in a sparse array. The above construction can be transformed into a matrix out of the left-hand-side of the pairs and a vector out of the right-hand side the pairs. With these two pairs, it is now possible to construct a _truly_ sparse matrix using finch.


```py
import finch

indices = np.array([
    [1, 2],
    [4, 5],
    [8, 9]
])
values = [91, 38, 22]

array_10x10 = finch.from_parts_coo(indices, values, (10, 10))
```

This is called coordinate format, often shortened to the pseudo-acronym COO.

## Performing the supported operations.

While indexing is not supported yet, other general operations are, but first, we need to define what engine is going to power our operations, as not all engines are as featured as other backends. To enable the most powerful Finch engine, set:

```py
import finch
from finch.autoschedule import COMPILE_NUMBA_GALLEY


finch.autoschedule.set_default_backend(COMPILE_NUMBA_GALLEY)
```

## Performing simple operations.

Due to finch's flexibility, it's actually possible to perform operations on these matrices in an efficient manner using `lazy` and `compute`. In order to efficiently perform these operations, it's wise to wrap them in `lazy` and `compute` function calls.

```py
lazy_computation = finch.lazy(array_10x10) + finch.lazy(array_10x10)
realized_compute = finch.compute(lazy_computation)
```


`realized_compute` will now hold the result of adding `array_10x10` to itself, and it will do no extra work to add the empty slots in the array.

## The JIT

While the code above would suffice for trivial operations, most programs that people want to write in high level languages are not as trivial, and eliding the laziness and compute is desirable. To this end, finch has a procedure for automatically inserting these calls in the form of an `@jit` decorator.


```
@finch.jit
def add_with_self(a):
    return A + A

add_with_self(array_10x10)
```


This performs the same operation as above, but automatically inserts the lazy and compute calls, reducing the mental overhead associated with writing procedures in sparse array regimes.

## So why lazy and compute?

Optimization. Sparse array problem spaces lend themselves to certain optimizations that are not possible when arrays are dense. By building the program execution tree _first_ and then executing it, the finch library can determine dependencies and re-order computations to minimize the time and calculations required to compute a result.





## Leaving performance on the table: Transforming between formats.

While the construction of COO is quite obvious, the actual usage is inefficient on modern computing hardware. Therefore, a variety of other formats that are more efficient for performing operations have been proposed instead. One of the most popular ones is called compressed sparse row format, or CSR, for short.


When operating on truly huge arrays and challenging compute limited problems, other formats lend themselves better to computation when compared to COO. COO is one of the least efficient formats, but it is much simpler to understand compared to other formats.


Compressed Sparse Row (CSR) format can often reduce the space required to store sparse formats, due to lack of duplication in coordinates, and can also accelerate reads due to better memory access patterns. CSR format is composed of three arrays. The first array is the value array, which stores the non-fill (more loosely, the non-zero) entries of the array in a flat, dense form. The second array is the column indices array, when converting a dense array to CSR, the column indices array records the column of every value in the dense array. The third array is the row ptr array, but essentially, it represents an array so that `ptr[r]..ptr[r+1]` slices into the column indices array. The difference between `ptr[r+1]` and `ptr[r]` computes how many non-zero values there are in that row, meanwhile, `col_indices[ptr[r]..ptr[r+1]]` represents the columns of entries in that row. This structure expresses the behavior of having each row map to a list of tuples of column and value, but it is important to note that in memory, the CSR format is storing it as provided above, not as tuples directly, as it would be less efficient.


CSR, as it turns out, is actually quite decent for many operations as the amount of reads is minimized as the `r` is factored out from every entry in the data format. It's analogous format, compressed sparse column is the same format, except it stores the column indices in ptr and the row indices in row idx, and is more efficient for other operations.

To convert a a matrix from COO to CSR format, the asarray function exists:

```py
csr_10x10 = finch.asarray(matrix_10x10, format="csr")
```

For 10x10 matricies, the cost of converting between formats likely exceeds the computation gains, but for large matricies, formats like CSR are what underpine the huge calculations possible in the space.
