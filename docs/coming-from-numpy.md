# Guide for Numpy Users

!!! warning
    This section relies on API functionality that is _not yet_ released in the form of [PR #763](https://github.com/finch-tensor/finch-tensor/pull/763). Until that PR, or an equivalent PR is merged in, these docs may not represent valid ways of using finch.

[Numpy](https://numpy.org/) is the most popular array programming library in Python. You are not expected to know Numpy to be able to use Finch; however, as many members of the community do use Numpy, this guide exists for them.

## Sparse Tensor Definition

Sparse tensors are tensors of values where most of the entries are a single value. This value is most often zero, but can be other values such as infinity in the case of distance matrices and is called the fill value. Sparse tensors are useful to express as their representation can require signficantly less memory or storage compared to more traditional, contiguous tensor storage. They also can allow some operations to be performed much faster than traditional, dense array storage mechanisms, as sparse array compilers, such as finch, can avoiding performing calculations on the fill value.

## When are sparse tensors useful?

Sparse arrays are a restriction on dense arrays, so anything numerical result that can be computed on the sparse arrays could alternatively been computed on dense arrays. Sparse arrays are more structured than dense arrays and more limited in scope, but, in exchange, they can be significantly more performant. One classic example is Sampled Dense Dense Matrix Multiplication, written <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"> <mrow> <mi>A</mi> <mo>∘</mo> <mrow> <mo stretchy="true" form="prefix">(</mo> <mi>B</mi> <mi>C</mi> <mo stretchy="true" form="postfix">)</mo> </mrow></mrow></math>, with B and C dense and A sparse. In traditional dense regimes, B and C must be multiplied in their entirety and then must compute the element-wise product with A. An optimizing sparse compiler, such as finch, can analyze this calculation and only compute the entries that will be involved in the element-wise product with sparse matrix A. If the matrix A is highly sparse, this optimziation is a significant improvement compared to the dense calculation.

While sparse arrays can reduce the amount of work required when the problem is structured suitably, they do incur overhead in their storage layouts. Sparse array formats often require a representation of their indices to be stored alongside the array values, and they often have less optimal memory access patterns on modern hardware when compared to dense arrays. Finch is smart enough to recognize when the data is obviously too dense for sparse arrays to have performance advantages and will fall back to using dense arrays when necessary. However, in general, dense arrays are performant enough for most use cases, and sticking with the simplicity of dense arrays when possible is often a pragmatic choice. For problems that cannot be computed densely, finch provides a state of the art evaluation engine which automatically optimizes sparse tensor programs.

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


```py
@finch.jit
def add_with_self(a):
    return A + A

add_with_self(array_10x10)
```




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

## Optimized Formats.

As opposed to the prior arts in sparse tensor computation within Python, finch is not limited to only handling 2D sparse arrays, and can instead nest sparse arrays to form sparse tensors. This restriction even allows for the mixing of the different sparse array formats to describe one array. The top level format (the first index) might be a sparse list format, but it might hold a sparse hash level, for instance.
