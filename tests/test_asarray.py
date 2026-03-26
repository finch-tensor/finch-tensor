"""Tests for the asarray function."""

import pytest
import numpy as np
from finch import asarray
from finch.tensor import FinchJLTensor

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestAsarrayNumpy:
    """Test asarray with numpy arrays."""

    def test_asarray_1d_array(self):
        """Test converting a 1D numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_2d_array(self):
        """Test converting a 2D numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_3d_array(self):
        """Test converting a 3D numpy array."""
        arr = np.arange(24).reshape(2, 3, 4).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_fortran_order(self):
        """Test converting a Fortran-order array."""
        arr = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_with_fill_value(self):
        """Test asarray with explicit fill_value."""
        arr = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = asarray(arr, fill_value=0.0)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_with_dtype(self):
        """Test asarray with explicit dtype."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = asarray(arr, dtype=np.int32)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_default_fill_value(self):
        """Test that default fill_value is 0.0."""
        arr = np.array([1.0, 2.0, 3.0])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_large_array(self):
        """Test converting a large numpy array."""
        arr = np.arange(1000).reshape(10, 100).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_zero_array(self):
        """Test converting an all-zero array."""
        arr = np.zeros((5, 5))
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_ones_array(self):
        """Test converting an all-ones array."""
        arr = np.ones((5, 5))
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)


class TestAsarrayFinchTensor:
    """Test asarray with FinchJLTensor inputs."""

    def test_asarray_finch_tensor_no_copy(self):
        """Test asarray on FinchJLTensor returns same object when copy=False."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor1 = asarray(arr)
        tensor2 = asarray(tensor1, copy=False)
        assert tensor1 is tensor2

    def test_asarray_finch_tensor_with_copy(self):
        """Test asarray on FinchJLTensor with copy=True."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor1 = asarray(arr)
        tensor2 = asarray(tensor1, copy=True)
        # Should be different objects
        assert tensor1 is not tensor2
        assert isinstance(tensor2, FinchJLTensor)

    def test_asarray_finch_tensor_default_copy(self):
        """Test asarray on FinchJLTensor with default copy behavior."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor1 = asarray(arr)
        tensor2 = asarray(tensor1)
        # Default should return same object (copy=None defaults to no copy)
        assert tensor1 is tensor2


class TestAsarrayEdgeCases:
    """Test asarray edge cases and special values."""

    def test_asarray_single_element(self):
        """Test converting single element array."""
        arr = np.array([5.0])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_negative_values(self):
        """Test converting array with negative values."""
        arr = np.array([[-1.0, -2.0], [3.0, 4.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_mixed_values(self):
        """Test converting array with mixed positive and negative values."""
        arr = np.array([[-5.0, 0.0, 5.0], [1.0, -1.0, 2.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_very_small_values(self):
        """Test converting array with very small values."""
        arr = np.array([[1e-10, 1e-15], [1e-20, 1e-25]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_very_large_values(self):
        """Test converting array with very large values."""
        arr = np.array([[1e10, 1e15], [1e20, 1e25]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_nan_values(self):
        """Test converting array with NaN values."""
        arr = np.array([[1.0, np.nan], [np.nan, 4.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_inf_values(self):
        """Test converting array with infinity values."""
        arr = np.array([[1.0, np.inf], [-np.inf, 4.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)


class TestAsarrayDataTypes:
    """Test asarray with different data types."""

    def test_asarray_float32(self):
        """Test converting float32 array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_float64(self):
        """Test converting float64 array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_int32(self):
        """Test converting int32 array."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_int64(self):
        """Test converting int64 array."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_complex64(self):
        """Test converting complex64 array."""
        arr = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_complex128(self):
        """Test converting complex128 array."""
        arr = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)


class TestAsarrayShapes:
    """Test asarray with various array shapes."""

    def test_asarray_row_vector(self):
        """Test converting row vector."""
        arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_column_vector(self):
        """Test converting column vector."""
        arr = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_square_matrix(self):
        """Test converting square matrix."""
        arr = np.arange(25).reshape(5, 5).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_rectangular_matrix(self):
        """Test converting rectangular matrix."""
        arr = np.arange(20).reshape(4, 5).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_tall_matrix(self):
        """Test converting tall matrix."""
        arr = np.arange(20).reshape(10, 2).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_wide_matrix(self):
        """Test converting wide matrix."""
        arr = np.arange(20).reshape(2, 10).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_4d_array(self):
        """Test converting 4D array."""
        arr = np.arange(120).reshape(2, 3, 4, 5).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_5d_array(self):
        """Test converting 5D array."""
        arr = np.arange(120).reshape(2, 3, 4, 5, 1).astype(float)
        result = asarray(arr)
        assert isinstance(result, FinchJLTensor)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestAsarrayScipy:
    """Test asarray with scipy.sparse matrices."""

    def test_asarray_scipy_csc(self):
        """Test converting scipy CSC sparse matrix."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        row = np.array([0, 1, 0, 2])
        col = np.array([0, 1, 1, 2])
        csc_matrix = sp.csr_matrix((data, (row, col)), shape=(3, 3)).tocsc()
        result = asarray(csc_matrix)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_scipy_coo(self):
        """Test converting scipy COO sparse matrix."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        row = np.array([0, 1, 0, 2])
        col = np.array([0, 1, 1, 2])
        coo_matrix = sp.coo_matrix((data, (row, col)), shape=(3, 3))
        result = asarray(coo_matrix)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_scipy_dense_csc(self):
        """Test converting dense scipy matrix in CSC format."""
        arr = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        csc_matrix = sp.csc_matrix(arr)
        result = asarray(csc_matrix)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_scipy_with_fill_value(self):
        """Test asarray on scipy matrix with fill_value."""
        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        csc_matrix = sp.csr_matrix((data, (row, col)), shape=(3, 3)).tocsc()
        result = asarray(csc_matrix, fill_value=0.0)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_scipy_with_copy_true(self):
        """Test asarray on scipy matrix with copy=True."""
        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        csc_matrix = sp.csr_matrix((data, (row, col)), shape=(3, 3)).tocsc()
        result = asarray(csc_matrix, copy=True)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_scipy_sorted_indices(self):
        """Test asarray on scipy matrix with sorted indices."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        row = np.array([0, 1, 0, 2])
        col = np.array([0, 1, 1, 2])
        coo_matrix = sp.coo_matrix((data, (row, col)), shape=(3, 3))
        csc_matrix = coo_matrix.tocsc()
        csc_matrix.sort_indices()
        result = asarray(csc_matrix)
        assert isinstance(result, FinchJLTensor)


class TestAsarrayErrors:
    """Test asarray error handling."""

    def test_asarray_invalid_type(self):
        """Test asarray with unsupported type."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            asarray("invalid string input")

    def test_asarray_invalid_list(self):
        """Test asarray with plain Python list (should fail)."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            asarray([1, 2, 3])

    def test_asarray_dict_input(self):
        """Test asarray with dict input."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            asarray({"a": 1})

    def test_asarray_none_input(self):
        """Test asarray with None input."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            asarray(None)


class TestAsarrayOptions:
    """Test asarray option combinations."""

    def test_asarray_copy_none(self):
        """Test asarray with copy=None."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = asarray(arr, copy=None)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_all_options(self):
        """Test asarray with all options specified."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = asarray(arr, dtype=np.float64, fill_value=0.0, copy=True)
        assert isinstance(result, FinchJLTensor)

    def test_asarray_numpy_no_copy(self):
        """Test asarray on numpy with copy=False."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = asarray(arr, copy=False)
        assert isinstance(result, FinchJLTensor)
