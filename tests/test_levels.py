"""Tests for the Finch levels module."""

import pytest
import numpy as np
from finch.levels import (
    ElementLevel,
    DenseLevel,
    PatternLevel,
    SparseListLevel,
    SparseByteMapLevel,
    RepeatRLELevel,
    SparseVBLLevel,
    SparseCOOLevel,
    SparseHashLevel,
)
from finch.buffer import NumpyBuffer


class TestElement:
    """Test Element level construction."""

    def test_element_creation_basic(self):
        """Test creating an Element level with a fill value."""
        elem = ElementLevel(0.0)
        assert elem._obj is not None

    def test_element_creation_with_int_fill(self):
        """Test creating an Element level with integer fill value."""
        elem = ElementLevel(0)
        assert elem._obj is not None

    def test_element_creation_with_float_fill(self):
        """Test creating an Element level with float fill value."""
        elem = ElementLevel(3.14)
        assert elem._obj is not None


class TestDense:
    """Test Dense level construction."""

    def test_dense_creation_basic(self):
        """Test creating a Dense level."""
        elem = ElementLevel(0.0)
        dense = DenseLevel(elem)
        assert dense._obj is not None

    def test_dense_creation_with_shape(self):
        """Test creating a Dense level with explicit shape."""
        elem = ElementLevel(0.0)
        dense = DenseLevel(elem, shape=10)
        assert dense._obj is not None

    def test_dense_nesting(self):
        """Test creating nested Dense levels."""
        elem = ElementLevel(0.0)
        dense1 = DenseLevel(elem)
        dense2 = DenseLevel(dense1)
        assert dense2._obj is not None


class TestPattern:
    """Test Pattern level construction."""

    def test_pattern_creation(self):
        """Test creating a Pattern level."""
        pattern = PatternLevel()
        assert pattern._obj is not None


class TestSparseList:
    """Test SparseList level construction."""

    def test_sparselist_creation_basic(self):
        """Test creating a SparseList level."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem)
        assert sparse._obj is not None

    def test_sparselist_creation_with_dim(self):
        """Test creating a SparseList level with explicit dimension."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem, dim=10)
        assert sparse._obj is not None

    def test_sparselist_creation_with_data_arrays(self):
        """Test creating a SparseList level with pointer and index arrays."""
        elem = ElementLevel(0.0)
        ptr = NumpyBuffer(np.array([0, 2, 2, 4], dtype=np.int32))
        idx = NumpyBuffer(np.array([1, 2, 1, 3], dtype=np.int32))
        sparse = SparseListLevel(elem, ptr=ptr, idx=idx)
        assert sparse._obj is not None

    def test_sparselist_creation_with_data_lists(self):
        """Test creating a SparseList level with pointer and index as lists."""
        elem = ElementLevel(0.0)
        ptr = [0, 2, 2, 4]
        idx = [1, 2, 1, 3]
        sparse = SparseListLevel(elem, ptr=ptr, idx=idx)
        assert sparse._obj is not None

    def test_sparselist_ptr_property(self):
        """Test accessing ptr property of SparseList."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem)
        # Property should be accessible
        ptr_buffer = sparse.ptr
        assert ptr_buffer is not None

    def test_sparselist_idx_property(self):
        """Test accessing idx property of SparseList."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem)
        # Property should be accessible
        idx_buffer = sparse.idx
        assert idx_buffer is not None


class TestSparseByteMap:
    """Test SparseByteMap level construction."""

    def test_sparsebytemap_creation_basic(self):
        """Test creating a SparseByteMap level."""
        elem = ElementLevel(0.0)
        sparse = SparseByteMapLevel(elem)
        assert sparse._obj is not None

    def test_sparsebytemap_creation_with_dim(self):
        """Test creating a SparseByteMap level with explicit dimension."""
        elem = ElementLevel(0.0)
        sparse = SparseByteMapLevel(elem, dim=10)
        assert sparse._obj is not None


class TestRepeatRLE:
    """Test RepeatRLE level construction."""

    def test_repeatrle_creation_basic(self):
        """Test creating a RepeatRLE level."""
        elem = ElementLevel(0.0)
        rle = RepeatRLELevel(elem)
        assert rle._obj is not None

    def test_repeatrle_creation_with_dim(self):
        """Test creating a RepeatRLE level with explicit dimension."""
        elem = ElementLevel(0.0)
        rle = RepeatRLELevel(elem, dim=10)
        assert rle._obj is not None


class TestSparseVBL:
    """Test SparseVBL level construction."""

    def test_sparsevbl_creation_basic(self):
        """Test creating a SparseVBL level."""
        elem = ElementLevel(0.0)
        vbl = SparseVBLLevel(elem)
        assert vbl._obj is not None

    def test_sparsevbl_creation_with_dim(self):
        """Test creating a SparseVBL level with explicit dimension."""
        elem = ElementLevel(0.0)
        vbl = SparseVBLLevel(elem, dim=10)
        assert vbl._obj is not None


class TestSparseCOO:
    """Test SparseCOO level construction."""

    def test_sparsecoo_creation_basic(self):
        """Test creating a SparseCOO level."""
        elem = ElementLevel(0.0)
        coo = SparseCOOLevel(2, elem)
        assert coo._obj is not None

    def test_sparsecoo_creation_with_dims(self):
        """Test creating a SparseCOO level with explicit dimensions."""
        elem = ElementLevel(0.0)
        coo = SparseCOOLevel(2, elem, dims=(4, 3))
        assert coo._obj is not None

    def test_sparsecoo_creation_with_dims_list(self):
        """Test creating a SparseCOO level with dimensions as list."""
        elem = ElementLevel(0.0)
        coo = SparseCOOLevel(2, elem, dims=[4, 3])
        assert coo._obj is not None

    def test_sparsecoo_creation_with_coordinate_arrays(self):
        """Test creating a SparseCOO level with coordinate arrays."""
        elem = ElementLevel(0.0)
        i_coords = NumpyBuffer(np.array([0, 1, 2, 3], dtype=np.int32))
        j_coords = NumpyBuffer(np.array([0, 0, 2, 2], dtype=np.int32))
        coo = SparseCOOLevel(2, elem, tbl=(i_coords, j_coords))
        assert coo._obj is not None

    def test_sparsecoo_creation_with_coordinate_lists(self):
        """Test creating a SparseCOO level with coordinate arrays as lists."""
        elem = ElementLevel(0.0)
        i_coords = [0, 1, 2, 3]
        j_coords = [0, 0, 2, 2]
        coo = SparseCOOLevel(2, elem, tbl=(i_coords, j_coords))
        assert coo._obj is not None

    def test_sparsecoo_3d(self):
        """Test creating a 3D SparseCOO level."""
        elem = ElementLevel(0.0)
        coo = SparseCOOLevel(3, elem, dims=(5, 4, 3))
        assert coo._obj is not None

    def test_sparsecoo_tbl_property(self):
        """Test accessing tbl property of SparseCOO."""
        elem = ElementLevel(0.0)
        coo = SparseCOOLevel(2, elem)
        # Property should be accessible
        tbl = coo.tbl
        assert tbl is not None
        assert isinstance(tbl, tuple)


class TestSparseHash:
    """Test SparseHash level construction."""

    def test_sparsehash_creation_basic(self):
        """Test creating a SparseHash level."""
        elem = ElementLevel(0.0)
        hash_level = SparseHashLevel(2, elem)
        assert hash_level._obj is not None

    def test_sparsehash_creation_with_dims(self):
        """Test creating a SparseHash level with explicit dimensions."""
        elem = ElementLevel(0.0)
        hash_level = SparseHashLevel(2, elem, dims=(4, 3))
        assert hash_level._obj is not None

    def test_sparsehash_creation_with_dims_list(self):
        """Test creating a SparseHash level with dimensions as list."""
        elem = ElementLevel(0.0)
        hash_level = SparseHashLevel(2, elem, dims=[4, 3])
        assert hash_level._obj is not None

    def test_sparsehash_3d(self):
        """Test creating a 3D SparseHash level."""
        elem = ElementLevel(0.0)
        hash_level = SparseHashLevel(3, elem, dims=(5, 4, 3))
        assert hash_level._obj is not None


class TestComposedLevels:
    """Test composed level hierarchies."""

    def test_csc_matrix_format(self):
        """Test creating CSC matrix format (Dense(SparseList(Element)))."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem)
        dense = DenseLevel(sparse)
        assert dense._obj is not None

    def test_csr_like_format(self):
        """Test creating CSR-like format (SparseList(Dense(Element)))."""
        elem = ElementLevel(0.0)
        dense = DenseLevel(elem)
        sparse = SparseListLevel(dense)
        assert sparse._obj is not None

    def test_dcsc_format(self):
        """Test creating DCSC format (SparseList(SparseList(Element)))."""
        elem = ElementLevel(0.0)
        sparse1 = SparseListLevel(elem)
        sparse2 = SparseListLevel(sparse1)
        assert sparse2._obj is not None

    def test_deep_nesting(self):
        """Test deeply nested levels."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem)
        dense = DenseLevel(sparse)
        sparse2 = SparseListLevel(dense)
        dense2 = DenseLevel(sparse2)
        assert dense2._obj is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_element_with_negative_fill(self):
        """Test Element with negative fill value."""
        elem = ElementLevel(-1.0)
        assert elem._obj is not None

    def test_sparselist_only_ptr_no_idx(self):
        """Test SparseList with ptr but no idx (should not add both)."""
        elem = ElementLevel(0.0)
        ptr = NumpyBuffer(np.array([0, 2, 2], dtype=np.int32))
        sparse = SparseListLevel(elem, ptr=ptr)
        assert sparse._obj is not None

    def test_sparselist_only_idx_no_ptr(self):
        """Test SparseList with idx but no ptr (should not add both)."""
        elem = ElementLevel(0.0)
        idx = NumpyBuffer(np.array([1, 2], dtype=np.int32))
        sparse = SparseListLevel(elem, idx=idx)
        assert sparse._obj is not None

    def test_sparsecoo_single_coordinate(self):
        """Test SparseCOO with single coordinate."""
        elem = ElementLevel(0.0)
        coords = NumpyBuffer(np.array([0], dtype=np.int32))
        coo = SparseCOOLevel(1, elem, tbl=(coords,))
        assert coo._obj is not None

    def test_large_dimension(self):
        """Test levels with large dimensions."""
        elem = ElementLevel(0.0)
        sparse = SparseListLevel(elem, dim=1000000)
        assert sparse._obj is not None


class TestArrayConversion:
    """Test that array arguments are properly converted."""

    def test_sparselist_converts_lists_to_arrays(self):
        """Test that SparseList converts list arguments to arrays."""
        elem = ElementLevel(0.0)
        ptr = NumpyBuffer(np.array([0, 2, 4], dtype=np.int32))
        idx = NumpyBuffer(np.array([1, 2, 3], dtype=np.int32))
        sparse = SparseListLevel(elem, ptr=ptr, idx=idx)
        # Should not raise an error during creation
        assert sparse._obj is not None

    def test_sparsecoo_converts_lists_to_arrays(self):
        """Test that SparseCOO converts list arguments to arrays."""
        elem = ElementLevel(0.0)
        coords_list = (
            NumpyBuffer(np.array([0, 1, 2], dtype=np.int32)),
            NumpyBuffer(np.array([0, 1, 2], dtype=np.int32))
        )
        coo = SparseCOOLevel(2, elem, tbl=coords_list)
        # Should not raise an error during creation
        assert coo._obj is not None

    def test_different_dtype_arrays(self):
        """Test that different dtype arrays are handled."""
        elem = ElementLevel(0.0)
        ptr = NumpyBuffer(np.array([0, 2, 4], dtype=np.int64))
        idx = NumpyBuffer(np.array([1, 2, 3], dtype=np.int32))
        sparse = SparseListLevel(elem, ptr=ptr, idx=idx)
        assert sparse._obj is not None
