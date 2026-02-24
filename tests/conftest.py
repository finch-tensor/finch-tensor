import pytest

import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def arr1d():
    return np.arange(100)


@pytest.fixture
def arr2d():
    return np.array(
        [
            [0, 0, 3, 2, 0],
            [1, 0, 0, 1, 0],
            [0, 5, 0, 0, 0],
        ]
    )


@pytest.fixture
def arr3d():
    return np.array(
        [
            [[0, 1, 0, 0], [1, 0, 0, 3]],
            [[4, 0, -1, 0], [2, 2, 0, 0]],
            [[0, 0, 0, 0], [1, 5, 0, 3]],
        ]
    )
