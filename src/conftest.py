import pytest

import numpy as np

import finch


@pytest.fixture(autouse=True)
def add_modules(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["finch"] = finch
