import finch


def test_public_exports_are_available_sorted_and_unique():
    assert all(hasattr(finch, attr) for attr in finch.__all__)
    assert sorted(set(finch.__all__)) == finch.__all__
