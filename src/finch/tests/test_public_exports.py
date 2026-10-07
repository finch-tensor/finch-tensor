import finch


def test_public_exports_are_available_and_unique():
    assert all(hasattr(finch, attr) for attr in finch.__all__)
    assert len(set(finch.__all__)) == len(finch.__all__)
