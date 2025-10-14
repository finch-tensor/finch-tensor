from numpy.testing import assert_equal

import finch

base_path = "tests/data"


def test_read(arr2d):
    tns = finch.read(f"{base_path}/matrix_1.ttx")

    assert_equal(tns.todense(), arr2d)


def test_write(tmp_path, arr2d):
    tns = finch.asarray(arr2d)
    finch.write(tmp_path / "tmp.ttx", tns)

    with open(f"{base_path}/matrix_1.ttx") as f:
        expected = f.read()
    with open(tmp_path / "tmp.ttx") as f:
        actual = f.read()

    assert actual == expected
