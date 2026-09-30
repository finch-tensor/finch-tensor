import finch
from finch.finch_assembly import element_type
from finch.symbolic import Reflector


def test_public_exports_are_available_sorted_and_unique():
    assert all(hasattr(finch, attr) for attr in finch.__all__)
    assert sorted(set(finch.__all__)) == finch.__all__


def test_existing_public_exports_stay_available():
    assert finch.Reflector is Reflector
    assert finch.element_type is element_type


def test_star_import_exposes_public_names():
    namespace = {}
    exec("from finch import *", namespace)

    for name in finch.__all__:
        assert namespace[name] is getattr(finch, name)
