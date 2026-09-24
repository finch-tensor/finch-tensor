import finch


def test_star_import_exposes_public_names():
    namespace = {}
    exec("from finch import *", namespace)

    for name in finch.__all__:
        assert namespace[name] is getattr(finch, name)
