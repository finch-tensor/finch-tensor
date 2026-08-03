from collections import OrderedDict

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.loop_order_cost import loop_order_cost
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.finch_logic import Alias, Field, Literal, MapJoin, Table


def test_empty_input_ordered_first_is_cheaper():
    sf = DCStatsFactory()
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    bindings = {
        a: sf(fl.asarray(np.ones((4, 4))), (i, j)),
        b: sf(fl.asarray(np.zeros((4, 4))), (j, k)),
    }

    dense_first = loop_order_cost(expr, (i, j, k), sf, bindings)
    empty_first = loop_order_cost(expr, (j, k, i), sf, bindings)

    assert empty_first < dense_first


def test_empty_relation():
    sf = DCStatsFactory()
    # l_ is l, precommit throws bad name error otehrwise
    i, j, k, l_, m = (Field(name) for name in "ijklm")
    a, b, c, d = (Alias(name) for name in "ABCD")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(a, (i, j)),
            Table(b, (j, k)),
            Table(c, (k, l_)),
            Table(d, (l_, m)),
        ),
    )
    bindings = OrderedDict(
        {
            a: sf(fl.asarray(np.ones((2, 2))), (i, j)),
            b: sf(fl.asarray(np.ones((2, 2))), (j, k)),
            c: sf(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: sf(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    forward = loop_order_cost(expr, (i, j, k, l_, m), sf, bindings)
    reverse = loop_order_cost(expr, (m, l_, k, j, i), sf, bindings)
    assert forward > reverse
