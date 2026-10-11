import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.cost_models import PigeonCostModel
from finch.autoschedule.tensor_stats import DCStatsFactory, UniformStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    Table,
)


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("kind", ["copy", "aggregate", "update"])
def test_pigeon_cost_model_query_forms(factory, shape, kind):
    sf = factory()
    model = PigeonCostModel()
    idxs = tuple(Field(f"i{n}") for n in range(len(shape)))
    a, out = Alias("a"), Alias("out")
    tensor = fl.asarray(np.ones(shape))
    stats = {a: sf(tensor, idxs)}
    bindings = {a: tensor.ftype}
    table = Table(a, idxs)
    if kind == "copy":
        query = Query(Table(out, tuple(reversed(idxs))), table)
    else:
        out_idxs = idxs[:-1]
        arg = Reorder(MapJoin(Literal(ffuncs.mul), (table, table)), idxs)
        if kind == "aggregate":
            query = Query(
                Table(out, out_idxs),
                Aggregate(Literal(ffuncs.add), Literal(0.0), arg, idxs[-1:]),
            )
        else:
            output = fl.asarray(np.zeros(shape[:-1]))
            stats[out] = sf(output, out_idxs)
            bindings[out] = output.ftype
            query = QueryInto(Table(out, out_idxs), Literal(ffuncs.add), arg)
    plan = Plan((query, Produces((out,))))
    original_stats = dict(stats)
    original_fills = {alias: stat.fill_value for alias, stat in stats.items()}

    model.validate_inputs(plan, bindings, stats, sf)
    features = model.get_features(plan, bindings, stats, sf)

    assert features.shape == model.coefficients.shape == (3,)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)
    assert model.predict_cost(plan, bindings, stats, sf) > 0
    if idxs:
        assert np.all(features > 0)
    else:
        np.testing.assert_array_equal(features[:2], [0, 0])
    assert stats == original_stats
    assert all(
        stat.fill_value is original_fills[alias] for alias, stat in stats.items()
    )


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_pigeon_cost_model_dependent_queries(factory):
    sf = factory()
    model = PigeonCostModel()
    i, j = Field("i"), Field("j")
    a, b, out = Alias("a"), Alias("b"), Alias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats = {a: sf(tensor, (i, j))}
    bindings = {a: tensor.ftype}
    queries = (
        Query(Table(b, (j, i)), Table(a, (i, j))),
        QueryInto(Table(b, (j, i)), Literal(ffuncs.overwrite), Literal(0.0)),
        QueryInto(
            Table(b, (j, i)),
            Literal(ffuncs.add),
            Reorder(Table(a, (i, j)), (j, i)),
        ),
        Query(
            Table(out, (j,)),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0.0),
                Reorder(Table(b, (j, i)), (j, i)),
                (i,),
            ),
        ),
    )
    plan = Plan((*queries, Produces((out,))))
    model.validate_inputs(plan, bindings, stats, sf)
    features = model.get_features(plan, bindings, stats, sf)

    intermediate_stats = dict(stats)
    machine = StatsMachine(sf, intermediate_stats)
    expected = np.zeros(3)
    for query in queries:
        expected += model.get_features(query, bindings, intermediate_stats, sf)
        machine(query)

    np.testing.assert_allclose(features, expected)
    assert model.predict_cost(plan, bindings, stats, sf) == pytest.approx(
        np.dot(model.coefficients, expected)
    )
    assert set(stats) == {a}
