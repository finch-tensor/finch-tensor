"""
The pipeline which fits the cost models of the physical optimizers, from
scripts/fitting, on a Gustavson matmul of 4096 by 4096 inputs with a few
entries in each row, whose output is sparse enough that a workspace for each
row costs less than scattering into it.
"""

import itertools
import json
import math

import pytest

import numpy as np
import scipy.sparse as sps

from finch import ffuncs
from finch.autoschedule.cost_models import (
    Affine,
    Concat,
    DenseLevelOption,
    Stratify,
    WithCopies,
    WriteClasses,
    write_class,
)
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.compile_jl.julia import julia_available
from finch.finch_logic import (
    Aggregate,
    Field,
    FusedAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
    parse_logic,
)
from finch.symbolic import PostOrderDFS
from scripts.fitting import physical_bases
from scripts.fitting.generation import (
    BLOCK_BITS,
    input_record,
    materialize,
    path,
    record_inputs,
)
from scripts.fitting.physical_bases import (
    BASES,
    TUNING_BASIS,
    A,
    B,
    C,
    factorized_queries,
    featurize_record,
    format_kinds,
    input_perms,
    loop_basis,
    parse_models,
    resolve,
    stratified,
)

i, j, k = Field("i"), Field("j"), Field("k")
LG = 12


def gustavson_record(rng, lg=LG, per_row=4):
    """
    The record of a loop stage example of C_ik = sum_j A_ij * B_jk, with A and
    B of shape 2 ** lg and about per_row entries in each row, whose variants
    are every loop order.
    """
    query = Query(
        Table(C, (i, k)),
        Aggregate(
            Literal(ffuncs.add),
            Literal(0.0),
            MapJoin(Literal(ffuncs.mul), (Table(A, (i, j)), Table(B, (j, k)))),
            (j,),
        ),
    )
    S = np.ones((2, 2)) / 4 * per_row / 2**lg
    return {
        "stage": "loop",
        "query": str(Plan((query, Produces((C,))))),
        "loop_order": ["i", "j", "k"],
        "exhaustive": True,
        "variants": [
            {"loop_order": [idx.name for idx in order]}
            for order in itertools.permutations((i, j, k))
        ],
        "dtype": "float64",
        "idx_type": "int64",
        "block_bits": BLOCK_BITS,
        "inputs": {tns.name: input_record(S, [lg, lg], rng) for tns in (A, B)},
    }


def workspaces(plan):
    return {node for node in PostOrderDFS(plan) if isinstance(node, FusedAlias)}


def test_resolve_stages_a_workspace_where_it_pays():
    record = gustavson_record(np.random.default_rng(0))
    inputs = record_inputs(record)
    models = parse_models(None)

    # Gustavson scatters into the columns of each row of C, which a workspace
    # of one row takes, so the copy out of it writes C in order.
    plan, formats = resolve(record, {"loop_order": ["i", "j", "k"]}, inputs, models)
    (ws,) = workspaces(plan)
    assert ws.nfused == 1
    assert set(formats) == {A, B, C, ws.unfused}
    assert len(formats[ws.unfused]) == 1
    assert formats[C][1].kind == "sparse_list"

    # An inner product writes C in order, so it needs no workspace.
    plan, formats = resolve(record, {"loop_order": ["i", "k", "j"]}, inputs, models)
    assert not workspaces(plan)
    assert set(formats) == {A, B, C}


def test_loop_features_charge_writes_before_the_workspacer_runs():
    record = gustavson_record(np.random.default_rng(0))
    models = parse_models(None)
    rows = featurize_record(record, record_inputs(record), "loop", models)
    basis = BASES["loop"]
    names = basis.feature_names

    def summed(variant_rows):
        total = dict(zip(names, np.asarray(variant_rows).sum(axis=0), strict=True))
        # A workspace could hold the region the writes of the first class
        # scatter into.
        return {
            "sequential_writes": total["class0.sequential_writes"]
            + total["class1.sequential_writes"],
            "workspace_writes": total["class0.scatter_writes"],
            "scatter_writes": total["class1.scatter_writes"],
            "lookups": total["lookups"],
        }

    totals = {
        tuple(variant["loop_order"]): summed(variant_rows)
        for variant, variant_rows in zip(record["variants"], rows, strict=True)
    }
    assert all(np.asarray(r).shape[1] == basis.n_features for r in rows)
    # The loop orderer prices each order before the workspacer runs, so the
    # scattered writes of Gustavson count, as those a small workspace takes.
    assert totals[("i", "j", "k")]["workspace_writes"] > 0
    assert totals[("i", "j", "k")]["scatter_writes"] == 0
    assert totals[("i", "k", "j")]["workspace_writes"] == 0
    assert totals[("i", "k", "j")]["scatter_writes"] == 0
    assert totals[("i", "k", "j")]["sequential_writes"] > 0
    # An outer product scatters into all of C, which no workspace fits.
    assert totals[("j", "i", "k")]["scatter_writes"] > 0
    # Every input gallops, whatever its format, so the loops look each up at
    # every point they visit.
    assert all(total["lookups"] > 0 for total in totals.values())


def test_loop_features_need_no_formats(monkeypatch):
    record = gustavson_record(np.random.default_rng(0), lg=8)
    inputs = record_inputs(record)

    def resolve(*args, **kwargs):
        raise AssertionError("The loop stage chose formats or protocols")

    monkeypatch.setattr(physical_bases, "resolve", resolve)
    rows = featurize_record(record, inputs, "loop", parse_models(None))
    assert len(rows) == len(record["variants"])


@pytest.mark.skipif(not julia_available(), reason="Finch.jl is not installed")
def test_timing_runs_the_resolved_programs(tmp_path):
    from scripts.fitting.time_physical_data import (
        FixedFormatter,
        Timer,
        julia_loader,
        reformat,
    )

    record = gustavson_record(np.random.default_rng(0))
    record["variants"] = [
        {"loop_order": ["i", "j", "k"]},
        {"loop_order": ["i", "k", "j"]},
    ]
    (tmp_path / "data").mkdir()
    path(tmp_path, "0000-00000", "program.json").write_text(json.dumps(record))
    runtimes = Timer(parse_models(None), repeat=1).time_example(
        tmp_path, "0000-00000", math.inf
    )
    assert set(runtimes) == {"0", "1"}
    assert all(runtime > 0 for runtime in runtimes.values())
    timing = json.loads(path(tmp_path, "0000-00000", "timing.json").read_text())
    assert timing["runtimes"] == runtimes

    # The variant with the workspace computes the product.
    inputs = record_inputs(record)
    materialized = {tns: materialize(tensor) for tns, tensor in inputs.items()}
    plan, formats = resolve(record, record["variants"][0], inputs, parse_models(None))
    assert workspaces(plan)
    # The timing saves the formats of every tensor each variant ran with.
    assert set(timing["formats"]) == {"0", "1"}
    assert timing["formats"]["0"] == format_kinds(formats)
    perms = input_perms(record, plan)
    bindings = {
        tns: reformat(tensor, formats[tns], perms[tns])
        for tns, tensor in materialized.items()
    }
    executor = LogicExecutor(
        FixedFormatter(julia_loader(), formats), stats_factory=DCStatsFactory()
    )
    (result,) = executor(plan, bindings)
    # materialize draws the values of a tensor from the seed 0.
    sparse = {
        tns: sps.coo_array(
            (1 - np.random.default_rng(0).random(tensor.nnz), tensor.coords()),
            shape=tensor.shape,
        )
        for tns, tensor in inputs.items()
    }
    expected = (sparse[A] @ sparse[B]).toarray()
    dense = reformat(result, [DenseLevelOption(), DenseLevelOption()])
    np.testing.assert_allclose(dense.to_numpy(), expected)


def test_stratified_rows_are_those_of_the_loop_basis_under_any_cutoff():
    record = gustavson_record(np.random.default_rng(0), lg=8)
    inputs = record_inputs(record)
    models = parse_models(None)
    raw = featurize_record(record, inputs, "loop", models, TUNING_BASIS)
    for classes in (write_class, WriteClasses(10.0)):
        rows = featurize_record(record, inputs, "loop", models, loop_basis(classes))
        for raw_rows, variant_rows in zip(raw, rows, strict=True):
            np.testing.assert_array_equal(stratified(raw_rows, classes), variant_rows)


def test_tuning_finds_the_region_past_which_scattered_writes_cost_more():
    from scripts.fitting.fit_physical_data import tune_cutoff

    # Each example scatters its writes into a region of some size, or writes in
    # order at a cost of 20 iterations, and scattered writes cost 50 times
    # more past 1e4 entries, where the regions skip from 5e3 to 2e4.
    rng = np.random.default_rng(0)
    raw, runtimes, examples = [], [], []
    for example in range(60):
        region = rng.choice([1e3, 2e3, 5e3, 2e4, 1e5, 1e6])
        # The columns are those of TUNING_BASIS: five of the Pigeon basis, four
        # of the write basis, and a constant.
        scattered = np.zeros((1, 10))
        scattered[0, 6:] = [1.0, region, 1.0, 1.0]
        in_order = np.zeros((1, 10))
        in_order[0, [0, 9]] = [20.0, 1.0]
        for index, (rows, runtime) in enumerate(
            [(scattered, 1.0 if region <= 1e4 else 50.0), (in_order, 20.0)]
        ):
            raw.append(rows)
            runtimes.append(runtime)
            examples.append((f"{example}", index))
    classes = tune_cutoff(raw, runtimes, examples, seed=0)
    assert 5e3 <= classes.region < 2e4


def test_models_with_a_tuned_cutoff_class_writes_by_it(tmp_path):
    classes = WriteClasses(3e3)
    basis = loop_basis(classes)
    path = tmp_path / "loop.model.json"
    saved = {
        "feature_names": list(basis.feature_names),
        "coefficients": list(np.arange(basis.n_features, dtype=float)),
        "region_cutoff": 3e3,
    }
    path.write_text(json.dumps(saved))
    model = parse_models([f"loop={path}"])["loop"]
    match model.basis:
        case WithCopies(
            basis=Affine(basis=Concat(bases=(_, Stratify(classify=classify))))
        ):
            assert classify == classes
        case other:
            raise AssertionError(f"Expected a stratified loop basis, got {other}")


def chain_record(rng, lg=8, per_row=4):
    """
    The record of a factorize stage example of E_il = sum_jk A_ij B_jk C_kl,
    with inputs of shape 2 ** lg and about per_row entries in each row, whose
    variants are its two factorizations, (A B) C and A (B C).
    """
    from scripts.fitting.generate_factorize_data import E, factorizations
    from scripts.fitting.physical_bases import input_stats, record_stats_factory

    last = Field("l")
    fields = {A: (i, j), B: (j, k), C: (k, last)}
    prgm = Plan(
        (
            Query(
                Table(E, (i, last)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0.0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        tuple(Table(tns, idxs) for tns, idxs in fields.items()),
                    ),
                    (j, k),
                ),
            ),
            Produces((E,)),
        )
    )
    S = np.ones((2, 2)) / 4 * per_row / 2**lg
    record = {
        "stage": "factorize",
        "query": str(prgm),
        "exhaustive": True,
        "dtype": "float64",
        "idx_type": "int64",
        "block_bits": BLOCK_BITS,
        "inputs": {tns.name: input_record(S, [lg, lg], rng) for tns in fields},
    }
    stats_factory = record_stats_factory(record)
    stats = input_stats(prgm, record_inputs(record), stats_factory)
    plans = factorizations(prgm, stats, stats_factory)
    record["variants"] = [{"plan": str(plan)} for plan in plans]
    return record


def test_factorizations_of_a_chain_are_its_two_products():
    record = chain_record(np.random.default_rng(0))
    # Each reduces one product into T1, then the other into T2, which E copies.
    assert [variant["plan"].splitlines()[0] for variant in record["variants"]] == [
        "T1[i, k] = Aggregate(add, 0.0, MapJoin(mul, A[i, j], B[j, k]), j)",
        "T1[j, l] = Aggregate(add, 0.0, MapJoin(mul, B[j, k], C[k, l]), k)",
    ]


def test_factorized_queries_read_the_exact_supports_of_earlier_ones():
    record = chain_record(np.random.default_rng(0))
    inputs = record_inputs(record)
    models = parse_models(None)
    for variant in record["variants"]:
        queries = list(factorized_queries(record, variant, inputs, models))
        # The copy of T2 into E happens outside the timer, so isn't a query.
        assert [query["query"].splitlines()[0][:2] for query, _ in queries] == [
            "T1",
            "T2",
        ]
        (first, first_tensors), (second, second_tensors) = queries
        assert set(first_tensors) <= set(inputs)
        assert all(first_tensors[tns] is inputs[tns] for tns in first_tensors)
        # The second reads T1, drawn from its support, which is that of the
        # product of the two inputs the first reads.
        (t1,) = set(second_tensors) - set(inputs)
        a, b = (first_tensors[tns] for tns in sorted(first_tensors, key=str))
        product = sps.coo_array(
            (np.ones(a.nnz), a.coords()), shape=a.shape
        ) @ sps.coo_array((np.ones(b.nnz), b.coords()), shape=b.shape)
        assert second_tensors[t1].nnz == product.nnz
        # Each query loops over all of its fields.
        for query, _ in queries:
            match parse_logic(query["query"]):
                case Plan((Query(Table(_, idxs), Aggregate(arg=arg)), _)):
                    order = [Field(name) for name in query["loop_order"]]
                    assert sorted(order, key=str) == sorted(
                        {*idxs, *arg.fields()}, key=str
                    )
                case other:
                    raise AssertionError(f"Expected one aggregate query, got {other}")


def test_factorize_features_price_each_query_from_the_inputs_alone(monkeypatch):
    record = chain_record(np.random.default_rng(0))

    def resolve(*args, **kwargs):
        raise AssertionError("The factorize stage chose formats or protocols")

    monkeypatch.setattr(physical_bases, "resolve", resolve)
    rows = featurize_record(
        record, record_inputs(record), "factorize", parse_models(None)
    )
    basis = BASES["factorize"]
    assert basis.feature_names == ("iterations", "outputs", "constant")
    # Each of the two products is a row, and the copy into E isn't.
    assert [np.asarray(r).shape for r in rows] == [(2, 3), (2, 3)]


@pytest.mark.skipif(not julia_available(), reason="Finch.jl is not installed")
def test_timing_a_factorization_sums_its_queries(tmp_path):
    from scripts.fitting.time_physical_data import Timer

    record = chain_record(np.random.default_rng(0))
    (tmp_path / "data").mkdir()
    path(tmp_path, "0000-00000", "program.json").write_text(json.dumps(record))
    runtimes = Timer(parse_models(None), repeat=1).time_example(
        tmp_path, "0000-00000", math.inf
    )
    timing = json.loads(path(tmp_path, "0000-00000", "timing.json").read_text())
    assert timing["runtimes"] == runtimes
    assert set(runtimes) == {"0", "1"}
    for index, runtime in runtimes.items():
        queries = timing["queries"][index]
        assert len(queries) == 2
        assert runtime == pytest.approx(sum(query["runtime"] for query in queries))
        assert all(query["runtime"] > 0 for query in queries)
        assert all(query["loop_order"] for query in queries)
