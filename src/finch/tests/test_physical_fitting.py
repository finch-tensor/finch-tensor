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
from finch.autoschedule.cost_models import DenseLevelOption
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
)
from finch.symbolic import PostOrderDFS
from scripts.fitting.generate_physical_data import (
    BLOCK_BITS,
    input_record,
    materialize,
    path,
    record_inputs,
)
from scripts.fitting.physical_bases import (
    BASES,
    A,
    B,
    C,
    featurize_record,
    input_perms,
    parse_models,
    resolve,
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
    rows = featurize_record(record, record_inputs(record), "loop", parse_models(None))
    basis = BASES["loop"]
    names = basis.feature_names
    totals = {
        tuple(variant["loop_order"]): dict(
            zip(names, np.asarray(variant_rows).sum(axis=0), strict=True)
        )
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
