import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch.algebra import TensorFType
from finch.autoschedule import NotationGenerator
from finch.autoschedule.stages import CompilerForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.compile import CompilerMode, NotationCompiler


@pytest.mark.parametrize(
    "widths", [(3, 3), (3, 5)], ids=["same_width", "different_width"]
)
def test_fused_workspace_reuse_with_copies(widths):
    i, j, k = map(lgc.Field, "ijk")
    a, b, w, c, d = map(lgc.HardAlias, "abwcd")
    fused = lgc.FusedAlias(w, 1)
    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Plan(
                    (
                        lgc.Fuse(
                            j, lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i, j)))
                        ),
                        lgc.Fuse(
                            j, lgc.Query(lgc.Table(c, (i, j)), lgc.Table(fused, (i, j)))
                        ),
                        lgc.Fuse(
                            k, lgc.Query(lgc.Table(fused, (i, k)), lgc.Table(b, (i, k)))
                        ),
                        lgc.Fuse(
                            k, lgc.Query(lgc.Table(d, (i, k)), lgc.Table(fused, (i, k)))
                        ),
                    )
                ),
            ),
            lgc.Produces((c, d)),
        )
    )
    nj, nk = widths
    data_a = np.arange(2 * nj, dtype=np.int64).reshape(2, nj) + 1
    data_b = np.arange(2 * nk, dtype=np.int64).reshape(2, nk) + 10
    bindings = {
        a: ft.asarray(data_a),
        b: ft.asarray(data_b),
        w: ft.asarray(np.zeros((0, 0), dtype=np.int64)),
        c: ft.asarray(np.zeros((0, 0), dtype=np.int64)),
        d: ft.asarray(np.zeros((0, 0), dtype=np.int64)),
    }
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: val.ftype for var, val in bindings.items()
    }
    CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    program = NotationGenerator(mode=CompilerMode(safe=True))(plan, ftypes, {}, None)
    result_c, result_d = NotationCompiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result_c.to_numpy(), data_a)
    np.testing.assert_array_equal(result_d.to_numpy(), data_b)
