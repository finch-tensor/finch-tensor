import numpy as np

from finchlite.algebra import ffuncs, ftype
from finchlite.tensor import dense, element, sparse_hash


# Abstract the cost
def _dense_next_num_pos(num_pos, n_l, nnz_l):
    return num_pos * n_l


def _sparse_hash_next_num_pos(num_pos, n_l, nnz_l):
    return nnz_l


# How we plan to build levels
DENSE = ("dense", lambda lvl, dim_type: dense(lvl, dim_type), _dense_next_num_pos)
SPARSE_HASH = (
    "sparse_hash",
    lambda lvl, dim_type: sparse_hash(lvl, dim_type, single_writer=False),
    _sparse_hash_next_num_pos,
)
CANDIDATES = (DENSE, SPARSE_HASH)


def optimize_format(
    fields, shape_type, stats, stats_factory, fill_value, cost_fn, candidates=CANDIDATES
):
    n = len(fields)
    fill_ftype = ftype(fill_value)
    leaf = element(fill_value, fill_ftype)
    val_size = np.dtype(fill_ftype.dtype).itemsize
    pos_size = np.dtype(leaf.position_type.dtype).itemsize

    def nnz_after(level):
        reduce_fields = tuple(fields[level + 1 :])
        if reduce_fields:
            reduced = stats_factory.aggregate(ffuncs.or_, False, reduce_fields, stats)
        else:
            reduced = stats
        return reduced.estimate_non_fill_values()

    # memoizing (l,nnz) : (best_cost,best_fmt)
    memo = {}

    def rec(level, num_pos):
        """
        We are trying to keep this as :
        def optimise_fmt(S,n,p):
            d_cost = C(S,(D,optimise_fmt(S,n-1,p*n_l)))
            s_cost = C(S,(D,optimise_fmt(S,n-1,q)))

            if d_cost < s_cost :
                return D, optimise_fmt(S,n-1,p*n_l)
            else:
                return D, optimise_fmt(S,n-1,q)
        """
        if level == n:
            return cost_fn("leaf", num_pos, None, None, val_size, pos_size), leaf

        key = (level, num_pos)
        if key in memo:
            return memo[key]

        n_l = stats.get_dim_size(fields[level])
        nnz_l = nnz_after(level)

        best_cost = None
        best_format = None
        for _name, build, next_num_pos in candidates:
            local = cost_fn(_name, num_pos, n_l, nnz_l, val_size, pos_size)
            child_num_pos = next_num_pos(num_pos, n_l, nnz_l)
            child_cost, child_fmt = rec(level + 1, child_num_pos)
            total = local + child_cost
            if best_cost is None or total < best_cost:
                best_cost, best_format = total, build(child_fmt, shape_type[level])

        memo[key] = (best_cost, best_format)
        return best_cost, best_format

    _, fmt = rec(0, 1.0)
    return fmt
