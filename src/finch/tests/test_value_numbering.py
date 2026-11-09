import numpy as np

from finch.symbolic import Apply, Constant, Leaf, Value, ValueUniverse

a, b, i, t, u, x, y = (Leaf(name) for name in "abituxy")
zero, one, two = Constant(0), Constant(1), Constant(2)


def add(*args: Value) -> Apply:
    return Apply("add", args)


def mul(*args: Value) -> Apply:
    return Apply("mul", args)


def run(universe: ValueUniverse, *writes, state=None) -> tuple[int, ...]:
    """the state after `writes`, pairs of a leaf and a value or None"""
    state = universe.empty() if state is None else state
    for leaf, value in writes:
        state = universe.assign(state, leaf, value)
    return state


def equal(universe: ValueUniverse, state: tuple[int, ...], *values: Value) -> bool:
    return len({universe.number(state, value) for value in values}) == 1


def test_values_are_interned():
    assert add(Leaf("a"), Leaf("b")) is add(a, b)
    assert add(a, b) is not add(b, a)
    assert add(a, b) is not mul(a, b)
    assert Leaf("a") is not Constant("a")


def test_constants_compare_exact_bits():
    assert Constant(np.float64(1.0)) is Constant(np.float64(1.0))
    assert Constant(np.float64("nan")) is Constant(np.float64("nan"))
    assert Constant(0.0) is not Constant(-0.0)
    assert Constant(1) is not Constant(True)
    assert Constant(1) is not Constant(1.0)
    assert Constant(np.int32(1)) is not Constant(np.int64(1))
    assert Constant(np.int64(2**53)) is not Constant(np.int64(2**53 + 1))


def test_universe_holds_args():
    universe = ValueUniverse([add(x, mul(x, one))])
    assert universe.values == [x, one, mul(x, one), add(x, mul(x, one))]
    assert universe.number(universe.empty(), y) is None
    assert run(universe, (y, x)) == universe.empty()


def test_assign():
    universe = ValueUniverse([x, y, add(a, b)])
    state = run(universe, (x, add(a, b)), (y, add(a, b)))
    assert equal(universe, state, x, y, add(a, b))
    # x and y keep the old value of `add(a, b)`
    state = run(universe, (a, None), state=state)
    assert equal(universe, state, x, y)
    assert not equal(universe, state, x, add(a, b))


def test_assign_is_congruent():
    universe = ValueUniverse([add(x, one), add(y, one)])
    state = run(universe, (x, y))
    assert equal(universe, state, add(x, one), add(y, one))
    state = run(universe, (y, one), state=state)
    assert not equal(universe, state, add(x, one), add(y, one))


def test_assign_reads_target():
    universe = ValueUniverse([t, add(x, one)])
    state = run(universe, (x, add(x, one)))
    assert not equal(universe, state, x, add(x, one))
    # `t` still holds the value written to x
    state = run(universe, (t, add(x, one)), (x, add(x, one)))
    assert equal(universe, state, x, t)
    assert not equal(universe, state, x, add(x, one))


def test_states_are_canonical():
    universe = ValueUniverse([x, y, t])
    state = run(universe, (x, t), (y, t))
    assert state == run(universe, (y, t), (x, y))
    assert universe.meet(state, state) == state
    assert run(universe, (x, x), state=state) == state


def test_meet():
    universe = ValueUniverse([x, y, add(a, b), mul(a, b)])
    state = universe.meet(
        run(universe, (x, add(a, b)), (y, add(a, b))),
        run(universe, (x, add(a, b)), (y, mul(a, b))),
    )
    assert equal(universe, state, x, add(a, b))
    assert not equal(universe, state, y, add(a, b))
    assert not equal(universe, state, y, mul(a, b))


def test_meet_relates_merged_values():
    """unlike phis named per variable, x and y are still equal after the join"""
    universe = ValueUniverse([x, y, add(x, one), add(y, one), one, two])
    state = universe.meet(
        run(universe, (x, one), (y, one)), run(universe, (x, two), (y, two))
    )
    assert equal(universe, state, x, y)
    assert equal(universe, state, add(x, one), add(y, one))
    assert not equal(universe, state, x, one)


def test_loop_fixpoint():
    """`i = 0; t = add(a, b); while ...: u = add(a, b); i = add(i, 1)`"""
    universe = ValueUniverse([t, u, zero, add(a, b), add(i, one)])
    entry = run(universe, (i, zero), (t, add(a, b)))
    header = entry
    for _ in universe.values:
        body = run(universe, (u, add(a, b)), state=header)
        latch = run(universe, (i, add(i, one)), state=body)
        if universe.meet(entry, latch) == header:
            break
        header = universe.meet(entry, latch)
    assert equal(universe, header, t, add(a, b))
    assert not equal(universe, header, i, zero)
    assert equal(universe, body, u, t)


def test_unknown_writes():
    heap = Leaf("heap")
    load = Apply("load", (a, i, heap))
    universe = ValueUniverse([x, y, load])
    state = run(universe, (x, load), (heap, None), (y, load))
    assert not equal(universe, state, x, y)
