import numpy as np

from finch import ffuncs
from finch.algebra.ftypes import ftype
from finch.finch_assembly import (
    Assign,
    Block,
    Call,
    ForLoop,
    IfElse,
    Literal,
    Load,
    Store,
    Variable,
    parse_assembly,
)


def test_for_loop():
    lvl_ptr = Variable("lvl_ptr", ftype(np.ndarray))
    pos_stop = Variable("pos_stop", ftype(int))
    qos_stop = Variable("qos_stop", ftype(int))
    lvl_idx = Variable("lvl_idx", ftype(np.ndarray))
    p = Variable("p", ftype(int))

    expr = """finch
    resize(lvl_ptr, pos_stop + 1)
    for (p in 0:pos_stop)
        lvl_ptr[p + 1] += lvl_ptr[p]
    end
    qos_stop = lvl_ptr[pos_stop] - 1
    // some comment
    resize(lvl_idx, qos_stop)
    """

    result = parse_assembly(expr, locals())

    expected = Block(
        (
            Call(
                op=Literal(val=np.resize),
                args=(
                    Variable(name="lvl_ptr", type_=ftype(np.ndarray)),
                    Call(
                        op=Literal(val=ffuncs.add),
                        args=(Variable(name="pos_stop", type_=ftype(int)), Literal(val=1)),
                    ),
                ),
            ),
            ForLoop(
                var=Variable(name="p", type_=ftype(int)),
                start=Literal(val=0),
                end=Variable(name="pos_stop", type_=ftype(int)),
                body=Block(
                    bodies=(
                        Store(
                            buffer=Variable(name="lvl_ptr", type_=ftype(np.ndarray)),
                            index=Call(
                                op=Literal(val=ffuncs.add),
                                args=(Variable(name="p", type_=ftype(int)), Literal(val=1)),
                            ),
                            value=Call(
                                op=Literal(val=ffuncs.add),
                                args=(
                                    Load(
                                        buffer=Variable(
                                            name="lvl_ptr", type_=ftype(np.ndarray)
                                        ),
                                        index=Call(
                                            op=Literal(val=ffuncs.add),
                                            args=(
                                                Variable(name="p", type_=ftype(int)),
                                                Literal(val=1),
                                            ),
                                        ),
                                    ),
                                    Load(
                                        buffer=Variable(
                                            name="lvl_ptr", type_=ftype(np.ndarray)
                                        ),
                                        index=Variable(name="p", type_=ftype(int)),
                                    ),
                                ),
                            ),
                        ),
                    )
                ),
            ),
            Assign(
                lhs=Variable(name="qos_stop", type_=ftype(int)),
                rhs=Call(
                    op=Literal(val=ffuncs.sub),
                    args=(
                        Load(
                            buffer=Variable(name="lvl_ptr", type_=ftype(np.ndarray)),
                            index=Variable(name="pos_stop", type_=ftype(int)),
                        ),
                        Literal(val=1),
                    ),
                ),
            ),
            Call(
                op=Literal(val=np.resize),
                args=(
                    Variable(name="lvl_idx", type_=ftype(np.ndarray)),
                    Variable(name="qos_stop", type_=ftype(int)),
                ),
            ),
        )
    )

    assert result == expected


def test_if_statement():
    lvl_ptr = Variable("lvl_ptr", ftype(np.ndarray))
    lvl_idx = Variable("lvl_idx", ftype(np.ndarray))
    pos = Variable("pos", ftype(int))
    q = Variable("q", ftype(int))
    q_stop = Variable("q_stop", ftype(int))
    i = Variable("i", ftype(int))
    i1 = Variable("i1", ftype(int))

    expr = """finch
    q = lvl_ptr[pos]
    q_stop = lvl_ptr[pos + 1]
    if (q < q_stop)
        i = lvl_idx[q]
        i1 = lvl_idx[q_stop - 1]
    else
        i = 1
        i1 = 0
    end
    """

    result = parse_assembly(expr, locals())

    expected = Block(
        (
            Assign(q, Load(lvl_ptr, pos)),
            Assign(
                q_stop,
                Load(lvl_ptr, Call(Literal(ffuncs.add), (pos, Literal(np.intp(1))))),
            ),
            IfElse(
                Call(Literal(ffuncs.lt), (q, q_stop)),
                Block(
                    (
                        Assign(i, Load(lvl_idx, q)),
                        Assign(
                            i1,
                            Load(
                                lvl_idx,
                                Call(
                                    Literal(ffuncs.sub), (q_stop, Literal(np.intp(1)))
                                ),
                            ),
                        ),
                    )
                ),
                Block(
                    (
                        Assign(i, Literal(np.intp(1))),
                        Assign(i1, Literal(np.intp(0))),
                    )
                ),
            ),
        )
    )

    assert result == expected
