"""
Saturate: fit cost models from measured programs.

Run as a script:

    python -m finch.autoschedule.fitting.saturate
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()



def saturate(prgm, bindings):
    prgms = Set()
    frontier = Set(prgm)
    workspace_2 = HardAlias("W")
    fresh = 0
    while frontier:
        prgm = frontier.pop()
        rewrites = []
        for node in PostOrderDFS(prgm):
            match node:
                case Fuse(i, Fuse(j, body)):
                    rewrites.append(node,
                        Fuse(j, Fuse(i, body))
                    )
                case Query(Table(HardAlias(A), i), rhs) if len(i) >= 2:
                    for n in range(1, len(i)):
                        workspace = HardAlias(f"W_{fresh := fresh + 1}")
                        rewrites.append((node,
                            Plan(
                                Query(Table(FusedAlias(workspace, n), i), rhs),
                                Query(Table(HardAlias(A), i), Table(workspace, i))
                            )
                        ))
                case Fuse(i, Plan(a, b)):
                    rewrites.append((node,
                        Plan(Fuse(i, a), Fuse(i, b))
                    ))


        def flatten_plan(node):
            if isinstance(node, Plan):
                args = []
                for body in node.bodies:
                    if isinstance(body, Plan):
                        args.extend(flatten_plan(body).bodies)
                    else:
                        args.append(body)

        for (lhs, rhs) in rewrites:
            def replace(node):
                if node == lhs:
                    return rhs
                else:
                    return None
            prgm2 = Postwalk(Chain(replace, flatten_plan))(node)
            prgm2 = normalize_names(prgm2)
            if prgm2 not in prgms:
                prgms.append(prgm2)
                frontier.append(prgm2)