"""
Saturate: fit cost models from measured programs.

Run as a script:

    python -m finch.autoschedule.fitting.saturate
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()



def saturate(prgm):
    prgms = Set()
    frontier = Set(prgm)
    workspace = HardAlias("W")
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
                        rewrites.append((node,
                            Plan(
                                Query(Table(FusedAlias(workspace, n), i), rhs),
                                Query(Table(HardAlias(A), i), Table(FusedAlias(A), i))
                            )
                        ))
                case Fuse(i, Plan(a, b)):
                    rewrites.append((node,
                        Plan(Fuse(i, a), Fuse(i, b))
                    ))
    
