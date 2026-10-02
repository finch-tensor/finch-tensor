def _generate_child(xp, S, W):
    S, W, xp = maybedefer((S, W, xp))
    turn = fused_call(_whose_turn, xp, S)
    return (S + W, turn * 2)
