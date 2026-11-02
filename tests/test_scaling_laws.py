"""L scales with turns squared, decreases with the gap, and matches the fringing model."""

import helpers


def test_l_scales_with_turns_squared():
    # current/turns are baked into the jitted assembly at the first trace,
    # so each configuration gets a fresh system with the value set first.
    prob1, fill1, _ = helpers.build_system()
    l_half = helpers.solve_l(prob1, fill1)  # turns = 0.5

    prob2, fill2, _ = helpers.build_system()
    prob2.turns = 1.0
    l_one = helpers.solve_l(prob2, fill2)  # turns = 1.0

    assert helpers.rel_err(l_one, 4.0 * l_half) < 1e-3, (
        f"L(turn) not ~ N^2: {l_one:.6e} vs 4*{l_half:.6e}"
    )


def test_l_decreases_monotonically_with_gap():
    gaps = (0.0005, 0.001, 0.002)
    ls = []
    for g in gaps:
        prob, fill, _ = helpers.build_system(
            mesh_size=0.0005, params=helpers.base_params(gap_size=g)
        )
        ls.append(helpers.solve_l(prob, fill))
    assert all(b < a for a, b in zip(ls, ls[1:])), f"L not decreasing: {ls}"


def test_gapped_l_matches_fringing_analytical():
    gap = 0.001
    prob, fill, _ = helpers.build_system(mesh_size=0.0005, params=helpers.base_params(gap_size=gap))
    l_sim = helpers.solve_l(prob, fill)
    spec = {"params": helpers.base_params(gap_size=gap)}
    from new_fem_engine.run_forward import analytical_inductance

    l_ana = analytical_inductance(spec)
    assert helpers.rel_err(l_sim, l_ana) < 0.10, f"sim {l_sim:.6e} vs analytical {l_ana:.6e}"
