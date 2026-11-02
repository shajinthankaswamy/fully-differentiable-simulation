"""Winding swap and z-mirror invariance of the inductance."""

import helpers


def test_winding_swap_invariance():
    prob, fill, _ = helpers.build_system()
    l_norm = helpers.solve_l(prob, fill)

    prob2, fill2, _ = helpers.build_system()
    helpers.swap_windings(prob2)
    l_swap = helpers.solve_l(prob2, fill2)

    assert helpers.rel_err(l_swap, l_norm) < 1e-3, (
        f"winding swap changed L: {l_norm:.6e} -> {l_swap:.6e}"
    )


def test_z_mirror_fill_invariance():
    prob, fill, f_core = helpers.build_system()
    l_orig = helpers.solve_l(prob, fill)

    n_z, n_r = f_core.shape
    fill_mirror = helpers.z_mirror_fill(fill, n_r, n_z)
    l_mirror = helpers.solve_l(prob, fill_mirror)

    assert helpers.rel_err(l_mirror, l_orig) < 1e-3, (
        f"z-mirror changed L: {l_orig:.6e} -> {l_mirror:.6e}"
    )
