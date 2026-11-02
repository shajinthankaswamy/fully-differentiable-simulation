"""Limits in the permeability: mur = 1 is air, and L increases with mur."""

import jax.numpy as jnp

import helpers


def test_mur_one_equals_no_core():
    params = helpers.base_params(mur=1.0)
    prob_core, fill_core, _ = helpers.build_system(params=params)
    l_mur1 = helpers.solve_l(prob_core, fill_core)

    prob_air, _, _ = helpers.build_system()
    fill_air = jnp.zeros_like(fill_core).at[..., 1:].set(1.0)  # area 0, mu_r = mu_z = 1
    l_air = helpers.solve_l(prob_air, fill_air)

    assert helpers.rel_err(l_air, l_mur1) < 1e-6, (
        f"mur=1 core ({l_mur1:.6e}) != no-core ({l_air:.6e})"
    )


def test_mur_one_is_leakage_only_and_small():
    params = helpers.base_params(mur=1.0)
    prob_core, fill_core, _ = helpers.build_system(params=params)
    l_mur1 = helpers.solve_l(prob_core, fill_core)

    prob_full, fill_full, _ = helpers.build_system()
    l_full = helpers.solve_l(prob_full, fill_full)

    assert l_mur1 > 0
    assert l_mur1 < 0.5 * l_full, f"mur=1 L ({l_mur1:.6e}) not far below ferrite L ({l_full:.6e})"


def test_l_monotone_increasing_in_mur():
    ls = []
    for mur in (100.0, 1680.0, 10000.0):
        prob, fill, _ = helpers.build_system(params=helpers.base_params(mur=mur))
        ls.append(helpers.solve_l(prob, fill))
    assert ls[0] < ls[1] < ls[2], f"L not increasing with mur: {ls}"
