"""The gapped inductance is converged at the working mesh."""

import pytest

import helpers


@pytest.mark.slow
def test_gapped_l_is_mesh_converged_at_operating_point():
    gap = 0.001
    params = helpers.base_params(gap_size=gap)
    l_coarse = helpers.solve_l(*helpers.build_system(mesh_size=0.00025, params=params)[:2])
    l_fine = helpers.solve_l(*helpers.build_system(mesh_size=0.000125, params=params)[:2])

    l_inf = helpers.richardson2(l_coarse, l_fine)
    assert l_inf > 0
    assert helpers.rel_err(l_fine, l_inf) < 0.02, (
        f"finest mesh {helpers.rel_err(l_fine, l_inf):.3%} off Richardson extrapolate {l_inf:.6e}"
    )


@pytest.mark.slow
def test_gapped_l_last_refinement_step_small():
    gap = 0.001
    params = helpers.base_params(gap_size=gap)
    l_h = helpers.solve_l(*helpers.build_system(mesh_size=0.00025, params=params)[:2])
    l_h2 = helpers.solve_l(*helpers.build_system(mesh_size=0.000125, params=params)[:2])
    assert abs(l_h2 - l_h) / l_h2 < 0.03, f"0.25->0.125 mm moved L by {abs(l_h2 - l_h) / l_h2:.3%}"
