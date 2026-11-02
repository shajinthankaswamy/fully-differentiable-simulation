"""Signs of the inductance gradient: more permeability raises L, more gap lowers it."""

import jax
import numpy as np

import helpers


def test_core_cells_positive_air_zero():
    prob, fill, f_core = helpers.build_system()
    fwd = helpers.make_fwd(prob)

    def l_of(f):
        return prob.compute_inductance(fwd(f))

    grad = np.asarray(jax.grad(l_of)(fill))
    fc = f_core.T.reshape(-1)  # r-major, matching cell order

    core_cells = np.flatnonzero(fc > 0.5)
    air_cells = np.flatnonzero(fc < 0.05)
    assert core_cells.size > 0 and air_cells.size > 0
    # The reluctivity reads only the two permeability components; the area
    # channel's gradient is identically zero.
    assert np.all(grad[..., 0] == 0.0), (
        "the area channel is unused by _reluctivity and must carry no gradient; "
        "a non-zero value here means the homogenisation changed"
    )
    # At fixed current the stored energy grows with the permeability of every
    # cell, so dL/dmu is positive in the core and non-negative everywhere.
    mu = grad[..., 1:]
    assert mu[core_cells].min() > 0, "core cells must have dL/dmu > 0"
    assert mu[air_cells].min() >= -1e-9 * np.abs(mu).max(), "dL/dmu must not be negative in air"


def test_geometry_param_gradient_signs():
    _, _, l_of = helpers.make_param_problem()
    geom = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "gap_size": 0.0003,
    }
    g = jax.grad(l_of)(geom)
    assert float(g["center_post_diameter"]) > 0, "wider post must raise L"
    assert float(g["leg_inner_diameter"]) < 0, "thinner legs must lower L"
    assert float(g["leg_height"]) > 0, "taller core must raise L"
    assert float(g["window_height"]) < 0, "taller window must lower L"
    assert float(g["gap_size"]) < 0, "more gap must lower L"
