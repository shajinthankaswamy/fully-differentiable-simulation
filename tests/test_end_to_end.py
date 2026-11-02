"""A loss built on the differentiable solve has a usable gradient and one step decreases it."""

import jax
import numpy as np
import pytest

import helpers

target = 260e-9


@pytest.mark.slow
def test_loss_gradient_finite_and_nonzero():
    _, _, l_of = helpers.make_param_problem()
    geom = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "gap_size": 0.0009,
    }

    def loss(g):
        return (l_of(g) - target) ** 2

    g = jax.grad(loss)(geom)
    for v in g.values():
        assert np.isfinite(float(v)), "non-finite loss gradient"
    assert any(abs(float(v)) > 0 for v in g.values()), "zero loss gradient"


@pytest.mark.slow
def test_one_gradient_step_decreases_loss():
    _, _, l_of = helpers.make_param_problem()
    geom = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "gap_size": 0.0009,
    }

    def loss(g):
        return (l_of(g) - target) ** 2

    loss0 = float(loss(geom))
    g = jax.grad(loss)(geom)

    geom1 = {k: v - np.sign(float(g[k])) * 1e-4 * abs(v) for k, v in geom.items()}
    loss1 = float(loss(geom1))
    assert loss1 < loss0, f"gradient step raised loss {loss0:.3e} -> {loss1:.3e}"
