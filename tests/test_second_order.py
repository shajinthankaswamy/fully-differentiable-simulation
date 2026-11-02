"""The Hessian from the adjoint gradient agrees with finite differences."""

import jax
import numpy as np
import pytest

import helpers


@pytest.mark.slow
def test_hessian_from_adjoint_gradient_matches_fd():
    _, _, l_of = helpers.make_param_problem()
    p0 = np.array([0.0149, 0.0005])  # center_post_diameter, gap_size

    def f(p):
        return l_of({"center_post_diameter": p[0], "gap_size": p[1]})

    def g(p):
        return np.asarray(jax.grad(f)(p))

    h = 1e-4
    h_adj = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        h_adj[:, j] = (g(p0 + e) - g(p0 - e)) / (2 * h)

    h_fd = np.array(
        [[helpers.central_hessian_fd(f, p0, i, j, h) for j in range(2)] for i in range(2)]
    )

    assert np.all(np.isfinite(h_adj))
    assert np.allclose(h_adj, h_adj.T, rtol=0.05), "adjoint Hessian asymmetric"
    assert np.allclose(h_adj, h_fd, rtol=0.05, atol=1e-6), (
        f"adjoint Hessian {h_adj} != FD Hessian {h_fd}"
    )
