"""Adjoint gradients agree with finite differences and with each other across the linear-solver backends."""

import jax
import numpy as np
import pytest

import helpers

feax_krylov = {"feax_solver": {}}

# (label, forward solver options, adjoint solver options)
pairs = [
    (
        "lineax-cg/lineax-cg",
        {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}},
        {"lineax_solver": {"solver": "cg"}},
    ),
    (
        "lineax-bicgstab/lineax-bicgstab",
        {"newton": {"linear": {"lineax_solver": {"solver": "bicgstab"}}}},
        {"lineax_solver": {"solver": "bicgstab"}},
    ),
    ("feax-krylov/feax-krylov", {"newton": {"linear": dict(feax_krylov)}}, dict(feax_krylov)),
    (
        "lineax-cg/feax-krylov",
        {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}},
        dict(feax_krylov),
    ),
    (
        "feax-krylov/lineax-cg",
        {"newton": {"linear": dict(feax_krylov)}},
        {"lineax_solver": {"solver": "cg"}},
    ),
]

geom = {"center_post_diameter": 0.0149, "gap_size": 0.0005}
eps = 1e-6


@pytest.mark.parametrize("name,fwd,adj", pairs, ids=[p[0] for p in pairs])
def test_grad_through_forward_backend_matches_fd(name, fwd, adj):
    _, _, l_of = helpers.make_param_problem(mesh_size=0.002, fwd_opts=fwd, adj_opts=adj)

    grad = {k: float(v) for k, v in jax.grad(l_of)(geom).items()}
    assert all(np.isfinite(v) for v in grad.values())

    for key in geom:
        e = {k: 0.0 for k in geom}
        e[key] = eps
        fp = float(l_of({k: v + e[k] for k, v in geom.items()}))
        fm = float(l_of({k: v - e[k] for k, v in geom.items()}))
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - grad[key]) / max(abs(fd), 1e-12)
        assert rel < 1e-3, (
            f"[{name}] param {key}: grad {grad[key]:.6e} vs FD {fd:.6e} (rel err {rel:.3e})"
        )


def test_backends_give_consistent_gradients():
    _, _, ref = helpers.make_param_problem(mesh_size=0.002)
    g_ref = {k: float(v) for k, v in jax.grad(ref)(geom).items()}

    for name, fwd, adj in pairs[1:]:
        _, _, l_of = helpers.make_param_problem(mesh_size=0.002, fwd_opts=fwd, adj_opts=adj)
        g = {k: float(v) for k, v in jax.grad(l_of)(geom).items()}
        for key in geom:
            rel = abs(g[key] - g_ref[key]) / max(abs(g_ref[key]), 1e-12)
            assert rel < 1e-3, (
                f"[{name}] param {key}: {g[key]:.6e} vs lineax-cg {g_ref[key]:.6e} "
                f"(rel err {rel:.3e})"
            )
