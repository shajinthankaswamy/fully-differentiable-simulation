"""The adjoint gradient of the inductance with respect to the fill field and to geometry parameters agrees with finite differences."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import new_fem_engine.solver as slv
from new_fem_engine.generate_mesh import create_mesh, rectangle_mesh
from new_fem_engine.geometry import compute_fill_fractions, define_core_rectangles
from new_fem_engine.problem import MagnetostaticProblem
from new_fem_engine.run_forward import geometry_fill

core_spec = {
    "params": {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "coil_clearance": 0.0005,
        "gap_size": 0.0003,
        "gap_number": 1,
        "mur": 1680.0,
        "a_e": 189e-6,
        "l_e": 93e-3,
        "sigma_l_over_a": 492.0,
        "al": 4300.0,
        "material": "N87",
        "yoke_taper": True,
    },
    "mesh_size": 0.002,
}


def make_problem():
    mesh_size = core_spec["mesh_size"]
    n_r = int(round(0.035 / mesh_size))
    n_z = int(round(0.06 / mesh_size))
    msh = rectangle_mesh(n_r, n_z, 0.0, 0.035, -0.03, 0.03)
    mesh = create_mesh(msh, "quad")
    locs = [
        lambda pt: jnp.isclose(pt[0], 0.035),
        lambda pt: jnp.isclose(pt[0], 0.0),
        lambda pt: jnp.isclose(pt[1], 0.03),
        lambda pt: jnp.isclose(pt[1], -0.03),
    ]
    prob = MagnetostaticProblem(
        mesh=mesh,
        vec=1,
        dim=2,
        dirichlet_bc_info=[locs, [0, 0, 0, 0], [lambda pt: 0.0] * 4],
        additional_info=(core_spec, mesh_size),
    )
    fill, (f_core, _, _) = geometry_fill(prob, n_r, n_z, mesh_size)
    prob.set_params(fill)
    return prob, fill, f_core


@pytest.fixture(scope="module")
def setup():
    return make_problem()


def test_l_gradient_wrt_fill_matches_finite_difference(setup):
    prob, fill, f_core = setup
    fwd = slv.ad_wrapper(
        prob,
        {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}},
        {"lineax_solver": {"solver": "cg"}},
    )

    def l_of(f):
        return prob.compute_inductance(fwd(f))

    grad = np.asarray(jax.grad(l_of)(fill))
    assert np.all(np.isfinite(grad)), "adjoint gradient contains NaN/Inf"

    # A core cell, a boundary cell, an air cell and two interior cells;
    # f_core is (n_z, n_r), the fill is r-major.
    f_core_rmaj = np.asarray(f_core).T.reshape(-1)
    num_cells = fill.shape[0]
    candidates = []
    for i in range(num_cells):
        if f_core_rmaj[i] > 0.5:
            candidates.append(i)
            break
    for i in range(num_cells):
        if 0.05 < f_core_rmaj[i] < 0.95:
            candidates.append(i)
            break
    for i in range(num_cells):
        if f_core_rmaj[i] < 0.05:
            candidates.append(i)
            break
    candidates += [num_cells // 3, 2 * num_cells // 3]
    # Component 1 is the laminate permeability mu_r; the reluctivity never reads the area channel.
    cells = [(c, 0, 1) for c in dict.fromkeys(candidates)]

    eps = 1e-5
    core_cell = candidates[0]
    for idx in cells:
        e = np.zeros_like(np.asarray(fill))
        e[idx] = eps
        fp = float(l_of(jnp.asarray(np.asarray(fill) + e)))
        fm = float(l_of(jnp.asarray(np.asarray(fill) - e)))
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - grad[idx]) / max(abs(fd), 1e-12)
        assert rel < 1e-3, f"cell {idx}: adjoint {grad[idx]:.6e} vs FD {fd:.6e} (rel err {rel:.3e})"

    assert grad[core_cell, 0, 1] > 0, "core-cell gradient should be positive"


def test_l_gradient_wrt_geometry_params_matches_finite_difference(setup):
    prob, fill, f_core = setup
    fwd = slv.ad_wrapper(
        prob,
        {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}},
        {"lineax_solver": {"solver": "cg"}},
    )

    n_r, n_z = f_core.shape[1], f_core.shape[0]
    ms = core_spec["mesh_size"]
    xs = 0.0 + (np.arange(n_r) + 0.5) * ms
    ys = -0.03 + (np.arange(n_z) + 0.5) * ms

    def fill_from_params(geom):
        p = dict(core_spec["params"])
        p.update(geom)
        rects, mask, prim, sec = define_core_rectangles(p)
        _, _, _, f_core = compute_fill_fractions(
            jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, p["mur"]
        )
        return jnp.repeat(f_core.T.reshape(-1)[:, None], 4, axis=1)

    def l_geom(geom):
        return prob.compute_inductance(fwd(fill_from_params(geom)))

    geom = {
        "center_post_diameter": core_spec["params"]["center_post_diameter"],
        "leg_inner_diameter": core_spec["params"]["leg_inner_diameter"],
        "leg_height": core_spec["params"]["leg_height"],
        "window_height": core_spec["params"]["window_height"],
    }
    grad = jax.grad(l_geom)(geom)
    assert np.all(np.isfinite(jnp.stack(list(grad.values()))))
    grad = {k: float(v) for k, v in grad.items()}

    eps = 1e-6
    for name in geom:
        e = {k: 0.0 for k in geom}
        e[name] = eps
        fp = float(l_geom({k: v + e[k] for k, v in geom.items()}))
        fm = float(l_geom({k: v - e[k] for k, v in geom.items()}))
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - grad[name]) / max(abs(fd), 1e-12)
        assert rel < 1e-3, (
            f"param {name}: adjoint {grad[name]:.6e} vs FD {fd:.6e} (rel err {rel:.3e})"
        )

    assert grad["center_post_diameter"] > 0, "wider post should raise L"


def test_ad_wrapper_gradient_nonzero_and_finite(setup):
    prob, fill, f_core = setup
    fwd = slv.ad_wrapper(
        prob,
        {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}},
        {"lineax_solver": {"solver": "cg"}},
    )

    def l_of(f):
        return prob.compute_inductance(fwd(f))

    grad = np.asarray(jax.grad(l_of)(fill))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) > 0
    f_core_rmaj = np.asarray(f_core).T.reshape(-1)
    core_mask = f_core_rmaj > 0.5
    # Spans only: the area channel carries no gradient by construction.
    assert grad[core_mask][..., 1:].min() > 0, "some core cells have non-positive dl/d(span)"
