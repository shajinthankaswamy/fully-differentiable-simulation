"""Shared fixtures for the engine tests: a small problem, its fill, solves and
finite-difference helpers."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import new_fem_engine.solver as slv
from new_fem_engine.generate_mesh import create_mesh, rectangle_mesh
from new_fem_engine.geometry import (
    compute_coil_areas,
    compute_fill_fractions,
    define_core_rectangles,
    geometry_settings,
)
from new_fem_engine.problem import MagnetostaticProblem, problem_settings
from new_fem_engine.run_forward import geometry_fill

sp = {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}}
adj = {"lineax_solver": {"solver": "cg"}}

rtol = 1e-3


def base_params(**overrides):
    p = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "coil_clearance": 0.0005,
        "gap_size": 0.0,
        "gap_number": 1,
        "mur": 1680.0,
        "a_e": 189e-6,
        "l_e": 93e-3,
        "sigma_l_over_a": 492.0,
        "al": 4300.0,
        "material": "N87",
        "yoke_taper": True,
    }
    p.update(overrides)
    return p


def build_system(mesh_size=0.002, params=None, nr=None, nz=None):
    """Return (problem, fill, f_core) on a uniform mesh over geometry_settings.fixed_bounds."""
    params = params if params is not None else base_params()
    spec = {"params": params, "mesh_size": mesh_size}
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    n_r = nr or max(int(round((rmax - rmin) / mesh_size)), 1)
    n_z = nz or max(int(round((zmax - zmin) / mesh_size)), 1)
    msh = rectangle_mesh(n_r, n_z, rmin, rmax, zmin, zmax)
    mesh = create_mesh(msh, "quad")
    locs = [
        lambda pt: jnp.isclose(pt[0], rmax),
        lambda pt: jnp.isclose(pt[0], rmin),
        lambda pt: jnp.isclose(pt[1], zmax),
        lambda pt: jnp.isclose(pt[1], zmin),
    ]
    prob = MagnetostaticProblem(
        mesh=mesh,
        vec=1,
        dim=2,
        dirichlet_bc_info=[locs, [0, 0, 0, 0], [lambda pt: 0.0] * 4],
        additional_info=(spec, mesh_size),
    )
    fill, (f_core, _, _) = geometry_fill(prob, n_r, n_z, mesh_size)
    return prob, fill, np.asarray(f_core)


def make_fwd(prob, fwd_opts=sp, adj_opts=adj):
    return slv.ad_wrapper(prob, fwd_opts, adj_opts)


def solve_l(prob, fill):
    """Solve with ``fill`` and return the inductance in henry."""
    prob.set_params(fill)
    sol = slv.solver(prob, sp)
    return float(prob.compute_inductance(sol))


def swap_windings(prob):
    """Exchange the primary and secondary coil rectangles on the problem."""
    prob.primary_rect, prob.secondary_rect = prob.secondary_rect, prob.primary_rect
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    xs = np.linspace(rmin, rmax, problem_settings.coil_grid_res)
    ys = np.linspace(zmin, zmax, problem_settings.coil_grid_res)
    _, f_prim, f_sec, _ = compute_fill_fractions(
        xs,
        ys,
        prob.core_rects_padded,
        prob.rect_mask,
        prob.primary_rect,
        prob.secondary_rect,
        prob.mur,
        prob.mu0,
    )
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    prob.area_prim, prob.area_sec = compute_coil_areas(f_prim, f_sec, cell_area)


def z_mirror_fill(fill, n_r, n_z):
    """Mirror the directional (num_cells, num_quads, 3) fill in z; cells are r-major."""
    f = jnp.asarray(fill)
    return f.reshape(n_r, n_z, f.shape[1], f.shape[2])[:, ::-1].reshape(f.shape)


def make_param_problem(mesh_size=0.002, fwd_opts=sp, adj_opts=adj):
    """Return (problem, fwd, l_of) with l_of(geom) the inductance of a geometry dict."""
    prob, _, f_core = build_system(mesh_size=mesh_size)
    fwd = make_fwd(prob, fwd_opts, adj_opts)
    n_r, n_z = f_core.shape[1], f_core.shape[0]
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    xs = rmin + (np.arange(n_r) + 0.5) * mesh_size
    ys = zmin + (np.arange(n_z) + 0.5) * mesh_size

    def fill_from_params(geom):
        p = base_params()
        p.update(geom)
        rects, mask, prim, sec = define_core_rectangles(p)
        _, _, _, fc = compute_fill_fractions(
            jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, p["mur"]
        )
        return jnp.repeat(fc.T.reshape(-1)[:, None], 4, axis=1)

    def l_of(geom):
        return prob.compute_inductance(fwd(fill_from_params(geom)))

    return prob, fwd, l_of


def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def richardson2(coarse, fine):
    return (4.0 * fine - coarse) / 3.0


def central_grad_fd(f, x, i, h):
    e = np.zeros_like(x)
    e[i] = h
    return float((f(x + e) - f(x - e)) / (2 * h))


def central_hessian_fd(f, x, i, j, h):
    ei = np.zeros_like(x)
    ei[i] = h
    ej = np.zeros_like(x)
    ej[j] = h
    if i == j:
        e2 = np.zeros_like(x)
        e2[i] = 2 * h
        return float((f(x + e2) - 2 * f(x) + f(x - e2)) / (4 * h * h))
    return float((f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h))


def dense_tangent(prob, n=None):
    """Return the assembled tangent as a dense (n, n) array, without Dirichlet elimination."""
    n = prob.num_total_dofs_all_vars if n is None else n
    tangent = np.zeros((n, n))
    np.add.at(
        tangent, (np.asarray(prob.row_inds), np.asarray(prob.col_inds)), np.asarray(prob.jac_values)
    )
    return tangent
