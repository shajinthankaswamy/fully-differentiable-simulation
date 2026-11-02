"""
Forward Solve of One Core
=========================
Solves the core in ExperimentConfig at 1 A through 1 turn with lineax and
feax, prints the inductance and energy checks, writes the figures under
output/<backend>_forward/, and times the forward and backward (adjoint) pass
of each backend.

    python new_fem_engine/run_forward.py [core_name]
"""

import copy
import csv
import gc
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field

_import_start = time.time()

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __package__ in (None, ""):
    import new_fem_engine.solver as slv
    from new_fem_engine.feax_compat import iterative_options_cls
    from new_fem_engine.generate_mesh import conforming_quad_mesh, create_mesh, rectangle_mesh
    from new_fem_engine.geometry import (
        compute_fill_fractions_grid,
        compute_fill_laminate,
        conforming_edges,
        geometry_settings,
    )
    from new_fem_engine.problem import MagnetostaticProblem, problem_settings
else:
    from . import solver as slv
    from .feax_compat import iterative_options_cls
    from .generate_mesh import conforming_quad_mesh, create_mesh, rectangle_mesh
    from .geometry import (
        compute_fill_fractions_grid,
        compute_fill_laminate,
        conforming_edges,
        geometry_settings,
    )
    from .problem import MagnetostaticProblem, problem_settings

_import_s = time.time() - _import_start

logging.getLogger("new_fem_engine").setLevel(logging.WARNING)

output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


# ---- SETTINGS ----


# fmt: off
@dataclass
class ExperimentConfig:
    """Settings of a forward run; edit here to change a run."""

    # --- Core ---
    core_name:        str   = "pq_40x40"     # core YAML under cores/
    mesh_size:        float = 0.0001         # cell size                [m]
    gap_size:         float = 0.001          # air gap                  [m]
    gap_number:       int   = 1              # number of gaps
    yoke_taper:       bool  = True           # yoke height h = ct / r
    conforming:       bool  = True           # grid lines on material edges

    # --- Solver ---
    reference_solver: str   = "lineax"       # backend quoted in the report
    backends:         tuple = ("lineax", "feax")
    preconditioner:   str   = "multigrid"    # shared by every backend
    mg_sweeps:        int   = 3              # smoother sweeps per level
    mg_min_dofs:      int   = 64             # coarsest multigrid level size
    grid_shape:       tuple = None           # node grid, set once the mesh exists

    # CG tolerance 1e-6: where the inductance error reaches its round-off
    # plateau (comparison_studies/tolerance_comparison.py).
    lineax: dict = field(default_factory=lambda: {"solver": "cg", "rtol": 1e-6, "atol": 1e-6, "max_steps": 10000})
    feax:   dict = field(default_factory=lambda: {"solver": "cg", "rtol": 1e-6, "atol": 1e-6, "maxiter": 10000})

    # --- Output ---
    save_plots:       bool  = True           # write the figure set
    output_dir:       str   = field(default_factory=lambda: output_root)
# fmt: on


# ---- SOLVER OPTIONS ----


def _precond_options(cfg):
    return {
        "preconditioner": cfg.preconditioner,
        "grid_shape": cfg.grid_shape,
        "mg_sweeps": cfg.mg_sweeps,
        "mg_min_dofs": cfg.mg_min_dofs,
    }


def lineax_options(cfg):
    """Return solver options for the lineax backend."""
    return {"newton": {"linear": {"lineax_solver": {**dict(cfg.lineax), **_precond_options(cfg)}}}}


def feax_options(cfg):
    """Return solver options for the feax iterative backend."""
    fe = cfg.feax
    options = iterative_options_cls()(
        solver=fe.get("solver", "cg"),
        tol=fe.get("rtol", 1e-10),
        atol=fe.get("atol", 1e-10),
        maxiter=fe.get("maxiter", 10000),
    )
    return {"newton": {"linear": {"feax_solver": {"options": options, **_precond_options(cfg)}}}}


_backend_builders = {
    "lineax": lineax_options,
    "feax": feax_options,
}


def build_solver_options(name, cfg):
    """Return the solver options for backend ``name`` under ``cfg``."""
    builder = _backend_builders.get(name)
    if builder is None:
        raise ValueError(f"Unknown backend {name!r}. Available: {sorted(_backend_builders)}")
    return builder(cfg)


def solver_label(cfg):
    """Return the solver every backend runs, e.g. ``cg + multigrid (3 sweeps, coarsest <= 64 dofs)``."""
    methods = sorted({cfg.lineax.get("solver", "cg"), cfg.feax.get("solver", "cg"), "cg"})
    pre = cfg.preconditioner
    if pre == "multigrid":
        pre += f" ({cfg.mg_sweeps} sweeps, coarsest <= {cfg.mg_min_dofs} dofs)"
    return f"{'/'.join(methods)} + {pre}"


def backend_label(name):
    """Return the display name of a backend."""
    return {"lineax": "lineax", "feax": "feax"}.get(name, name)


# ---- CORE SPEC ----


# Used only when the core YAML cannot be loaded.
_inline_core_spec = {
    "name": "pq_40x40",
    "params": {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "coil_clearance": 0.0005,
        "mur": 1680.0,
        "a_e": 189e-6,
        "l_e": 93e-3,
        "sigma_l_over_a": 492.0,
        "al": 4300.0,
        "material": "N87",
    },
}

_steinmetz_temp = 100
_steinmetz_fit = "linear"


def load_core_spec(core_name):
    """Load a core spec through optimizers.objective, or the inline fallback."""
    try:
        from optimizers import objective

        return objective.load_core_spec(core_name)
    except Exception as exc:
        print(f"core spec unavailable ({exc}); using inline spec")
        spec = dict(_inline_core_spec)
        spec["name"] = core_name
        return spec


def apply_config(core_spec, cfg):
    """Write the mesh and gap settings of ``cfg`` into ``core_spec`` in place."""
    core_spec["mesh_size"] = cfg.mesh_size
    core_spec["params"]["gap_size"] = cfg.gap_size
    core_spec["params"]["gap_number"] = cfg.gap_number
    core_spec["params"]["yoke_taper"] = cfg.yoke_taper
    core_spec["conforming"] = cfg.conforming


def load_steinmetz(material):
    """Return (k, alpha, beta, fit_mode) for ``material`` from the cores CSV, or None."""
    csv_path = os.path.join(_repo_root, "cores", "steinmetz_full_coefficients.csv")
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["material"] == material
                and int(float(row["T_C"])) == _steinmetz_temp
                and row["fit_mode"] == _steinmetz_fit
            ):
                return (
                    float(row["k_stein"]),
                    float(row["alpha"]),
                    float(row["beta"]),
                    row["fit_mode"],
                )
    return None


# ---- MESH AND MATERIAL ----


def build_problem(core_spec):
    """Build the mesh and the magnetostatic problem for a core spec.

    Returns:
        (problem, n_r, n_z, mesh_size) with n_r x n_z cells over geometry_settings.fixed_bounds.
    """
    mesh_size = float(core_spec.get("mesh_size", 0.001))
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    conforming = bool(core_spec.get("conforming", False))

    if conforming:
        r_edges, z_edges = conforming_edges(
            core_spec["params"], geometry_settings.fixed_bounds, mesh_size
        )
        n_r, n_z = len(r_edges) - 1, len(z_edges) - 1
        msh = conforming_quad_mesh(r_edges, z_edges)
    else:
        n_r = max(int(round((rmax - rmin) / mesh_size)), 1)
        n_z = max(int(round((zmax - zmin) / mesh_size)), 1)
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
        additional_info=(core_spec, mesh_size),
    )
    if conforming:
        prob._edges = (r_edges, z_edges)
    return prob, n_r, n_z, mesh_size


def geometry_fill(prob, n_r, n_z, mesh_size):
    """Return the material fill of every cell.

    Returns:
        (fill, (f_core, f_prim, f_sec)): fill is (cells, 4, 3), the core area
        fraction and r and z spans per quadrature point, r-major; the three
        grids are (n_z, n_r).
    """
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    edges = getattr(prob, "_edges", None)
    if edges is not None:
        r_edges, z_edges = np.asarray(edges[0]), np.asarray(edges[1])
    else:
        r_edges = np.linspace(rmin, rmax, n_r + 1)
        z_edges = np.linspace(zmin, zmax, n_z + 1)
    r_e, z_e = jnp.asarray(r_edges), jnp.asarray(z_edges)

    _, f_prim, f_sec, f_core = compute_fill_fractions_grid(
        r_e,
        z_e,
        prob.core_rects_padded,
        prob.rect_mask,
        prob.primary_rect,
        prob.secondary_rect,
        prob.mur,
        prob.mu0,
    )
    f_area, mu_r, mu_z = compute_fill_laminate(
        r_e, z_e, prob.core_rects_padded, prob.rect_mask, prob.mur
    )

    # The grids are z-major and the engine's cells r-major, hence the .T.
    cell = jnp.stack([f_area.T.reshape(-1), mu_r.T.reshape(-1), mu_z.T.reshape(-1)], axis=-1)
    fill = jnp.repeat(cell[:, None, :], 4, axis=1)
    return fill, (np.asarray(f_core), np.asarray(f_prim), np.asarray(f_sec))


# ---- INDUCTANCE AND FIELDS ----


def core_path_reluctance_sum(p):
    """Return sum(l / A) of the ferrite path [1/m]: post, outer leg and the two tapered yokes."""
    r_post = p["center_post_diameter"] / 2.0
    r_inner = p["leg_inner_diameter"] / 2.0
    r_outer = np.sqrt(r_inner**2 + p["a_e"] / np.pi)
    yoke_path = 0.5 * (r_inner + r_outer) - 0.5 * r_post
    yoke = 0.5 * (p["leg_height"] - p["window_height"])
    yoke_area = 2.0 * np.pi * r_post * yoke
    return (
        p["window_height"] / (np.pi * r_post**2)
        + p["window_height"] / p["a_e"]
        + 2.0 * yoke_path / yoke_area
    )


def analytical_inductance(core_spec):
    """Return the reluctance-model inductance of one turn [H], from the design's own sum(l / A)."""
    p = core_spec["params"]
    mur = p["mur"]
    sigma_l_over_a = core_path_reluctance_sum(p)
    window_height = p["window_height"]
    r_post = p["center_post_diameter"] / 2.0
    gap = p.get("gap_size", 0.0)

    a_c = np.pi * r_post**2
    if gap > 0:
        a_gap = a_c + gap * np.sqrt(a_c) * np.log(2.0 * window_height / gap)
    else:
        a_gap = r_post**2

    return problem_settings.mu0 / (
        (1.0 / mur) * (sigma_l_over_a - gap / (r_post**2 * np.pi)) + gap / a_gap
    )


def nodal_b_components(psi_grid, edges=None):
    """Return (b_r, b_z) at the mesh nodes by centred differences of A_phi; for plots only."""
    n_r, n_z = psi_grid.shape[0] - 1, psi_grid.shape[1] - 1
    r_nodes, z_nodes = _uniform_edges(n_r, n_z) if edges is None else edges
    r_nodes, z_nodes = np.asarray(r_nodes), np.asarray(z_nodes)
    on_axis = (r_nodes <= 0.0)[:, None]
    r_safe = np.where(r_nodes <= 0.0, 1.0, r_nodes)[:, None]
    a_phi = np.where(on_axis, 0.0, psi_grid / r_safe)
    da_dr = np.gradient(a_phi, r_nodes, axis=0)
    b_r = -np.gradient(a_phi, z_nodes, axis=1)
    b_z = np.where(on_axis, 2.0 * da_dr, a_phi / r_safe + da_dr)  # A_phi ~ c r on the axis
    return b_r, b_z


def nodal_bfield(psi_grid, edges=None):
    """Return |B| at the mesh nodes; see nodal_b_components."""
    b_r, b_z = nodal_b_components(psi_grid, edges)
    return np.sqrt(b_r**2 + b_z**2)


def nodal_aphi(psi_grid, edges=None):
    """Return A_phi = psi / r at the nodes, zero on the axis."""
    n_r = psi_grid.shape[0] - 1
    n_z = psi_grid.shape[1] - 1
    r_nodes = np.asarray(_uniform_edges(n_r, n_z)[0] if edges is None else edges[0])
    r_safe = np.where(r_nodes <= 0.0, 1.0, r_nodes)[:, None]
    return np.where((r_nodes <= 0.0)[:, None], 0.0, psi_grid / r_safe)


def cell_grid(values_per_cell, n_r, n_z):
    """Reshape an r-major per-cell vector to an (n_r, n_z) array."""
    return np.asarray(values_per_cell).reshape(n_r, n_z)


def cell_permeability(f_core, mur, n_r, n_z):
    """Return the display permeability per cell as (n_r, n_z), from a (n_z, n_r) grid or a per-cell vector."""
    f = np.asarray(f_core)
    f = f.T if f.ndim == 2 else cell_grid(f, n_r, n_z)
    return f * mur + (1.0 - f)


def cell_energy_density(b_mag_node, mu_r_cell):
    """Return w = |B|^2 / (2 mu0 mu_r) per cell from the nodal |B|."""
    b = 0.25 * (
        b_mag_node[:-1, :-1] + b_mag_node[1:, :-1] + b_mag_node[:-1, 1:] + b_mag_node[1:, 1:]
    )
    return b**2 / (2.0 * problem_settings.mu0 * mu_r_cell)


def field_energy(prob, sol_list, fill):
    """Return (core, total) magnetic energy [J], integrated as the stiffness was assembled."""
    nu2 = np.asarray(prob.internal_vars[0])
    grad_psi = np.asarray(prob.fes[0].sol_to_grad(sol_list[0]))
    e_cell = np.sum(
        0.5 * np.sum(nu2 * grad_psi[:, :, 0, :] ** 2, axis=-1) * np.asarray(prob.fes[0].jxw), axis=1
    )
    f = np.asarray(fill)
    core = (f[:, 0, 0] if f.ndim == 3 else f[:, 0]) > 0.5
    return float(e_cell[core].sum()), float(e_cell.sum())


def psi_to_grid(psi, n_r, n_z):
    """Reshape the nodal solution to an (n_r + 1, n_z + 1) grid."""
    return np.asarray(psi).ravel().reshape(n_r + 1, n_z + 1)


def grid_edges(prob, n_r, n_z):
    """Return (r_nodes, z_nodes) of the structured mesh, read off the mesh itself."""
    pts = np.asarray(prob.fes[0].points).reshape(n_r + 1, n_z + 1, 2)
    return pts[:, 0, 0], pts[0, :, 1]


def _uniform_edges(n_r, n_z):
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    return np.linspace(rmin, rmax, n_r + 1), np.linspace(zmin, zmax, n_z + 1)


def source_ampere_turns(prob):
    """Return the source ampere-turns, integrated with the RHS quadrature."""
    return float(np.sum(prob.coil_current_density() * prob.fes[0].jxw))


def linkage_inductance(prob, sol_list):
    """Return the flux-linkage inductance L = lambda / I with lambda = int J A_phi dV."""
    psi_quads = prob.fes[0].convert_from_dof_to_quad(sol_list[0])
    jxw = prob.fes[0].jxw[:, :, None]
    j_quads = prob.coil_current_density()[:, :, None]
    lam = 2 * np.pi * np.sum(psi_quads * j_quads * jxw)
    return float(lam / prob.current**2)


# ---- PLOTS ----


def _pick_indices(n, max_lines=80):
    if n <= max_lines:
        return np.arange(n)
    step = int(np.ceil(n / max_lines))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _post_gaps(rects, mask):
    """Return [(z_low, z_high, r_post)] for every gap in the centre post."""
    column = sorted(
        (float(z0), float(z1), float(r1))
        for (r0, z0, r1, z1), used in zip(rects, mask)
        if used and r0 == 0.0 and r1 > 0.0
    )
    return [
        (z1_low, z0_high, r_post)
        for (_, z1_low, r_post), (z0_high, _, _) in zip(column, column[1:])
        if z0_high > z1_low
    ]


def plot_geometry(prob, n_r, n_z, mesh_size, path, show_mesh=False):
    """Draw the core and coil rectangles, optionally with the mesh lines."""
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    fig, ax = plt.subplots(figsize=(6.5, 6))

    rects = np.asarray(prob.core_rects_padded)
    mask = np.asarray(prob.rect_mask)
    core_patches = []
    for i in range(len(rects)):
        if mask[i]:
            r0, z0, r1, z1 = rects[i]
            core_patches.append(Rectangle((r0, z0), r1 - r0, z1 - z0))
    ax.add_collection(PatchCollection(core_patches, facecolor="gray", alpha=0.8, label="Core"))
    for z_low, z_high, r_post in _post_gaps(rects, mask):
        width = max(z_high - z_low, mesh_size)  # a sub-pixel gap would vanish
        z_mid = 0.5 * (z_low + z_high)
        ax.add_patch(Rectangle((0.0, z_mid - width / 2.0), r_post, width, fc="white", ec="none"))

    legend_elements = [
        Patch(facecolor="gray", alpha=0.8, label="Core"),
        Patch(facecolor="orange", alpha=0.6, label="Primary Coil"),
    ]
    prim = np.asarray(prob.primary_rect)
    sec = np.asarray(prob.secondary_rect)
    for rect, color, label in (
        (prim, "orange", None),
        (sec, "yellow", None),
    ):
        r0, z0, r1, z1 = rect
        ax.add_patch(Rectangle((r0, z0), r1 - r0, z1 - z0, fc=color, ec="none", alpha=0.7))
    if sec[2] > sec[0]:
        legend_elements.append(Patch(facecolor="yellow", alpha=0.6, label="Secondary Coil"))
    legend_elements.append(Patch(facecolor="white", edgecolor="black", label="Air"))
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7,
        handlelength=1.0,
        handleheight=0.8,
        borderpad=0.3,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.2,
    )

    if show_mesh:
        r_edges, z_edges = grid_edges(prob, n_r, n_z)
        for i in _pick_indices(r_edges.size):
            ax.plot([r_edges[i], r_edges[i]], [zmin, zmax], color="k", linewidth=0.3, alpha=0.35)
        for j in _pick_indices(z_edges.size):
            ax.plot([rmin, rmax], [z_edges[j], z_edges[j]], color="k", linewidth=0.3, alpha=0.35)

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Geometry with Mesh" if show_mesh else "Geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psi(psi_grid, path, edges=None):
    """Draw psi with its level sets, which are the flux lines."""
    n_r, n_z = psi_grid.shape[0] - 1, psi_grid.shape[1] - 1
    r_edges, z_edges = _uniform_edges(n_r, n_z) if edges is None else edges
    rr, zz = np.meshgrid(np.asarray(r_edges), np.asarray(z_edges), indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.pcolormesh(rr, zz, psi_grid, shading="gouraud", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$\psi = r A_\phi$")
    ax.contour(rr, zz, np.asarray(psi_grid), levels=24, colors="white", linewidths=0.5, alpha=0.8)
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"Magnetic vector potential $\psi$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _colour_ceiling(field, pct=99.5):
    """Return a colour-scale top that the few singular corner pixels cannot set."""
    f = np.asarray(field)
    live = f[f > 0.01 * f.max()] if f.max() > 0 else f
    return float(np.percentile(live, pct)) if live.size else float(f.max())


def plot_bfield(b_mag_node, n_r, n_z, path, edges=None, psi_grid=None, note=""):
    """Draw |B| with a capped colour scale and, given psi_grid, the flux lines."""
    r_edges, z_edges = _uniform_edges(n_r, n_z) if edges is None else edges
    rr, zz = np.meshgrid(np.asarray(r_edges), np.asarray(z_edges), indexing="ij")
    b = np.asarray(b_mag_node)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.pcolormesh(
        rr, zz, b, shading="gouraud", cmap="magma", vmin=0.0, vmax=_colour_ceiling(b)
    )
    fig.colorbar(im, ax=ax, label=r"$|B|$ [T]", extend="max")
    if psi_grid is not None:
        ax.contour(
            rr, zz, np.asarray(psi_grid), levels=24, colors="white", linewidths=0.5, alpha=0.8
        )
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    title = r"Magnetic flux density $|B|$"
    if note:
        title += f"  ({note})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_material_and_current(f_core, j_cell, mur, n_r, n_z, path, edges=None):
    """Draw the relative permeability and the source current density per cell."""
    r_edges, z_edges = _uniform_edges(n_r, n_z) if edges is None else edges
    rr, zz = np.meshgrid(np.asarray(r_edges), np.asarray(z_edges), indexing="ij")
    mu_r = cell_permeability(f_core, mur, n_r, n_z)
    j = cell_grid(j_cell, n_r, n_z)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im = axes[0].pcolormesh(rr, zz, mu_r, shading="flat", cmap="hot")
    fig.colorbar(im, ax=axes[0], label=r"$\mu_r$")
    axes[0].set_title(r"Relative permeability $\mu_r$")
    im = axes[1].pcolormesh(rr, zz, j, shading="flat", cmap="Blues")
    fig.colorbar(im, ax=axes[1], label=r"$J_\phi$ [A/m$^2$]")
    axes[1].set_title(r"Current density $J_\phi$ (1 A, 1 turn)")
    for ax in axes:
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aphi(psi_grid, path, edges=None):
    """Draw A_phi = psi / r."""
    n_r, n_z = psi_grid.shape[0] - 1, psi_grid.shape[1] - 1
    r_edges, z_edges = _uniform_edges(n_r, n_z) if edges is None else edges
    rr, zz = np.meshgrid(np.asarray(r_edges), np.asarray(z_edges), indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.pcolormesh(rr, zz, nodal_aphi(psi_grid, edges), shading="gouraud", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$A_\phi$ [Wb/m]")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"Magnetic vector potential $A_\phi$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_density(w_cell, n_r, n_z, path, edges=None):
    """Draw the magnetic energy density per cell with a capped colour scale."""
    r_edges, z_edges = _uniform_edges(n_r, n_z) if edges is None else edges
    rr, zz = np.meshgrid(np.asarray(r_edges), np.asarray(z_edges), indexing="ij")
    w = np.asarray(w_cell)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.pcolormesh(rr, zz, w, shading="flat", cmap="plasma", vmin=0.0, vmax=_colour_ceiling(w))
    fig.colorbar(im, ax=ax, label=r"$w$ [J/m$^3$]", extend="max")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"Magnetic energy density $w = |B|^2 / (2\mu)$  (1 A, 1 turn)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- ONE BACKEND ----


def run_backend(
    solver_options, out_subdir, prob, fill, f_core, n_r, n_z, mesh_size, mur, save_plots=True
):
    """Solve with one backend (one warm-up, median of 3 timed solves), post-process and plot.

    Returns:
        Dict with ``l``, ``l_link``, ``w_field``, ``pct_core``, ``pct_air`` and
        the phase times ``assemble_s``, ``solve_s``, ``warmup_s``,
        ``solves_s``, ``post_s``, ``plot_s``, ``backend_s``.
    """
    t_backend = time.perf_counter()
    prob.set_params(fill)
    t0 = time.perf_counter()
    slv.solver(prob, solver_options, return_timing=True)
    warmup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = []
    sol = None
    for _ in range(3):
        sol, timing = slv.solver(prob, solver_options, return_timing=True)
        samples.append(timing)
    solves_s = time.perf_counter() - t0
    timing = {k: statistics.median([s[k] for s in samples]) for k in samples[0]}

    t0 = time.perf_counter()
    l_sim = float(prob.compute_inductance(sol))
    l_link = linkage_inductance(prob, sol)
    psi_grid = psi_to_grid(sol[0], n_r, n_z)
    edges = grid_edges(prob, n_r, n_z)
    b_mag_node = nodal_bfield(psi_grid, edges)
    e_core, w_field = field_energy(prob, sol, fill)
    pct_core = 100.0 * e_core / w_field if w_field > 0 else 0.0
    pct_air = 100.0 - pct_core
    post_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if save_plots:
        os.makedirs(out_subdir, exist_ok=True)
        j_cell = np.mean(np.asarray(prob.coil_current_density()), axis=1)
        plot_geometry(prob, n_r, n_z, mesh_size, os.path.join(out_subdir, "geometry.png"))
        plot_geometry(
            prob, n_r, n_z, mesh_size, os.path.join(out_subdir, "geometry_mesh.png"), show_mesh=True
        )
        plot_psi(psi_grid, os.path.join(out_subdir, "psi.png"), edges=edges)
        plot_aphi(psi_grid, os.path.join(out_subdir, "aphi.png"), edges=edges)
        plot_material_and_current(
            f_core,
            j_cell,
            mur,
            n_r,
            n_z,
            os.path.join(out_subdir, "material_current.png"),
            edges=edges,
        )
        plot_bfield(
            b_mag_node,
            n_r,
            n_z,
            os.path.join(out_subdir, "bfield.png"),
            edges=edges,
            psi_grid=psi_grid,
            note="1 A, 1 turn",
        )
        w_cell = cell_energy_density(b_mag_node, cell_permeability(f_core, mur, n_r, n_z))
        plot_energy_density(
            w_cell, n_r, n_z, os.path.join(out_subdir, "energy_density.png"), edges=edges
        )
    plot_s = time.perf_counter() - t0

    return {
        "l": l_sim,
        "l_link": l_link,
        "w_field": w_field,
        "pct_core": pct_core,
        "pct_air": pct_air,
        "assemble_s": timing["local_assembly"] + timing["global_matrix"],
        "solve_s": timing["linear"],
        "warmup_s": warmup_s,
        "solves_s": solves_s,
        "post_s": post_s,
        "plot_s": plot_s,
        "backend_s": time.perf_counter() - t_backend,
    }


# ---- REPORT ----


def print_experiment_header(cfg):
    """Print the run configuration."""
    print("=== Experiment configuration ===")
    print(f"Core:        {cfg.core_name}")
    print(f"Mesh size:   {cfg.mesh_size * 1000:.3f} mm")
    print(f"Gap size:    {cfg.gap_size * 1000:.3f} mm")
    print(f"Yoke:        {'tapered (h=ct/r)' if cfg.yoke_taper else 'flat'}")
    print(f"Solver:      {solver_label(cfg)}")
    print(f"Reference:   {backend_label(cfg.reference_solver)}")
    print(
        f"Backends:    {' vs '.join(backend_label(b) for b in cfg.backends)}"
        "  (the same solver, one library each; feax runs jax.scipy's CG)"
    )
    print()


def print_material_load(material, core_spec):
    """Print the Steinmetz coefficients, or the permeability when none are on file."""
    stein = load_steinmetz(material)
    if stein is not None:
        k, alpha, beta, fit_mode = stein
        print(
            f"Loaded Steinmetz coefficients for {material} at {_steinmetz_temp}C: "
            f"k={k:.6g}, alpha={alpha:.6g}, beta={beta:.6g}, fit_mode={fit_mode}"
        )
    else:
        print(
            f"Loaded core model for {core_spec['name']} | {material} "
            f"(ur={core_spec['params']['mur']})"
        )


def print_reference(ref, prob, f_core, n_r, n_z, mesh_size, core_spec):
    """Print the inductance, energy split, source check and AL comparison of one backend."""
    params = core_spec["params"]

    print(f"FEM matrix assembly time: {ref['assemble_s']:.2f} s")
    print(f"FEM solve time: {ref['solve_s']:.2f} s")
    print(
        f"1-turn base L (energy): {ref['l'] * 1e6:.2f} µH | (linkage): {ref['l_link'] * 1e6:.2f} µH"
    )
    print()
    print("Scaled (N_p=1, N_s=1):")
    print(f"L (energy): {ref['l'] * 1e6:.2f} µH | (linkage): {ref['l_link'] * 1e6:.2f} µH")
    print(
        f"Energy in core: {ref['pct_core']:.2f}% | Energy in air/gap/leakage: {ref['pct_air']:.2f}%"
    )

    # A conforming grid has cells of several sizes.
    r_nodes, z_nodes = grid_edges(prob, n_r, n_z)
    cell_area = np.diff(r_nodes)[None, :] * np.diff(z_nodes)[:, None]
    core_area_mm2 = float(np.sum(f_core * cell_area)) * 1e6
    at_base = source_ampere_turns(prob)
    at_expected = 2.0 * prob.turns * prob.current
    w_source = 0.5 * ref["l"] * prob.current**2
    print()
    print("--- Geometry & Source Verification (1-turn base) ---")
    print(
        f"Gap Size:           {params.get('gap_size', 0.0) * 1000:.3f} mm "
        f"(x{params.get('gap_number', 0)})"
    )
    print(f"Mesh Size:          {mesh_size * 1000:.3f} mm")
    print(f"Simulated Core Area (approx): {core_area_mm2:.2f} mm²")
    print(f"Total Simulated AT (base): {at_base:.4f} A-turns")
    print(f"Expected AT (base):        {at_expected:.4f} A-turns")
    print(f"Total energy (source method): {w_source:.6e} J")
    print(f"Total energy (field method):  {ref['w_field']:.6e} J")
    print("--------------------------------------")
    print()

    al_sim = ref["l"] * 1e9
    l_an = analytical_inductance(core_spec)
    al_ds = params.get("al", 0.0)
    print(f"Simulated AL: {al_sim:.2f} nH/turn²")
    print(f"Analytical AL: {l_an * 1e9:.2f} nH/turn²")
    print(f"Datasheet AL: {al_ds:.2f} nH/turn²")
    print()


# ---- CG ITERATION COUNT ----


_precond_calls = [0]


def _bump_precond_calls():
    _precond_calls[0] += 1


class _CountingPreconditioner(eqx.Module):
    """A preconditioner that counts its applications: CG iterations + 1."""

    inner: object

    def __call__(self, r):
        jax.debug.callback(_bump_precond_calls)
        return r if self.inner is None else self.inner(r)


def _fresh_options(solver_options):
    """Copy the options without the cached compiled solve, so a new one is traced."""
    opts = copy.copy(solver_options)
    opts["newton"] = dict(opts["newton"])
    linear = dict(opts["newton"]["linear"])
    linear.pop("_jitted_solve", None)
    opts["newton"]["linear"] = linear
    return opts


def cg_iterations(solver_options, prob):
    """Return the CG iteration count of one solve (a separate, untimed run)."""
    original = slv._make_preconditioner
    slv._make_preconditioner = lambda a, o: _CountingPreconditioner(original(a, o))
    try:
        _precond_calls[0] = 0
        sol = slv.solver(prob, _fresh_options(solver_options))
        jax.block_until_ready(sol)
        jax.effects_barrier()
    finally:
        slv._make_preconditioner = original
    return _precond_calls[0] - 1


# ---- FORWARD AND BACKWARD TIMING ----


def inductance_functions(cfg, prob):
    """Return {backend: fill -> L}; each backend solves both the forward and the adjoint."""
    fns = {}
    for name in cfg.backends:
        # Fresh options: run_backend caches a compiled solve on its dict.
        options = build_solver_options(name, cfg)
        solve = slv.ad_wrapper(prob, options, dict(options["newton"]["linear"]), traceable=True)
        fns[name] = lambda f, solve=solve: prob.compute_inductance(solve(f))
    return fns


def gradient_functions(cfg, prob):
    """Return {backend: jitted value_and_grad of L with respect to the fill}."""
    return {
        name: jax.jit(jax.value_and_grad(fn))
        for name, fn in inductance_functions(cfg, prob).items()
    }


def _timed(fn, *args):
    """Return (fn(*args), seconds), timed with pending work drained and GC paused."""
    jax.effects_barrier()
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        return out, time.perf_counter() - t0
    finally:
        gc.enable()


def complete_solve(loss_fns, fill, n_calls=3):
    """Time the forward (jax.vjp) and backward (its pullback) pass of every backend.

    Compiled ahead of time and warmed up once; the backends take turns call
    by call so machine load hits both alike; times are medians of ``n_calls``.

    Returns:
        {backend: {"l", "grad", "compile_s", "forward_s", "backward_s",
        "grad_s"}}: the inductance in henry, the gradient as a flat array,
        and the seconds; ``grad_s`` is forward + backward.
    """
    compiled = {}
    for name, loss in loss_fns.items():
        forward = jax.jit(lambda f, loss=loss: jax.vjp(loss, f))
        backward = jax.jit(lambda pullback, seed: pullback(seed)[0])

        t0 = time.perf_counter()
        forward_c = forward.lower(fill).compile()
        l_sim, pullback = forward_c(fill)
        seed = jnp.ones_like(l_sim)
        backward_c = backward.lower(pullback, seed).compile()
        compile_s = time.perf_counter() - t0
        jax.block_until_ready(backward_c(pullback, seed))
        compiled[name] = (forward_c, backward_c, seed, compile_s)

    runs = {name: {"forward": [], "backward": []} for name in compiled}
    for _ in range(n_calls):
        for name, (forward_c, backward_c, seed, _) in compiled.items():
            (l_sim, pullback), dt = _timed(forward_c, fill)
            runs[name]["forward"].append(dt)
            grad, dt = _timed(backward_c, pullback, seed)
            runs[name]["backward"].append(dt)
            runs[name]["l"], runs[name]["grad"] = l_sim, grad

    out = {}
    for name, r in runs.items():
        out[name] = {
            "l": float(r["l"]),
            "grad": np.asarray(r["grad"], dtype=float).ravel(),
            "compile_s": compiled[name][3],
            "forward_s": statistics.median(r["forward"]),
            "backward_s": statistics.median(r["backward"]),
            "grad_s": statistics.median(f + b for f, b in zip(r["forward"], r["backward"])),
        }
    return out


# ---- TIMING REPORT ----


def print_complete_solve(solves, cfg):
    """Print each backend's complete-solve time and the check that both did the same task."""
    names = [n for n in cfg.backends if n in solves]
    print("=== Complete solve: forward + backward (adjoint) per backend, multigrid ===")
    print("   forward  = assembly + multigrid setup + forward CG + L")
    print("   backward = reassembly + adjoint CG + VJP through the assembly")
    for name in names:
        r = solves[name]
        print(
            f"   {backend_label(name):9s}  L = {r['l'] * 1e9:.9f} nH  "
            f"forward {r['forward_s']:.4f} s | backward {r['backward_s']:.4f} s | "
            f"total {r['grad_s']:.4f} s  (median of 3)"
        )
    if len(names) == 2:
        a, b = names
        g_a, g_b = solves[a]["grad"], solves[b]["grad"]
        cos = float(g_a @ g_b / (np.linalg.norm(g_a) * np.linalg.norm(g_b)))
        ratio = float(np.linalg.norm(g_b) / np.linalg.norm(g_a))
        print(f"   same task: gradient cosine = {cos:.12f} | norm ratio = {ratio:.12f}")
        diff = solves[a]["grad_s"] / solves[b]["grad_s"] - 1.0
        print(f"   {backend_label(a)} / {backend_label(b)} - 1 = {100 * diff:+.1f} %")
    print()


def print_backend_comparison(results, cfg):
    """Print the inductance of every backend side by side."""
    names = [n for n in cfg.backends if n in results]
    if not names:
        return
    print(f"=== Backends: {solver_label(cfg)}, one library each ===")
    for name in names:
        print(f"   {backend_label(name):9s}  L = {results[name]['l'] * 1e9:.9f} nH")
    if len(names) >= 2:
        l0 = results[names[0]]["l"]
        dl = max(abs(results[n]["l"] - l0) for n in names) * 1e9
        print(f"   |max dL| across backends = {dl:.3e} nH")
    print()


def print_timing(all_start, ref, label="Total", stages=None):
    """Print the timing summary; with ``stages``, a full wall-clock breakdown."""
    elapsed = time.time() - all_start
    print("=== Timing ===")
    print(f"FEM matrix assembly time: {ref['assemble_s']:.2f} s")
    print(f"FEM solve time: {ref['solve_s']:.2f} s")

    if stages is not None:
        w = 44
        print()
        print("--- Where the time goes ---")
        accounted = 0.0

        def row(name, seconds, indent=0, pct=True):
            pad = " " * indent
            share = f"{100.0 * seconds / elapsed:5.1f}%" if pct else ""
            print(f"{pad}{name:<{w - indent}} {seconds:8.2f} s  {share}")

        for name, key in (
            ("Imports (jax, matplotlib, feax, lineax)", "import_s"),
            ("Core spec load", "spec_s"),
            ("Mesh + problem build", "build_s"),
            ("Geometry fill (material fractions)", "fill_s"),
        ):
            row(name, stages[key])
            if key != "import_s":  # imports run before all_start
                accounted += stages[key]

        for lbl, res in stages["backends"]:
            row(f"Backend {lbl}", res["backend_s"])
            row("warmup solve (pays the XLA compile)", res["warmup_s"], indent=4)
            row("3 timed solves", res["solves_s"], indent=4)
            print(f"{' ' * 8}{'assembly (median of 3)':<{w - 8}} {res['assemble_s']:8.3f} s")
            print(f"{' ' * 8}{'linear solve (median of 3)':<{w - 8}} {res['solve_s']:8.3f} s")
            row("post-processing (L, B field, energy)", res["post_s"], indent=4)
            row("plots", res["plot_s"], indent=4)
            accounted += res["backend_s"]

        row("Complete solve (compile + forward/backward)", stages["complete_s"])
        accounted += stages["complete_s"]
        row("Verification + report printing", stages["report_s"])
        accounted += stages["report_s"]
        row("Unattributed (printing, GC, teardown)", elapsed - accounted)
        print("-" * (w + 18))

    print(f"{label} execution time: {elapsed:.2f} s")
    if stages is not None:
        print(f"{'Wall clock including imports':<44} {elapsed + stages['import_s']:8.2f} s")
    print()


# ---- MAIN ----


def main():
    """Solve the configured core on every backend and print the report."""
    all_start = time.time()
    cfg = ExperimentConfig()
    if len(sys.argv) > 1:
        cfg.core_name = sys.argv[1]

    stages = {"import_s": _import_s, "backends": []}

    t0 = time.perf_counter()
    core_spec = load_core_spec(cfg.core_name)
    apply_config(core_spec, cfg)
    material = core_spec["params"]["material"]
    stages["spec_s"] = time.perf_counter() - t0

    print_experiment_header(cfg)
    print_material_load(material, core_spec)

    jax_major, jax_minor = jax.__version__.split(".")[:2]
    if (int(jax_major), int(jax_minor)) < (0, 5):
        print(
            "NOTE: jax < 0.5 present — lineax CG runs ~10-15x slower than on "
            "jax >= 0.5 (use the feax_env / jax 0.10 env for fast lineax)."
        )

    t0 = time.perf_counter()
    prob, n_r, n_z, mesh_size = build_problem(core_spec)
    cfg.grid_shape = (n_r + 1, n_z + 1)
    stages["build_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fill, (f_core, _, _) = geometry_fill(prob, n_r, n_z, mesh_size)
    stages["fill_s"] = time.perf_counter() - t0

    results = {}
    for name in cfg.backends:
        results[name] = run_backend(
            build_solver_options(name, cfg),
            os.path.join(cfg.output_dir, f"{name}_forward"),
            prob,
            fill,
            f_core,
            n_r,
            n_z,
            mesh_size,
            core_spec["params"]["mur"],
            save_plots=cfg.save_plots,
        )
        stages["backends"].append((backend_label(name), results[name]))

    ref = results.get(cfg.reference_solver, results[cfg.backends[0]])

    t0 = time.perf_counter()
    print("Complete solve (1 forward + 1 adjoint) on each backend ...", flush=True)
    solves = complete_solve(inductance_functions(cfg, prob), fill)
    stages["complete_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print_reference(ref, prob, f_core, n_r, n_z, mesh_size, core_spec)
    print_backend_comparison(results, cfg)
    stages["report_s"] = time.perf_counter() - t0
    print_timing(all_start, ref, stages=stages)

    if cfg.save_plots:
        print(f"Plots saved under {cfg.output_dir}/")

    print_complete_solve(solves, cfg)


if __name__ == "__main__":
    main()
