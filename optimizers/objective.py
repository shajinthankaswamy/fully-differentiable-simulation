"""
Shared Objective of the Five Optimizers
=======================================
The core specification, the uniform mesh, the design-to-material map, the
solver options, the loss with its physical conditions, the stopping rule and
the saved record.  The optimizer modules contain only their search algorithm.
"""

import csv
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import config  # noqa: E402

import new_fem_engine.solver as slv  # noqa: E402
from new_fem_engine.generate_mesh import create_mesh, rectangle_mesh  # noqa: E402
from new_fem_engine.geometry import (  # noqa: E402
    compute_fill_laminate,
    define_core_rectangles,
    geometry_settings,
)
from new_fem_engine.problem import MagnetostaticProblem  # noqa: E402
from new_fem_engine.run_forward import (  # noqa: E402
    analytical_inductance,
    cell_energy_density,
    cell_permeability,
    grid_edges,
    nodal_bfield,
    plot_aphi,
    plot_bfield,
    plot_energy_density,
    plot_geometry,
    plot_material_and_current,
    plot_psi,
    psi_to_grid,
)

# ---- SOLVER OPTIONS ----


def solver_options_for(backend, grid_shape):
    """Return (forward_options, adjoint_options, traceable) for ``backend``; only lineax is supported."""
    if backend != "lineax":
        raise ValueError(f"Unknown backend {backend!r}; choices: {config.backends}")
    opts = {**config.lineax_opts, "grid_shape": tuple(grid_shape)}
    return ({"newton": {"linear": {"lineax_solver": opts}}}, {"lineax_solver": opts}, True)


# ---- CORE SPEC AND DESIGN VECTOR ----


def load_core_spec(core_name):
    """Load a core's YAML from cores/ and attach a flat ``params`` view."""
    path = os.path.join(config.cores_dir, core_name, f"{core_name}.yaml")
    with open(path) as f:
        spec = yaml.safe_load(f)
    spec["name"] = core_name
    spec["params"] = base_params(spec)
    return spec


def apply_config(core_spec, target_l_nh=None, mesh_size_mm=None, material=None):
    """Return a copy of the spec with target inductance [nH], mesh size [mm] and material set.

    None keeps config.target_l_nh, config.mesh_size_mm and config.material.
    """
    core_spec = dict(core_spec)
    core_spec["target_l"] = (config.target_l_nh if target_l_nh is None else target_l_nh) * 1e-9
    core_spec["mesh_size"] = (config.mesh_size_mm if mesh_size_mm is None else mesh_size_mm) * 1e-3
    core_spec["_material"] = config.material if material is None else material
    return core_spec


def base_params(core_spec):
    """Return the datasheet geometry [m] and material as the engine's parameter dict.

    Adds the derived ``window_width``, ``yoke_thickness`` and
    ``outer_leg_area_ratio`` (outer leg to post area, which design_params
    uses to let the outer leg follow the post).
    """
    geom = core_spec["geometry"]
    ds = core_spec["datasheet"]
    material = core_spec.get("_material", config.material)
    mat = core_spec["additional_data"]["ungapped_data"][material]
    gap_candidates = core_spec["parameters"]["gap_size"]
    default_gap_size = float(min(gap_candidates, key=lambda v: abs(v - 0.001)))
    post_diameter = float(geom["center_post_diameter"])
    leg_inner_diameter = float(geom["leg_inner_diameter"])
    leg_height = float(geom["leg_height"])
    window_height = float(geom["window_height"])
    return {
        "center_post_diameter": post_diameter,
        "leg_inner_diameter": leg_inner_diameter,
        "leg_height": leg_height,
        "window_height": window_height,
        "window_width": (leg_inner_diameter - post_diameter) / 2.0,
        "yoke_thickness": (leg_height - window_height) / 2.0,
        "coil_clearance": float(geom["coil_clearance"]),
        "mur": float(mat["mu_e"]),
        "al": float(mat["AL"]),
        # define_core_rectangles reads a_e as the outer-leg cross-section.
        "a_e": float(ds["A_e"]),
        "outer_leg_area_ratio": float(ds["A_e"]) / (np.pi * (post_diameter / 2.0) ** 2),
        "l_e": float(ds["l_e"]),
        "sigma_l_over_a": float(ds["sigma_l_over_A"]),
        "material": material,
        "gap_size": default_gap_size,
        "gap_number": 1,
        "yoke_taper": True,
    }


def x0_from_spec(core_spec):
    """Return the datasheet geometry as a design vector in metres."""
    p = base_params(core_spec)
    return np.array([float(p[k]) for k in config.vary_keys])


def to_relative(x, core_spec):
    """Map a design in metres to log coordinates v = ln(x / x_datasheet)."""
    return np.log(np.asarray(x, dtype=float) / x0_from_spec(core_spec))


def from_relative(v, core_spec):
    """Map log coordinates to metres, x = x_datasheet * exp(v); one point or an (n, d) population."""
    v = jnp.clip(jnp.asarray(v, dtype=float), -config.log_coord_max, config.log_coord_max)
    return jnp.asarray(x0_from_spec(core_spec)) * jnp.exp(v)


def make_relative_loss(loss_and_l, core_spec):
    """Wrap a metre-space loss so it takes log coordinates, differentiably."""

    def loss_and_l_relative(v):
        return loss_and_l(from_relative(v, core_spec))

    return loss_and_l_relative


def reference_gap(core_spec):
    """Return the gap [m] that puts the datasheet core at the target on the reluctance model.

    Found by bisection above g_min = config.converge_tol_rel * l_e / mur; g_min
    itself when the core is already at or below the target there.
    """
    base = base_params(core_spec)
    target_l = float(core_spec["target_l"])
    g_min = config.converge_tol_rel * base["l_e"] / base["mur"]
    g_max = config.gap_to_window_max * base["window_height"]

    def l_at(gap):
        return analytical_inductance({"params": {**base, "gap_size": gap}})

    if l_at(g_min) <= target_l:
        return g_min
    lo, hi = g_min, g_max
    while l_at(hi) > target_l:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if l_at(mid) > target_l else (lo, mid)
    return 0.5 * (lo + hi)


def reference_design(core_spec):
    """Return the datasheet core gapped to the target as a design vector in metres."""
    x = x0_from_spec(core_spec)
    x[config.vary_keys.index("gap_size")] = reference_gap(core_spec)
    return x


def start_point(core_spec, seed):
    """Return the start design [m] for repeat ``seed``.

    Seed 0 is the datasheet core gapped to the target; seed k >= 1 perturbs
    every variable by up to +-config.start_perturbation from its own random stream.
    """
    x_ref = reference_design(core_spec)
    if int(seed) == 0:
        return x_ref
    rng = np.random.default_rng(config.start_seed_offset + int(seed))
    return x_ref * (
        1.0 + rng.uniform(-config.start_perturbation, config.start_perturbation, size=len(x_ref))
    )


def design_params(x, base):
    """Return the full parameter dict of design ``x`` on top of ``base``; works on floats and tracers.

    The outer-leg area follows the post at the datasheet ratio.
    """
    p = dict(base)
    for i, k in enumerate(config.vary_keys):
        p[k] = x[i]
    p["window_width"] = (p["leg_inner_diameter"] - p["center_post_diameter"]) / 2.0
    p["yoke_thickness"] = (p["leg_height"] - p["window_height"]) / 2.0
    r_post = p["center_post_diameter"] / 2.0
    p["a_e"] = p["outer_leg_area_ratio"] * np.pi * r_post**2
    return p


def params_dict_from_x(x, core_spec):
    """Return design_params for ``x`` with plain floats."""
    return design_params([float(v) for v in x], base_params(core_spec))


# ---- OPERATING POINT AND VOLUME ----


def operating_point(core_spec):
    """Return the rated operating point for the spec's target.

    Keys: ``ni_peak`` and ``ni_rms`` in amp-turns, the amp-turns at which
    the datasheet post runs at config.b_op_t with the target inductance;
    ``energy_j``, the energy they store; ``copper_area_needed``, the
    coil-rectangle area in m^2 that ni_rms needs at config.j_a_per_m2 and config.k_fill;
    ``winding_area`` and ``clearance``, the coil-rectangle area in m^2 and
    the coil clearance in m of the datasheet geometry, which the optimized
    core must house; and ``outer_radius`` and ``half_height`` in m of the
    datasheet geometry, the outline the optimized core must fit inside.
    """
    base = base_params(core_spec)
    a_post_ds = np.pi * (base["center_post_diameter"] / 2.0) ** 2
    target_l = float(core_spec["target_l"])
    ni_peak = config.b_op_t * a_post_ds / target_l
    ni_rms = ni_peak / config.crest_factor
    r_outer, z_top = core_extent(base)
    return {
        "ni_peak": ni_peak,
        "ni_rms": ni_rms,
        "energy_j": 0.5 * target_l * ni_peak**2,
        "copper_area_needed": ni_rms / (config.j_a_per_m2 * config.k_fill),
        "winding_area": float(section_areas(base)["coil"]),
        "clearance": float(base["coil_clearance"]),
        "outer_radius": float(r_outer),
        "half_height": float(z_top),
    }


def core_volume(core_spec, x):
    """Return the ferrite volume of design ``x`` in m^3 (exact, from the core rectangles)."""
    p = params_dict_from_x(x, core_spec)
    rects, mask, _, _ = define_core_rectangles(p)
    r0, z0, r1, z1 = (np.asarray(rects[:, i], dtype=float) for i in range(4))
    v = np.pi * (r1**2 - r0**2) * (z1 - z0)
    return float(np.sum(np.where(np.asarray(mask, dtype=bool), v, 0.0)))


def datasheet_volume(core_spec):
    """Return the datasheet V_e [m^3], the reference every volume is rated against."""
    return float(core_spec["datasheet"]["V_e"])


# ---- MESH AND config.material ----


def build_problem(core_spec):
    """Build the magnetostatic problem on a uniform grid over geometry_settings.fixed_bounds.

    Uniform, so the compiled array shapes stay fixed while the geometry moves.

    Returns:
        (problem, n_r, n_z, mesh_size) with the mesh size in metres.
    """
    mesh_size = float(core_spec["mesh_size"])
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
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
        additional_info=({"params": base_params(core_spec)}, mesh_size),
    )
    return prob, n_r, n_z, mesh_size


def make_geometry_fn(core_spec, n_r, n_z, mesh_size=None):
    """Return a jitted map ``x [m] -> (fill, primary_rect, secondary_rect, volume [m^3])``."""
    base = base_params(core_spec)
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    r_edges = jnp.linspace(rmin, rmax, n_r + 1)
    z_edges = jnp.linspace(zmin, zmax, n_z + 1)

    def geometry_from_x(x):
        p = design_params(x, base)
        rects, mask, prim, sec = define_core_rectangles(p)
        f_area, mu_r, mu_z = compute_fill_laminate(r_edges, z_edges, rects, mask, base["mur"])
        # Engine cells are r-major; the fill grids are z-major.
        cell = jnp.stack([f_area.T.reshape(-1), mu_r.T.reshape(-1), mu_z.T.reshape(-1)], axis=-1)
        fill = jnp.repeat(cell[:, None, :], 4, axis=1)
        vol = jnp.sum(
            jnp.where(
                mask,
                jnp.pi * (rects[:, 2] ** 2 - rects[:, 0] ** 2) * (rects[:, 3] - rects[:, 1]),
                0.0,
            )
        )
        return fill, prim, sec, vol

    return jax.jit(geometry_from_x)


# ---- PHYSICAL CONDITIONS ----


def post_flux_per_at(psi, r_post, z, n_r, n_z):
    """Return the flux through the centre post at height ``z`` per amp-turn: 2 pi psi(r_post, z).

    A diagnostic for report_design; the conditions use the flux linkage l.
    """
    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    psi_grid = jnp.reshape(psi, (n_r + 1, n_z + 1))
    fr = (r_post - rmin) / (rmax - rmin) * n_r
    fz = (z - zmin) / (zmax - zmin) * n_z
    i = jnp.clip(jnp.floor(fr), 0, n_r - 1).astype(jnp.int32)
    j = jnp.clip(jnp.floor(fz), 0, n_z - 1).astype(jnp.int32)
    tr = fr - i
    tz = fz - j
    val = (
        (1 - tr) * (1 - tz) * psi_grid[i, j]
        + tr * (1 - tz) * psi_grid[i + 1, j]
        + (1 - tr) * tz * psi_grid[i, j + 1]
        + tr * tz * psi_grid[i + 1, j + 1]
    )
    return 2.0 * jnp.pi * jnp.abs(val)


def core_extent(p):
    """Return (outer radius, half height) of the core in metres; works on floats and tracers."""
    r_inner = p["leg_inner_diameter"] / 2.0
    r_outer = jnp.sqrt(r_inner**2 + p["a_e"] / (jnp.pi * p.get("phi", 1.0)))
    return r_outer, p["leg_height"] / 2.0


def section_areas(p):
    """Return the cross-sections [m^2] the flux and the copper pass through.

    The tapered yoke has one annular area at every radius.  The coil area is
    (window_width - 3c) by (window_height - 2c) for clearance c.  A closed
    yoke or window reads as a vanishing section, so its conditions fail.
    """
    r_post = p["center_post_diameter"] / 2.0
    c = p["coil_clearance"]
    tiny = 1e-12
    return {
        "post": np.pi * r_post**2,
        "yoke": 2.0 * np.pi * r_post * jnp.maximum(p["yoke_thickness"], tiny),
        "outer_leg": p["a_e"],
        "coil": jnp.maximum(p["window_width"] - 3.0 * c, 0.0)
        * jnp.maximum(p["window_height"] - 2.0 * c, 0.0),
    }


def constraint_ratios(p, l, vol, op, volume_ref):
    """Return the conditions of config.constraint_names as value / limit - 1; works on floats and tracers.

    The core flux per amp-turn is the flux linkage ``l`` of the one-turn
    winding, as a datasheet defines it; the nodal post flux jumps when the
    post's top face crosses a mesh line.
    """
    a = section_areas(p)
    phi = l * op["ni_peak"]
    r_outer, z_top = core_extent(p)
    coil = jnp.maximum(a["coil"], 1e-12)
    return jnp.stack(
        [
            phi / a["post"] / config.b_max_t - 1.0,
            phi / a["yoke"] / config.b_max_t - 1.0,
            phi / a["outer_leg"] / config.b_max_t - 1.0,
            op["copper_area_needed"] / coil - 1.0,
            op["winding_area"] / coil - 1.0,
            op["clearance"] / jnp.maximum(p["coil_clearance"], 1e-12) - 1.0,
            p["gap_size"] / (config.gap_to_window_max * p["window_height"]) - 1.0,
            jnp.maximum(r_outer / op["outer_radius"], z_top / op["half_height"]) - 1.0,
            jnp.maximum(
                r_outer / geometry_settings.fixed_bounds[1],
                z_top / geometry_settings.fixed_bounds[3],
            )
            - 1.0,
            vol / volume_ref - 1.0,
        ]
    )


# ---- LOSS ----


def make_loss_and_l(
    prob, geometry_from_x, target_l, fwd_opts, adj_opts, traceable, core_spec, weight=None
):
    """Return ``loss_and_l(x) -> (loss, l)``, the one objective all five optimizers share.

    ``loss`` is the volume in datasheet units plus the band penalty on the
    inductance and the hinge penalties of the conditions; ``l`` is the
    inductance [H].  ``weight`` defaults to config.constraint_weight.
    """
    fwd = slv.ad_wrapper(prob, fwd_opts, adj_opts, traceable=traceable)
    w = config.constraint_weight if weight is None else float(weight)
    base = base_params(core_spec)
    op = operating_point(core_spec)
    v_ref = datasheet_volume(core_spec)

    def loss_and_l(x):
        fill, prim, sec, vol = geometry_from_x(x)
        sol = fwd((fill, prim, sec))
        l = prob.compute_inductance(sol, prim, sec)
        loss = vol / v_ref + band_penalty(l, target_l)
        if w != 0.0:
            g = constraint_ratios(design_params(x, base), l, vol, op, v_ref)
            loss = loss + w * jnp.sum(jnp.maximum(g + config.constraint_margin, 0.0) ** 2)
        return loss, l

    return loss_and_l


make_loss_fn = make_loss_and_l


def band_penalty(l, target_l):
    """Return the inductance term: one datasheet volume at the edge of the band."""
    return config.band_weight * (l / target_l - 1.0) ** 2


def make_population_evaluator(loss_fn, traceable):
    """Return ``eval_population(designs) -> (losses, l_values)`` for an (n, d) matrix.

    With config.pop_chunk > 1 designs are solved vmapped in padded chunks of that
    size.  A non-finite loss is a solver failure and is returned as +inf.
    """
    chunk = max(int(config.pop_chunk), 1)
    if traceable and chunk > 1:
        batched = jax.jit(jax.vmap(loss_fn))

        def eval_population(designs):
            designs = np.asarray(designs, dtype=float)
            n = len(designs)
            pad = (-n) % chunk
            if pad:
                designs = np.vstack([designs, np.repeat(designs[:1], pad, axis=0)])
            losses, l_values = [], []
            for i in range(0, len(designs), chunk):
                loss_batch, l_batch = batched(jnp.asarray(designs[i : i + chunk]))
                losses.append(np.asarray(loss_batch, dtype=float))
                l_values.append(np.asarray(l_batch, dtype=float))
            losses = np.concatenate(losses)[:n]
            return np.where(np.isfinite(losses), losses, np.inf), np.concatenate(l_values)[:n]

        return eval_population

    single = jax.jit(loss_fn) if traceable else loss_fn

    def eval_population(designs):
        losses = np.empty(len(designs))
        l_values = np.empty(len(designs))
        for i, xi in enumerate(designs):
            loss, l = single(jnp.asarray(xi))
            losses[i] = float(loss)
            l_values[i] = float(l)
        return np.where(np.isfinite(losses), losses, np.inf), l_values

    return eval_population


def setup(core_spec, backend):
    """Build everything an optimizer needs.

    Returns:
        (problem, n_r, n_z, mesh_size, geometry_from_x, loss_and_l, traceable).
    """
    prob, n_r, n_z, mesh_size = build_problem(core_spec)
    geometry_from_x = make_geometry_fn(core_spec, n_r, n_z)
    fwd_opts, adj_opts, traceable = solver_options_for(backend, (n_r + 1, n_z + 1))
    loss_and_l = make_loss_and_l(
        prob,
        geometry_from_x,
        float(core_spec["target_l"]),
        fwd_opts,
        adj_opts,
        traceable,
        core_spec,
    )
    return prob, n_r, n_z, mesh_size, geometry_from_x, loss_and_l, traceable


# ---- DESIGN REPORT ----


def _shape_lines(params_mm, indent="    "):
    """Return the six design variables as aligned text lines."""
    return [f"{indent}{k:22s} = {params_mm[k]:9.4f} mm" for k in config.vary_keys if k in params_mm]


def _check_lines(checks, indent="    "):
    """Return the physical conditions of a finished design as text lines."""
    if not checks:
        return []

    def ok(name):
        return "ok" if checks["within_tol"][name] else "VIOLATED"

    return [
        f"{indent}at {checks['ni_peak_at']:.1f} A-turns peak: "
        f"B post {checks['b_post_t']:.3f} T ({ok('b_post')}), "
        f"yoke {checks['b_yoke_t']:.3f} T ({ok('b_yoke')}), "
        f"outer leg {checks['b_outer_leg_t']:.3f} T ({ok('b_outer_leg')}); "
        f"limit {checks['b_max_t']:.2f} T",
        f"{indent}copper: coil area {checks['coil_area_mm2']:.1f} mm2, needs "
        f"{checks['copper_area_needed_mm2']:.1f} mm2 "
        f"({checks['copper_area_needed_mm2'] * config.k_fill:.1f} mm2 of copper at "
        f"{config.k_fill:g} fill) ({ok('copper_fit')})",
        f"{indent}winding: coil area {checks['coil_area_mm2']:.1f} mm2, datasheet winding "
        f"{checks['winding_area_mm2']:.1f} mm2 ({ok('winding')}); clearance "
        f"{checks['clearance_mm']:.3f} mm, datasheet {checks['datasheet_clearance_mm']:.3f} mm "
        f"({ok('clearance')})",
        f"{indent}fringing: gap / window height = "
        f"{checks['gap_to_window']:.3f}, limit {config.gap_to_window_max:g} "
        f"({ok('fringing')})",
        f"{indent}envelope: outer diameter {2 * checks['outer_radius_mm']:.1f} mm of "
        f"{2 * checks['datasheet_outer_radius_mm']:.1f}, height "
        f"{2 * checks['half_height_mm']:.1f} mm of {2 * checks['datasheet_half_height_mm']:.1f} "
        f"({ok('envelope')})",
        f"{indent}domain: outer radius {checks['outer_radius_mm']:.1f} mm of "
        f"{checks['domain_r_max_mm']:g}, half height {checks['half_height_mm']:.1f} mm of "
        f"{checks['domain_z_max_mm']:g} ({ok('domain')})",
        f"{indent}volume: {checks['volume_mm3']:.0f} mm3 = {checks['volume_ratio']:.3f}x "
        f"datasheet V_e, limit 1 ({ok('volume')})",
    ]


def _print_design(p_opt, l_sim, l_an, target_nh, opt_time, out_dir, checks=None):
    """Print a finished design; used when a module runs on its own (see config.print_design)."""
    print(
        f"  -> L = {l_sim * 1e9:.2f} nH | analytical = {l_an * 1e9:.2f} nH | "
        f"target = {target_nh:.2f} nH | {opt_time:.1f}s"
    )
    print("  optimized geometry:")
    for line in _shape_lines({k: v * 1e3 for k, v in p_opt.items() if isinstance(v, (int, float))}):
        print(line)
    for line in _check_lines(checks):
        print(line)
    print(f"  saved under {out_dir}/")


def design_checks(core_spec, x, l, fem_post_flux_per_at=None):
    """Evaluate the physical conditions of design ``x`` [m] at inductance ``l`` [H] as plain numbers.

    ``met`` is the strict verdict per condition; ``within_tol`` applies the
    band's relative tolerance, as the finish line does.
    """
    p = params_dict_from_x(x, core_spec)
    op = operating_point(core_spec)
    a = {k: float(v) for k, v in section_areas(p).items()}
    vol = core_volume(core_spec, x)
    v_ds = datasheet_volume(core_spec)
    g = np.asarray(constraint_ratios(p, float(l), vol, op, v_ds), dtype=float)
    phi = float(l) * op["ni_peak"]
    r_outer, z_top = core_extent(p)
    return {
        "ni_peak_at": op["ni_peak"],
        "ni_rms_at": op["ni_rms"],
        "b_op_t": config.b_op_t,
        "b_max_t": config.b_max_t,
        "flux_per_at_wb": float(l),
        "fem_post_top_flux_per_at_wb": (
            None if fem_post_flux_per_at is None else float(fem_post_flux_per_at)
        ),
        "b_post_t": phi / a["post"],
        "b_yoke_t": phi / a["yoke"],
        "b_outer_leg_t": phi / a["outer_leg"],
        "post_area_mm2": a["post"] * 1e6,
        "yoke_area_mm2": a["yoke"] * 1e6,
        "outer_leg_area_mm2": a["outer_leg"] * 1e6,
        "coil_area_mm2": a["coil"] * 1e6,
        "copper_area_needed_mm2": op["copper_area_needed"] * 1e6,
        "winding_area_mm2": op["winding_area"] * 1e6,
        "clearance_mm": p["coil_clearance"] * 1e3,
        "datasheet_clearance_mm": op["clearance"] * 1e3,
        "copper_fill_factor": config.k_fill,
        "current_density_a_per_mm2": config.j_a_per_m2 * 1e-6,
        "gap_to_window": p["gap_size"] / p["window_height"],
        "gap_to_window_max": config.gap_to_window_max,
        "volume_mm3": vol * 1e9,
        "volume_ratio": vol / v_ds,
        "outer_radius_mm": float(r_outer) * 1e3,
        "half_height_mm": float(z_top) * 1e3,
        "datasheet_outer_radius_mm": op["outer_radius"] * 1e3,
        "datasheet_half_height_mm": op["half_height"] * 1e3,
        "domain_r_max_mm": geometry_settings.fixed_bounds[1] * 1e3,
        "domain_z_max_mm": geometry_settings.fixed_bounds[3] * 1e3,
        "constraint_margin": config.constraint_margin,
        "ratios": {k: float(v) for k, v in zip(config.constraint_names, g)},
        "met": {k: bool(v <= 0.0) for k, v in zip(config.constraint_names, g)},
        "within_tol": {
            k: bool(v <= config.converge_tol_rel) for k, v in zip(config.constraint_names, g)
        },
        "feasible": bool(np.all(g <= config.converge_tol_rel)),
    }


def report_design(core_spec, prob, n_r, n_z, mesh_size, x_opt, out_dir):
    """Re-solve a finished design from scratch, check it and save its figures.

    The inductance and conditions in the record come from this re-solve, not
    from the search.

    Returns:
        (l_sim [H], checks from design_checks).
    """
    p = params_dict_from_x(x_opt, core_spec)
    rects, mask, prim, sec = define_core_rectangles(p)
    prob.core_rects_padded = rects
    prob.rect_mask = mask
    prob.primary_rect = prim
    prob.secondary_rect = sec
    prob.mur = p["mur"]

    rmin, rmax, zmin, zmax = geometry_settings.fixed_bounds
    r_edges = jnp.linspace(rmin, rmax, n_r + 1)
    z_edges = jnp.linspace(zmin, zmax, n_z + 1)
    f_area, mu_r, mu_z = compute_fill_laminate(r_edges, z_edges, rects, mask, p["mur"])
    cell = jnp.stack([f_area.T.reshape(-1), mu_r.T.reshape(-1), mu_z.T.reshape(-1)], axis=-1)
    fill = jnp.repeat(cell[:, None, :], 4, axis=1)

    prob.set_params(fill)
    report_opts, _, _ = solver_options_for("lineax", (n_r + 1, n_z + 1))
    sol, _ = slv.solver(prob, report_opts, return_timing=True)
    l_sim = float(prob.compute_inductance(sol))
    psi_grid = psi_to_grid(sol[0], n_r, n_z)
    edges = grid_edges(prob, n_r, n_z)
    phi_fem = float(
        post_flux_per_at(
            jnp.asarray(psi_grid),
            p["center_post_diameter"] / 2.0,
            p["window_height"] / 2.0,
            n_r,
            n_z,
        )
    )
    checks = design_checks(core_spec, x_opt, l_sim, phi_fem)

    os.makedirs(out_dir, exist_ok=True)
    plot_geometry(prob, n_r, n_z, mesh_size, os.path.join(out_dir, "geometry.png"))
    plot_geometry(
        prob, n_r, n_z, mesh_size, os.path.join(out_dir, "geometry_mesh.png"), show_mesh=True
    )
    plot_psi(psi_grid, os.path.join(out_dir, "psi.png"), edges=edges)
    plot_aphi(psi_grid, os.path.join(out_dir, "aphi.png"), edges=edges)
    b_mag_node = nodal_bfield(psi_grid, edges)
    plot_bfield(
        b_mag_node,
        n_r,
        n_z,
        os.path.join(out_dir, "bfield.png"),
        edges=edges,
        psi_grid=psi_grid,
        note="1 A, 1 turn",
    )
    # f_area is the (n_z, n_r) z-major grid run_forward's cell helpers expect.
    f_core_grid = np.asarray(f_area)
    j_cell = np.mean(np.asarray(prob.coil_current_density()), axis=1)
    plot_material_and_current(
        f_core_grid,
        j_cell,
        p["mur"],
        n_r,
        n_z,
        os.path.join(out_dir, "material_current.png"),
        edges=edges,
    )
    w_cell = cell_energy_density(b_mag_node, cell_permeability(f_core_grid, p["mur"], n_r, n_z))
    plot_energy_density(w_cell, n_r, n_z, os.path.join(out_dir, "energy_density.png"), edges=edges)

    return l_sim, checks


# ---- FINISH LINE ----


class TargetReached:
    """The finish line, shared by all five optimizers.

    A run has converged when the optimizer's proposal -- the iterate for the
    gradient methods, the best-so-far for the population methods, the
    incumbent for Bayesian optimization -- has satisfied the finish line for
    config.converge_rounds consecutive rounds:

        |l - l_target| <= tol,  every physical condition within tol,

    and over the last config.settle_rounds rounds its volume changed by no more
    than tol: the core has stopped shrinking.  The condition side is read off the loss the
    optimizer already holds: ``loss - v / v_datasheet - band_penalty`` is the
    condition penalty, and a penalty of at most ``weight * (margin + tol)^2``
    guarantees no single condition exceeds tol.  With ``weight == 0`` the
    condition side is dropped.
    """

    def __init__(self, target_l, core_spec=None, tol_nh=None, enabled=None, weight=None):
        """Create the rule for a target inductance ``target_l`` [H].

        ``tol_nh``, ``enabled`` and ``weight`` default to config.converge_tol_rel of
        the target, config.early_stop and config.constraint_weight.
        """
        self.target_nh = float(target_l) * 1e9
        self.tol_nh = config.converge_tol_rel * self.target_nh if tol_nh is None else float(tol_nh)
        self.weight = config.constraint_weight if weight is None else float(weight)
        self.penalty_tol = self.weight * (config.constraint_margin + config.converge_tol_rel) ** 2
        self.core_spec = core_spec
        self.v_ds = None if core_spec is None else datasheet_volume(core_spec)
        self.enabled = config.early_stop if enabled is None else bool(enabled)
        self.rounds = 0
        self.solves = 0
        # Per-round (solves, volume in mm3 or None) for the settling check.
        self.trace = []
        self.best_err_nh = float("inf")
        self.last_err_nh = float("inf")
        self.last_penalty = None
        self.last_volume_mm3 = None
        self.best_on_target_volume_mm3 = None
        self.first_round_in_tol = None
        self.first_round_all_met = None
        self.streak = 0
        self.streak_round = None
        self.streak_solves = None
        self.converged_round = None
        self.converged_solves = None
        self.settled_by_optimizer = False

    def update(self, l, solves=None, x=None, loss=None):
        """Record this round's proposal and return True when the run should stop.

        Without ``loss`` the condition side counts as not met.
        """
        self.rounds += 1
        if solves is not None:
            self.solves = int(solves)
        err = abs(float(l) * 1e9 - self.target_nh)
        self.last_err_nh = err
        if err < self.best_err_nh:
            self.best_err_nh = err
        in_tol = err <= self.tol_nh
        if in_tol and self.first_round_in_tol is None:
            self.first_round_in_tol = self.rounds

        volume = None
        if x is not None and self.core_spec is not None:
            volume = core_volume(self.core_spec, x)
            self.last_volume_mm3 = volume * 1e9
            if in_tol and (
                self.best_on_target_volume_mm3 is None
                or self.last_volume_mm3 < self.best_on_target_volume_mm3
            ):
                self.best_on_target_volume_mm3 = self.last_volume_mm3
        self.trace.append((self.solves, volume))

        if self.weight == 0.0:
            conditions_ok = True
        elif loss is None:
            conditions_ok = False
        else:
            volume_term = 0.0 if volume is None else volume / self.v_ds
            band = float(band_penalty(float(l), self.target_nh * 1e-9))
            self.last_penalty = max(float(loss) - volume_term - band, 0.0)
            conditions_ok = self.last_penalty <= self.penalty_tol

        all_met = in_tol and conditions_ok
        if all_met and self.first_round_all_met is None:
            self.first_round_all_met = self.rounds
        if all_met:
            if self.streak == 0:
                self.streak_round, self.streak_solves = self.rounds, self.solves
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= config.converge_rounds and self.converged_round is None:
            # The volume must have settled over the last config.settle_rounds rounds
            # (all rounds so far in a short run), or the core is still shrinking.
            start = max(0, len(self.trace) - config.settle_rounds)
            v_then, v_now = self.trace[start][1], self.trace[-1][1]
            settled = (
                v_then is None
                or v_now is None
                or abs(v_now - v_then) <= config.converge_tol_rel * v_now
            )
            if settled:
                # Cost is charged up to the later of the streak's start and the window's.
                first = max(self.streak_round, start + 1)
                self.converged_round = first
                self.converged_solves = self.trace[first - 1][0]
        return self.enabled and self.converged_round is not None

    def settle(self):
        """Accept an in-progress streak as converged when the optimizer stops on its own.

        The cost is charged from the first round of the streak after which the
        volume stayed within config.converge_tol_rel of the final one, so a design
        that kept shrinking during the streak is not charged as finished
        before it stopped changing.
        """
        if self.streak > 0 and self.converged_round is None:
            first = self.streak_round
            v_final = self.trace[-1][1]
            if v_final is not None:
                i = len(self.trace) - 1
                while i > first - 1 and self.trace[i - 1][1] is not None:
                    if abs(self.trace[i - 1][1] - v_final) > config.converge_tol_rel * v_final:
                        break
                    i -= 1
                first = max(first, i + 1)
            self.converged_round = first
            self.converged_solves = self.trace[first - 1][0]
            self.settled_by_optimizer = True

    def conditions_text(self):
        """Return the condition and volume sides of the finish line as text for messages."""
        text = f" and the volume settled within {config.converge_tol_rel:g} over {config.settle_rounds} rounds"
        if self.weight == 0.0:
            return text
        have = "?" if self.last_penalty is None else f"{self.last_penalty:.2e}"
        return (
            f", every physical condition within {config.converge_tol_rel:g} "
            f"(penalty <= {self.penalty_tol:.2e}; now {have})" + text
        )

    def stop_message(self):
        """Return one line saying why, or that, the run ended."""
        if self.converged_round is None:
            side = "inside" if self.last_err_nh <= self.tol_nh else "outside"
            return (
                f"not converged; proposal |L - target| = {self.last_err_nh:.4f} nH, "
                f"{side} the {self.tol_nh:.3f} nH tolerance"
                f"{self.conditions_text()}"
            )
        how = (
            f"on the optimizer's own stop after {self.streak} such rounds"
            if self.settled_by_optimizer
            else f"for {config.converge_rounds} consecutive rounds"
        )
        return (
            f"converged: proposal met |L - target| <= {self.tol_nh:.3f} nH"
            f"{self.conditions_text()} {how}, since round {self.converged_round} at "
            f"{self.converged_solves} solves "
            f"(|L - target| now {self.last_err_nh:.4f} nH)"
        )

    def summary(self):
        """Return the convergence record written into result.json."""
        return {
            "early_stop_enabled": self.enabled,
            "converged": self.converged_round is not None,
            "finish_line": (
                "inductance band and settled volume"
                if self.weight == 0.0
                else "inductance band, physical conditions and settled volume"
            ),
            "constraint_weight": self.weight,
            "penalty_tol": self.penalty_tol,
            "last_penalty": self.last_penalty,
            "last_volume_mm3": self.last_volume_mm3,
            "best_on_target_volume_mm3": self.best_on_target_volume_mm3,
            "first_round_all_conditions_met": self.first_round_all_met,
            "confirm_rounds": config.converge_rounds,
            "settle_rounds": config.settle_rounds,
            "converged_round": self.converged_round,
            "converged_solves": self.converged_solves,
            "rounds": self.rounds,
            "solves": self.solves,
            "tol_nh": self.tol_nh,
            "best_abs_error_nh": self.best_err_nh,
            "last_abs_error_nh": self.last_err_nh,
            "first_round_in_tolerance": self.first_round_in_tol,
        }


# ``solves`` counts FEM linear solves and is the one column that compares
# across the five methods; ``evals`` counts designs evaluated.
# ---- RECORDS ----


def write_history_csv(path, history, core_spec):
    """Write the per-round history to CSV, one row per evaluated design.

    Besides config.history_fields, each row carries its design in mm, the
    volume ratio V / V_e and ``feasible``: in the inductance band and every
    condition within the tolerance, the quality a comparison scores on.
    """
    keys = [f"{k}_mm" for k in config.vary_keys]
    tol_nh = config.converge_tol_rel * float(core_spec["target_l"]) * 1e9
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(config.history_fields) + keys + ["volume_ratio", "feasible"])
        for i, h in enumerate(history):
            row = [
                h.get("iteration", i),
                h.get("evals", i + 1),
                h.get("solves", ""),
                repr(float(h["loss"])),
                repr(float(h["l_nh"])),
                "" if h.get("t_s") is None else f"{float(h['t_s']):.6f}",
            ]
            x = h.get("x")
            if x is None or not np.isfinite(float(h["l_nh"])):
                row += [""] * (len(keys) + 2)
            else:
                checks = design_checks(
                    core_spec, np.asarray(x, dtype=float), float(h["l_nh"]) * 1e-9
                )
                on_target = abs(float(h["l_nh"]) - float(core_spec["target_l"]) * 1e9) <= tol_nh
                row += [f"{v * 1e3:.6f}" for v in x]
                row += [f"{checks['volume_ratio']:.6f}", int(on_target and checks["feasible"])]
            w.writerow(row)


def save_run(
    out_dir,
    optimizer,
    backend,
    core_spec,
    n_r,
    n_z,
    mesh_size,
    x_opt,
    history,
    opt_time,
    l_sim,
    l_an,
    settings=None,
    stopper=None,
    seed=None,
    x_start=None,
    checks=None,
):
    """Write history.csv and result.json for one finished run and return the record.

    Cost (``n_evals``, ``n_solves``, ``opt_time_s``) is charged up to the round
    at which the proposal reached the finish line for good; ``reached_target``
    is judged on the re-solved design.  Lengths in metres, inductances in henry.
    """
    os.makedirs(out_dir, exist_ok=True)
    write_history_csv(os.path.join(out_dir, "history.csv"), history, core_spec)

    target_nh = float(core_spec["target_l"]) * 1e9
    p_opt = params_dict_from_x(x_opt, core_spec)
    losses = [float(h["loss"]) for h in history]
    last = history[-1] if history else {}
    conv = stopper.summary() if stopper is not None else {}
    c_solves = conv.get("converged_solves")
    at_conv = last
    if c_solves is not None:
        rows = [h for h in history if h.get("solves") is not None and h["solves"] <= c_solves]
        at_conv = rows[-1] if rows else last
    cost_evals = at_conv.get("evals", len(history))
    cost_solves = c_solves if c_solves is not None else last.get("solves")
    cost_time = (
        float(opt_time) if c_solves is None or at_conv.get("t_s") is None else float(at_conv["t_s"])
    )
    if x_start is None:
        x_start = start_point(core_spec, 0 if seed is None else int(seed))
    x_start = np.asarray(x_start, dtype=float)

    v_opt = core_volume(core_spec, x_opt)
    v_ds = datasheet_volume(core_spec)
    on_target = abs(float(l_sim) * 1e9 - target_nh) <= config.converge_tol_rel * target_nh
    if config.constraint_weight == 0.0:
        feasible = True
    elif checks is None:
        feasible = None
    else:
        feasible = bool(checks.get("feasible"))

    result = {
        "optimizer": optimizer,
        "backend": backend,
        "seed": None if seed is None else int(seed),
        "protocol_version": config.protocol_version,
        "core": core_spec.get("name", core_spec.get("type")),
        "material": core_spec.get("_material", config.material),
        "target_nh": target_nh,
        "mesh_size_mm": float(mesh_size) * 1e3,
        "n_r": int(n_r),
        "n_z": int(n_z),
        "cells": int(n_r) * int(n_z),
        "x_opt_m": {k: float(v) for k, v in zip(config.vary_keys, x_opt)},
        "x_opt_mm": {k: float(v) * 1e3 for k, v in zip(config.vary_keys, x_opt)},
        "x_start_mm": {k: float(v) * 1e3 for k, v in zip(config.vary_keys, x_start)},
        "x_change_mm": {
            k: (float(a) - float(b)) * 1e3 for k, a, b in zip(config.vary_keys, x_opt, x_start)
        },
        "params_opt": {
            k: (v if isinstance(v, (bool, str)) else float(v))
            for k, v in p_opt.items()
            if isinstance(v, (bool, int, float, str))
        },
        "core_volume_mm3": v_opt * 1e9,
        "datasheet_volume_mm3": v_ds * 1e9,
        "volume_reference": "datasheet V_e",
        "volume_ratio": v_opt / v_ds,
        "l_sim_nh": float(l_sim) * 1e9,
        "l_analytical_nh": float(l_an) * 1e9,
        "abs_error_nh": abs(float(l_sim) * 1e9 - target_nh),
        "converge_tol_rel": config.converge_tol_rel,
        "converge_tol_nh": config.converge_tol_rel * target_nh,
        "on_target": bool(on_target),
        "feasible": feasible,
        "reached_target": bool(
            on_target and (feasible is not False) and conv.get("converged", True)
        ),
        "design_checks": checks,
        "best_loss": min(losses) if losses else None,
        "final_loss": losses[-1] if losses else None,
        "n_history_rows": len(history),
        "n_evals": cost_evals,
        "n_solves": cost_solves,
        "opt_time_s": cost_time,
        "run_evals": last.get("evals", len(history)),
        "run_solves": last.get("solves"),
        "run_time_s": float(opt_time),
        "convergence": conv,
        "config": {
            "protocol_version": config.protocol_version,
            "coordinates": "log-relative to the datasheet design; x = x_datasheet * exp(v)",
            "converge_rounds": config.converge_rounds,
            "start_perturbation": config.start_perturbation,
            "vary_keys": list(config.vary_keys),
            "frozen_keys": list(config.frozen_keys),
            "early_stop": config.early_stop,
            "constraint_weight": config.constraint_weight,
            "constraint_margin": config.constraint_margin,
            "constraint_names": list(config.constraint_names),
            "finish_line": (
                "inductance band and settled volume"
                if config.constraint_weight == 0.0
                else "inductance band, physical conditions and settled volume"
            ),
            "operating_point": {
                "b_op_t": config.b_op_t,
                "b_max_t": config.b_max_t,
                "crest_factor": config.crest_factor,
                "j_a_per_mm2": config.j_a_per_m2 * 1e-6,
                "k_fill": config.k_fill,
                **{k: float(v) for k, v in operating_point(core_spec).items()},
            },
            "band_weight": config.band_weight,
            "reference_gap_mm": reference_gap(core_spec) * 1e3,
            "converge_tol_rel": config.converge_tol_rel,
            "converge_tol_nh": config.converge_tol_rel * target_nh,
            "datasheet_design_mm": {
                k: float(v) * 1e3 for k, v in zip(config.vary_keys, x0_from_spec(core_spec))
            },
            "fem_domain_mm": {
                "r_max": geometry_settings.fixed_bounds[1] * 1e3,
                "z_max": geometry_settings.fixed_bounds[3] * 1e3,
            },
            "linear_solver": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in config.lineax_opts.items()
            },
            "optimizer_settings": settings or {},
        },
    }

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    return result
