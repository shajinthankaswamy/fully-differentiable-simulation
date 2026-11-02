"""Core geometry as axis-aligned rectangles in the (r, z) half-plane, and
the material-fill fractions of a structured grid over them."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp

# ---- CONSTANTS ----


# fmt: off
@dataclass(frozen=True)
class GeometrySettings:
    """Fixed constants of the geometry model."""

    max_core_rects: int   = 60                         # rectangle slots of one core
    fixed_bounds:   tuple = (0.0, 0.035, -0.03, 0.03)  # domain (r_min, r_max, z_min, z_max) [m]
    yoke_strips:    int   = 20                         # rectangles of the tapered yoke
# fmt: on


geometry_settings = GeometrySettings()


# ---- CORE RECTANGLES ----


def define_core_rectangles(params):
    """Build the core and coil rectangles of a design.

    The yoke is geometry_settings.yoke_strips rectangles of height h(r) = t r_post / r when
    ``yoke_taper`` is set.  The gap is cut from the post by ``gap_size`` and
    ``gap_number`` or by ``gap_shifts`` and ``gap_widths``.

    Returns:
        (rects_padded, mask, primary_rect, secondary_rect): (geometry_settings.max_core_rects, 4)
        rectangles (r0, z0, r1, z1) with a mask of the used rows, and the two
        coil rectangles.
    """
    r_center_post = params["center_post_diameter"] / 2
    r_inner = params["leg_inner_diameter"] / 2
    leg_height = params["leg_height"] / 2
    window_height = params["window_height"]
    gap_number = int(params["gap_number"])
    coil_clearance = params["coil_clearance"]

    yoke_thickness = leg_height - window_height / 2.0

    a_e = params.get("a_e", jnp.pi * r_center_post**2)
    phi = params.get("phi", 1.0)
    r_outer = jnp.sqrt(r_inner**2 + a_e / (jnp.pi * phi))

    rects = []

    def add_rect(r0, z0, r1, z1):
        rects.append(jnp.array([r0, z0, r1, z1]))

    yoke_taper = bool(params.get("yoke_taper", False))

    add_rect(0.0, leg_height - yoke_thickness, r_center_post, leg_height)
    add_rect(0.0, -leg_height, r_center_post, -leg_height + yoke_thickness)

    r_strips = jnp.linspace(r_center_post, r_outer, geometry_settings.yoke_strips + 1)
    for i in range(geometry_settings.yoke_strips):
        r0 = r_strips[i]
        r1 = r_strips[i + 1]
        if yoke_taper:
            r_mid = 0.5 * (r0 + r1)
            h = yoke_thickness * r_center_post / r_mid
        else:
            h = yoke_thickness
        add_rect(r0, leg_height - yoke_thickness, r1, leg_height - yoke_thickness + h)
        add_rect(r0, -leg_height + yoke_thickness - h, r1, -leg_height + yoke_thickness)

    add_rect(r_inner, -window_height / 2.0, r_outer, window_height / 2.0)

    center_z_bottom = -window_height / 2.0
    center_z_top = window_height / 2.0

    use_gap = params.get("gap_shifts", None) is not None
    gap_shifts = params.get("gap_shifts", jnp.array([]))
    gap_widths = params.get("gap_widths", jnp.array([]))

    if use_gap:
        gap_number = len(gap_shifts)
        r1_solid = jnp.where(gap_number > 0, 0.0, r_center_post)
        add_rect(0.0, center_z_bottom, r1_solid, center_z_top)

        zone_size = (center_z_top - center_z_bottom) / gap_number
        z_start = center_z_bottom

        for k in range(gap_number):
            base_center = center_z_bottom + (k + 0.5) * zone_size
            wk = gap_widths[k]
            max_shift = jnp.maximum((zone_size - wk) / 2.0, 0.0)
            center = base_center + gap_shifts[k] * max_shift

            g0 = center - wk / 2.0
            g1 = center + wk / 2.0

            gap_active = g0 > z_start
            z0_seg = jnp.where(gap_active, z_start, 0.0)
            z1_seg = jnp.where(gap_active, g0, 0.0)
            r1_seg = jnp.where(gap_active, r_center_post, 0.0)
            add_rect(0.0, z0_seg, r1_seg, z1_seg)
            z_start = g1

        last_active = z_start < center_z_top
        z0_last = jnp.where(last_active, z_start, 0.0)
        z1_last = jnp.where(last_active, center_z_top, 0.0)
        r1_last = jnp.where(last_active, r_center_post, 0.0)
        add_rect(0.0, z0_last, r1_last, z1_last)
    else:
        gap_size = params.get("gap_size", 0.0)
        use_gap_size = gap_size > 0

        gap_num_eff = jnp.maximum(jnp.array(gap_number), 1)
        gap_each = gap_size / gap_num_eff
        spacing = (center_z_top - center_z_bottom) / gap_num_eff

        r1_solid = jnp.where(use_gap_size, 0.0, r_center_post)
        add_rect(0.0, center_z_bottom, r1_solid, center_z_top)

        z_start = center_z_bottom
        for k in range(gap_number):
            center = center_z_bottom + (k + 0.5) * spacing
            g0 = center - gap_each / 2.0
            g1 = center + gap_each / 2.0
            gap_active = use_gap_size & (g0 > z_start)
            z0_seg = jnp.where(gap_active, z_start, 0.0)
            z1_seg = jnp.where(gap_active, g0, 0.0)
            r1_seg = jnp.where(gap_active, r_center_post, 0.0)
            add_rect(0.0, z0_seg, r1_seg, z1_seg)
            z_start = g1

        last_active = use_gap_size & (z_start < center_z_top)
        z0_last = jnp.where(last_active, z_start, 0.0)
        z1_last = jnp.where(last_active, center_z_top, 0.0)
        r1_last = jnp.where(last_active, r_center_post, 0.0)
        add_rect(0.0, z0_last, r1_last, z1_last)

    r_coil_inner = r_center_post + coil_clearance
    r_coil_outer = r_inner - coil_clearance
    r_coil_mid = (r_coil_inner + r_coil_outer) / 2
    z_coil_bottom = -window_height / 2.0 + coil_clearance
    z_coil_top = window_height / 2.0 - coil_clearance

    # Radial split with an insulation gap of coil_clearance between the windings.
    primary_rect = jnp.array(
        [r_coil_inner, z_coil_bottom, r_coil_mid - coil_clearance / 2.0, z_coil_top]
    )
    secondary_rect = jnp.array(
        [r_coil_mid + coil_clearance / 2.0, z_coil_bottom, r_coil_outer, z_coil_top]
    )

    n = len(rects)
    padded = jnp.zeros((geometry_settings.max_core_rects, 4))
    for i in range(min(n, geometry_settings.max_core_rects)):
        padded = padded.at[i].set(rects[i])
    mask = jnp.zeros(geometry_settings.max_core_rects, dtype=jnp.bool_)
    mask = mask.at[:n].set(True)

    return padded, mask, primary_rect, secondary_rect


# ---- MATERIAL FILL ----


def intersect_area(cxmin, cxmax, cymin, cymax, rxmin, rymin, rxmax, rymax):
    """Return the overlap area of cell box (cx, cy) and rectangle (rx, ry)."""
    ixmin = jnp.maximum(cxmin, rxmin)
    ixmax = jnp.minimum(cxmax, rxmax)
    iymin = jnp.maximum(cymin, rymin)
    iymax = jnp.minimum(cymax, rymax)
    return jnp.maximum(0.0, ixmax - ixmin) * jnp.maximum(0.0, iymax - iymin)


def compute_fill_fractions(
    xs, ys, core_rects_padded, rect_mask, primary_rect, secondary_rect, mur, mu0=None
):
    """Fill fractions on a uniform grid of cell centres ``xs`` x ``ys``.

    Returns:
        (k_arr, f_prim, f_sec, f_core) as (ny, nx) arrays: the reluctivity
        1/(mu0 mu_eff) of the series-homogenised cell and the coil and core
        area fractions.
    """
    if mu0 is None:
        mu0 = 4 * jnp.pi * 1e-7
    h = xs[1] - xs[0]
    nx = xs.shape[0]
    ny = ys.shape[0]

    xs_2d = jnp.broadcast_to(xs[None, :], (ny, nx))
    ys_2d = jnp.broadcast_to(ys[:, None], (ny, nx))

    cxmin = jnp.maximum(0.0, xs_2d - h / 2)
    cxmax = xs_2d + h / 2
    cymin = ys_2d - h / 2
    cymax = ys_2d + h / 2
    cell_area = (cxmax - cxmin) * (cymax - cymin)

    def rect_fill(rect):
        r0, z0, r1, z1 = rect
        return intersect_area(cxmin, cxmax, cymin, cymax, r0, z0, r1, z1) / cell_area

    f_core = jnp.zeros((ny, nx))
    for i in range(geometry_settings.max_core_rects):
        f_core = f_core + jnp.where(rect_mask[i], rect_fill(core_rects_padded[i]), 0.0)

    f_core = jnp.clip(f_core, 0.0, 1.0)
    f_prim = jnp.clip(rect_fill(primary_rect), 0.0, 1.0)
    f_sec = jnp.clip(rect_fill(secondary_rect), 0.0, 1.0)

    mr_eff = jnp.where(
        f_core > 0.999, mur, jnp.where(f_core < 0.001, 1.0, 1.0 / (f_core / mur + (1.0 - f_core)))
    )
    k_arr = 1.0 / (mu0 * mr_eff)

    return k_arr, f_prim, f_sec, f_core


def conforming_edges(params, bounds, mesh_size):
    """Return (r_edges, z_edges) of a grid with a node on every material interface.

    Each interval between interfaces is divided into cells of roughly
    ``mesh_size``, so the geometry does not shift with the mesh.
    """
    rmin, rmax, zmin, zmax = bounds
    r_cp = params["center_post_diameter"] / 2
    r_inner = params["leg_inner_diameter"] / 2
    a_e = params.get("a_e", jnp.pi * r_cp**2)
    phi = params.get("phi", 1.0)
    r_outer = float(jnp.sqrt(r_inner**2 + a_e / (jnp.pi * phi)))
    clearance = params["coil_clearance"]
    r_coil_inner = r_cp + clearance
    r_coil_outer = r_inner - clearance
    r_coil_mid = 0.5 * (r_coil_inner + r_coil_outer)
    prim_outer = r_coil_mid - clearance / 2.0
    sec_inner = r_coil_mid + clearance / 2.0

    leg_h = params["leg_height"] / 2.0
    window_half = params["window_height"] / 2.0

    r_keys = [
        float(r_cp),
        float(r_coil_inner),
        float(prim_outer),
        float(sec_inner),
        float(r_coil_outer),
        float(r_inner),
        float(r_outer),
    ]
    z_keys = [
        float(-leg_h),
        float(-window_half),
        float(-window_half + clearance),
        float(window_half - clearance),
        float(window_half),
        float(leg_h),
    ]

    yoke_thickness = leg_h - window_half
    taper = bool(params.get("yoke_taper", False))
    r_strips = onp.linspace(float(r_cp), r_outer, geometry_settings.yoke_strips + 1)
    r_keys += [float(v) for v in r_strips]
    for i in range(geometry_settings.yoke_strips):
        r_mid = 0.5 * (r_strips[i] + r_strips[i + 1])
        h = yoke_thickness * float(r_cp) / r_mid if taper else yoke_thickness
        top = window_half + h
        z_keys += [float(top), float(-top)]

    gap_size = float(params.get("gap_size", 0.0))
    gap_number = int(params.get("gap_number", 0))
    if gap_size > 0 and gap_number >= 1:
        gap_each = gap_size / gap_number
        spacing = 2.0 * window_half / gap_number
        for k in range(gap_number):
            center = -window_half + (k + 0.5) * spacing
            z_keys += [center - gap_each / 2.0, center + gap_each / 2.0]

    def edges(lo, hi, keys):
        es = sorted(set([lo, hi] + [float(k) for k in keys if lo <= float(k) <= hi]))
        out = []
        for a, b in zip(es, es[1:]):
            n = max(int(round((b - a) / mesh_size)), 1)
            out.extend(jnp.linspace(a, b, n, endpoint=False))
        out.append(es[-1])
        return jnp.asarray(out)

    return edges(rmin, rmax, r_keys), edges(zmin, zmax, z_keys)


def compute_fill_fractions_grid(
    r_edges, z_edges, core_rects_padded, rect_mask, primary_rect, secondary_rect, mur, mu0=None
):
    """Fill fractions on the cells of an edge-defined grid; see compute_fill_fractions.

    Returns:
        (k_arr, f_prim, f_sec, f_core) as (nz, nr) arrays.
    """
    if mu0 is None:
        mu0 = 4 * jnp.pi * 1e-7
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    dr = r_edges[1:] - r_edges[:-1]
    dz = z_edges[1:] - z_edges[:-1]
    nr, nz = len(rc), len(zc)

    rc_2d = jnp.broadcast_to(rc[None, :], (nz, nr))
    zc_2d = jnp.broadcast_to(zc[:, None], (nz, nr))
    dr_2d = jnp.broadcast_to(dr[None, :], (nz, nr))
    dz_2d = jnp.broadcast_to(dz[:, None], (nz, nr))

    cxmin = rc_2d - dr_2d / 2
    cxmax = rc_2d + dr_2d / 2
    cymin = zc_2d - dz_2d / 2
    cymax = zc_2d + dz_2d / 2
    cell_area = dr_2d * dz_2d

    def rect_fill(rect):
        r0, z0, r1, z1 = rect
        return intersect_area(cxmin, cxmax, cymin, cymax, r0, z0, r1, z1) / cell_area

    f_core = jnp.zeros((nz, nr))
    for i in range(geometry_settings.max_core_rects):
        f_core = f_core + jnp.where(rect_mask[i], rect_fill(core_rects_padded[i]), 0.0)
    f_core = jnp.clip(f_core, 0.0, 1.0)
    f_prim = jnp.clip(rect_fill(primary_rect), 0.0, 1.0)
    f_sec = jnp.clip(rect_fill(secondary_rect), 0.0, 1.0)

    mr_eff = jnp.where(
        f_core > 0.999, mur, jnp.where(f_core < 0.001, 1.0, 1.0 / (f_core / mur + (1.0 - f_core)))
    )
    k_arr = 1.0 / (mu0 * mr_eff)

    return k_arr, f_prim, f_sec, f_core


def compute_fill_laminate(r_edges, z_edges, core_rects_padded, rect_mask, mur, max_rects=4):
    """Return (f_area, mu_r, mu_z) per cell as (nz, nr) arrays.

    f_area is the core area fraction.  mu_r and mu_z are the laminate
    permeabilities for flux along r and z: the cell is cut into slices along
    the other axis, each slice a series path through its ferrite width, and
    the slices act in parallel.  The result is continuous in the rectangle
    positions and, for a cell cut by one rectangle, equals the classical
    laminate rule.  ``max_rects`` bounds the rectangles that may touch one
    cell; the core model reaches three.
    """
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    dr = r_edges[1:] - r_edges[:-1]
    dz = z_edges[1:] - z_edges[:-1]

    cxmin = (rc - dr / 2)[None, :]
    cxmax = (rc + dr / 2)[None, :]
    cymin = (zc - dz / 2)[:, None]
    cymax = (zc + dz / 2)[:, None]

    r0 = core_rects_padded[:, 0][:, None, None]
    z0 = core_rects_padded[:, 1][:, None, None]
    r1 = core_rects_padded[:, 2][:, None, None]
    z1 = core_rects_padded[:, 3][:, None, None]

    ixmin = jnp.maximum(cxmin[None], r0)
    ixmax = jnp.minimum(cxmax[None], r1)
    iymin = jnp.maximum(cymin[None], z0)
    iymax = jnp.minimum(cymax[None], z1)
    w = jnp.maximum(0.0, ixmax - ixmin)
    h = jnp.maximum(0.0, iymax - iymin)
    hit = (rect_mask[:, None, None] > 0) & (w > 0) & (h > 0)

    area = jnp.where(hit, w * h, 0.0)
    f_area = jnp.clip(jnp.sum(area, axis=0) / (dr[None, :] * dz[:, None]), 0.0, 1.0)

    # The rectangles overlapping each cell most; the rest do not touch it.
    _, idx = jax.lax.top_k(jnp.moveaxis(area, 0, -1), max_rects)

    def pick(values, empty):
        full = jnp.where(hit, jnp.broadcast_to(values, hit.shape), empty[None])
        return jnp.take_along_axis(jnp.moveaxis(full, 0, -1), idx, axis=-1)

    zero = jnp.zeros((len(zc), len(rc)))
    lo_z, hi_z = pick(iymin, cymin + zero), pick(iymax, cymin + zero)
    lo_r, hi_r = pick(ixmin, cxmin + zero), pick(ixmax, cxmin + zero)
    width, height = pick(w, zero), pick(h, zero)

    mu_r = _slice_laminate(
        lo_z, hi_z, width, cymin + zero, cymax + zero, dr[None, :], dz[:, None], mur
    )
    mu_z = _slice_laminate(
        lo_r, hi_r, height, cxmin + zero, cxmax + zero, dz[:, None], dr[None, :], mur
    )
    return f_area, mu_r, mu_z


def _slice_laminate(lo, hi, width, c0, c1, cell_width, cell_length, mur):
    """Return the slice-averaged series permeability of the cells, (nz, nr).

    Slices run along the axis where the rectangles span [lo, hi); ``width`` is
    each rectangle's extent across it, ``cell_width`` and ``cell_length`` the
    cell's extents across and along.
    """
    points = jnp.sort(jnp.concatenate([lo, hi, c0[..., None], c1[..., None]], axis=-1), axis=-1)
    mid = 0.5 * (points[..., 1:] + points[..., :-1])
    length = points[..., 1:] - points[..., :-1]
    inside = (lo[..., None, :] <= mid[..., :, None]) & (mid[..., :, None] < hi[..., None, :])
    filled = jnp.sum(jnp.where(inside, width[..., None, :], 0.0), axis=-1)
    frac = jnp.clip(filled / cell_width[..., None], 0.0, 1.0)
    mu = 1.0 / (frac / mur + (1.0 - frac))
    return jnp.sum(length * mu, axis=-1) / cell_length


# ---- COIL CURRENT ----


def compute_current_density(f_prim_map, f_sec_map, area_prim, area_sec, current=1.0, turns=1):
    """Return the azimuthal current density per cell for ``turns`` x ``current``.

    A coil of zero area carries no current, with a finite gradient.
    """
    safe_prim = jnp.where(area_prim > 0, area_prim, 1.0)
    safe_sec = jnp.where(area_sec > 0, area_sec, 1.0)
    j_prim = jnp.where(area_prim > 0, (turns * current / safe_prim) * f_prim_map, 0.0)
    j_sec = jnp.where(area_sec > 0, (turns * current / safe_sec) * f_sec_map, 0.0)
    return j_prim + j_sec


def compute_coil_areas(f_prim_map, f_sec_map, cell_area):
    """Return (primary, secondary) coil cross-section areas from the fill maps."""
    return jnp.sum(f_prim_map * cell_area), jnp.sum(f_sec_map * cell_area)
