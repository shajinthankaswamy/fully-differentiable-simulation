"""Geometric multigrid V-cycle preconditioner for the structured (r, z) grid.

The Q1 tangent is a nine-point stencil on the nodal grid, so the hierarchy is
built from array shifts and runs under jax.jit.  Coarse operators are Galerkin
(P^T A P), because the coefficient jumps by mur at the core boundary.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

# ---- GRID TRANSFER ----


def _coarsen_1d(n):
    """Return the 1-D prolongation plan (m, c0, c1, w0, w1) for n fine nodes.

    Coarse node c sits on fine node 2c; odd fine nodes average their two
    coarse neighbours.  The even fine rows form an identity submatrix, so
    P has full column rank and P^T A P stays SPD for any n.
    """
    m = (n + 1) // 2
    f = onp.arange(n)
    c0 = f // 2
    c1 = onp.minimum(c0 + 1, m - 1)
    odd = (f % 2) == 1
    # An even-length axis ends on an odd node with no right-hand neighbour.
    orphan = odd & (c0 + 1 > m - 1)
    interior_odd = odd & ~orphan
    w0 = onp.where(interior_odd, 0.5, 1.0)
    w1 = onp.where(interior_odd, 0.5, 0.0)
    return m, c0, c1, w0, w1


def _make_plan(ny, nx):
    return (_coarsen_1d(ny), _coarsen_1d(nx))


def _prolong(vc, plan):
    (_, c0y, c1y, w0y, w1y), (_, c0x, c1x, w0x, w1x) = plan
    v = np.take(vc, c0y, axis=0) * w0y[:, None] + np.take(vc, c1y, axis=0) * w1y[:, None]
    v = np.take(v, c0x, axis=1) * w0x[None, :] + np.take(v, c1x, axis=1) * w1x[None, :]
    return v


def _restrict(vf, plan):
    """Exactly the transpose of _prolong, which keeps the V-cycle symmetric."""
    (my, c0y, c1y, w0y, w1y), (mx, c0x, c1x, w0x, w1x) = plan
    ny, _ = vf.shape
    t = np.zeros((ny, mx), vf.dtype)
    t = t.at[:, c0x].add(vf * w0x[None, :])
    t = t.at[:, c1x].add(vf * w1x[None, :])
    out = np.zeros((my, mx), vf.dtype)
    out = out.at[c0y].add(t * w0y[:, None])
    out = out.at[c1y].add(t * w1y[:, None])
    return out


# ---- STENCILS ----


def stencil_matvec(stencil, v):
    """Apply a (3, 3, ny, nx) stencil; stencil[dy + 1, dx + 1] multiplies v[y + dy, x + dx]."""
    ny, nx = v.shape
    vp = np.pad(v, 1)
    out = np.zeros_like(v)
    for dy in range(3):
        for dx in range(3):
            out = out + stencil[dy, dx] * vp[dy : dy + ny, dx : dx + nx]
    return out


def _extract_stencil(op, shape):
    """Recover the nine-point stencil of ``op`` with nine probe vectors.

    Probe (a, b) is 1 where y % 3 == a and x % 3 == b, so exactly one
    (dy, dx) offset contributes to each output entry.
    """
    ny, nx = shape
    yy, xx = onp.meshgrid(onp.arange(ny), onp.arange(nx), indexing="ij")
    probes = []
    for a in range(3):
        for b in range(3):
            mask = ((yy % 3 == a) & (xx % 3 == b)).astype(onp.float64)
            probes.append(op(np.asarray(mask)))
    responses = np.stack(probes).reshape(3, 3, ny, nx)

    entries = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            entries.append(responses[(yy + dy) % 3, (xx + dx) % 3, yy, xx])
    return np.stack(entries).reshape(3, 3, ny, nx)


def _stencil_from_entries(data, indices, shape):
    """Scatter BCOO entries of the r-major grid into a (3, 3, ny, nx) stencil.

    Entries at offsets other than the nine neighbours are dropped silently;
    verify_stencil checks against the sparse matvec.
    """
    ny, nx = shape
    rows = indices[:, 0]
    cols = indices[:, 1]
    y, x = rows // nx, rows % nx
    dy = cols // nx - y
    dx = cols % nx - x
    stencil = np.zeros((3, 3, ny, nx), dtype=data.dtype)
    return stencil.at[dy + 1, dx + 1, y, x].add(data)


def stencil_from_bcoo(a_bcoo, shape):
    """Return the stencil of a BCOO matrix on the grid of ``shape``."""
    return _stencil_from_entries(a_bcoo.data, a_bcoo.indices, shape)


def verify_stencil(a_bcoo, shape, seed=0):
    """Return the relative disagreement between the stencil and the sparse matvec."""
    v = np.asarray(onp.random.default_rng(seed).standard_normal(shape[0] * shape[1]))
    ref = a_bcoo @ v
    got = stencil_matvec(stencil_from_bcoo(a_bcoo, shape), v.reshape(shape)).reshape(-1)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


# ---- HIERARCHY ----


def _jacobi_weight(stencil):
    """Damped-Jacobi weight from a Gershgorin bound on lambda_max(D^-1 A)."""
    d = np.abs(stencil[1, 1])
    total = np.sum(np.abs(stencil), axis=(0, 1))
    lam = np.max(total / np.where(d > 0, d, 1.0))
    return 4.0 / (3.0 * np.maximum(lam, 1.0))


def _dense_from_stencil(stencil):
    ny, nx = stencil.shape[2], stencil.shape[3]
    n = ny * nx
    yy, xx = onp.meshgrid(onp.arange(ny), onp.arange(nx), indexing="ij")
    row = (yy * nx + xx).ravel()
    dense = np.zeros((n, n), dtype=stencil.dtype)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y2, x2 = yy + dy, xx + dx
            inside = np.asarray((y2 >= 0) & (y2 < ny) & (x2 >= 0) & (x2 < nx))
            col = (onp.clip(y2, 0, ny - 1) * nx + onp.clip(x2, 0, nx - 1)).ravel()
            dense = dense.at[row, col].add(np.where(inside, stencil[dy + 1, dx + 1], 0.0).ravel())
    return dense


def _plans_for(shape, min_dofs=64, max_levels=12):
    plans = []
    ny, nx = shape
    while len(plans) + 1 < max_levels and ny * nx > min_dofs:
        plan = _make_plan(ny, nx)
        my, mx = plan[0][0], plan[1][0]
        if my * mx >= ny * nx:
            break
        plans.append(plan)
        ny, nx = my, mx
    return plans, (ny, nx)


def _dirichlet_mask(stencil):
    """Return 1 on free nodes and 0 on constrained ones (unit rows with no coupling).

    Prolongation writes corrections onto boundary nodes and Jacobi only
    decays them, so without the mask the cycle returns roundoff there.
    """
    off = np.sum(np.abs(stencil), axis=(0, 1)) - np.abs(stencil[1, 1])
    return (off > 0).astype(stencil.dtype)


def _levels_from_stencil(stencil_fine, plans):
    """Build the Galerkin hierarchy from the fine stencil.

    Returns:
        (stencils, dinvs, omegas, coarse_dense, mask) with one stencil,
        inverse diagonal and Jacobi weight per level, the coarsest operator
        as a dense matrix, and the fine-level Dirichlet mask.
    """
    stencils, dinvs, omegas = [], [], []
    stencil = stencil_fine
    for plan in plans:
        stencils.append(stencil)
        d = stencil[1, 1]
        dinvs.append(np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0))
        omegas.append(_jacobi_weight(stencil))
        my, mx = plan[0][0], plan[1][0]

        def coarse_op(vc, stencil=stencil, plan=plan):
            return _restrict(stencil_matvec(stencil, _prolong(vc, plan)), plan)

        stencil = _extract_stencil(coarse_op, (my, mx))
    return stencils, dinvs, omegas, _dense_from_stencil(stencil), _dirichlet_mask(stencil_fine)


# ---- V-CYCLE ----


def _smooth(stencil, dinv, omega, b, x, sweeps):
    for _ in range(sweeps):
        x = x + omega * dinv * (b - stencil_matvec(stencil, x))
    return x


def _vcycle(levels, plans, i, b, sweeps):
    stencils, dinvs, omegas, coarse, _mask = levels
    if i == len(plans):
        return np.linalg.solve(coarse, b.reshape(-1)).reshape(b.shape)
    stencil, dinv, omega, plan = stencils[i], dinvs[i], omegas[i], plans[i]
    x = _smooth(stencil, dinv, omega, b, np.zeros_like(b), sweeps)
    r = b - stencil_matvec(stencil, x)
    x = x + _prolong(_vcycle(levels, plans, i + 1, _restrict(r, plan), sweeps), plan)
    # Equal pre- and post-smoothing keeps the cycle symmetric for CG.
    return _smooth(stencil, dinv, omega, b, x, sweeps)


# Compiled (setup, cycle) pairs cached by grid geometry: a V-cycle is hundreds
# of small ops, so eager dispatch would cost more than the arithmetic.
_kernels_cache = {}


def _kernels(shape, sweeps, min_dofs):
    key = (shape, sweeps, min_dofs)
    cached = _kernels_cache.get(key)
    if cached is None:
        plans, _ = _plans_for(shape, min_dofs)
        setup = jax.jit(
            lambda data, idx: _levels_from_stencil(_stencil_from_entries(data, idx, shape), plans)
        )

        # Masking input and output keeps the cycle symmetric and returns
        # exact zeros on the constrained DOFs.
        def cycle_fn(lv, r):
            mask = lv[4]
            z = _vcycle(lv, plans, 0, mask * r.reshape(shape), sweeps)
            return (mask * z).reshape(-1)

        cycle = jax.jit(cycle_fn)
        cached = (plans, setup, cycle)
        _kernels_cache[key] = cached
    return cached


class VCycle(eqx.Module):
    """One V-cycle as an equinox module: hierarchy as traced leaves, compiled
    cycle as a static field, so lineax's identity-hashed static does not
    rebuild the CG loop per solve."""

    levels: tuple
    cycle: object = eqx.field(static=True)

    def __call__(self, r):
        return self.cycle(self.levels, r)


def make_preconditioner(a_bcoo, shape, sweeps=3, min_dofs=64):
    """Build the multigrid preconditioner for ``a_bcoo`` on the grid of ``shape``.

    Returns:
        (vcycle, shapes): a callable approximating A^-1 r on flat vectors,
        and the grid shape of every level.
    """
    shape = tuple(shape)
    plans, setup, cycle = _kernels(shape, sweeps, min_dofs)
    levels = setup(a_bcoo.data, a_bcoo.indices)
    shapes = [shape] + [(p[0][0], p[1][0]) for p in plans]
    return VCycle(levels, cycle), shapes
