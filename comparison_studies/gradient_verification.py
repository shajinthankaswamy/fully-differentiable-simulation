"""
Verification of the Adjoint Gradient
====================================
Five checks beyond the Taylor test of gradient_taylor_test.py, each run in its
own process and written to verification.json:

    geometry2    Taylor test at a second geometry (5 mm gap), both backends
    independent  adjoint gradient against JAX / lineax's own differentiation
    regions      Taylor test in directions confined to five regions
    dirichlet    a loss seeded on every Dirichlet DOF must leave the gradient unchanged
    chain        Taylor test of L(x) and of the full optimizer loss through
                 geometry -> fill -> solve, per design variable and direction,
                 at the datasheet start (edges on grid lines) and off the grid

A Taylor rate of 2 means the gradient is the true derivative; 1 means a kink
(the one-sided slopes differ); 0 means a jump.

    python comparison_studies/gradient_verification.py            # run every check
    python comparison_studies/gradient_verification.py regions    # run one check
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

import new_fem_engine.solver as slv  # noqa: E402
from new_fem_engine.run_forward import (  # noqa: E402
    ExperimentConfig,
    apply_config,
    build_problem,
    build_solver_options,
    geometry_fill,
    inductance_functions,
    load_core_spec,
    output_root,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class Study:
    """Settings of the verification; edit here to change a run."""

    core:          str   = "pq_40x40"                             # core YAML under cores/
    backends:      tuple = ("lineax", "feax")
    tol:           float = 1e-10                                  # CG rtol = atol of every solve
    second_gap_mm: float = 5.0                                    # gap of the second geometry
    eps_fill:      tuple = tuple(1e-2 / 2**k for k in range(10))  # steps on the permeabilities
    eps_design:    tuple = tuple(1e-4 / 2**k for k in range(12))  # relative steps on the design
    off_grid_m:    float = 13.7e-6                                # shift of the off-grid start [m]
    floor:         float = 1e-10                                  # rates are read above this remainder
    out_dir:       str   = os.path.join(output_root, "gradient_verification")
# fmt: on


study = Study()


# ---- HELPERS ----


def setup(gap_mm=1.0):
    """Return (cfg, prob, fill) of the datasheet core at ``gap_mm`` on the working mesh."""
    cfg = ExperimentConfig(core_name=study.core, gap_size=gap_mm * 1e-3, backends=study.backends)
    cfg.lineax = {**cfg.lineax, "rtol": study.tol, "atol": study.tol}
    cfg.feax = {**cfg.feax, "rtol": study.tol, "atol": study.tol}
    spec = load_core_spec(study.core)
    apply_config(spec, cfg)
    prob, n_r, n_z, h = build_problem(spec)
    cfg.grid_shape = (n_r + 1, n_z + 1)
    fill, _ = geometry_fill(prob, n_r, n_z, h)
    return cfg, prob, fill


def rate(eps, r):
    """Return the Taylor rate over the smallest steps whose remainder is above the floor."""
    keep = [i for i in range(len(eps)) if r[i] > study.floor][-5:]
    if len(keep) < 3:
        return None
    x, y = np.log([eps[i] for i in keep]), np.log([r[i] for i in keep])
    return round(float(np.polyfit(x, y, 1)[0]), 3)


def taylor(fn, x0, v, eps, compiled=None):
    """Return (r0 rate, r1 rate, r0 at the smallest step) of fn at x0 along v.

    ``compiled`` is a (jit(fn), jit(value_and_grad(fn))) pair to reuse across calls.
    """
    f, vg = compiled or (jax.jit(fn), jax.jit(jax.value_and_grad(fn)))
    f0, g = vg(x0)
    f0, gv = float(f0), float(jnp.sum(g * v))
    r0, r1 = [], []
    for e in eps:
        fe = float(f(x0 + e * v))
        r0.append(abs(fe - f0) / abs(f0))
        r1.append(abs(fe - f0 - e * gv) / abs(f0))
    return rate(eps, r0), rate(eps, r1), r0[-1]


def random_direction(shape, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, shape)


# ---- CHECKS ----


def check_geometry2():
    cfg, prob, fill = setup(study.second_gap_mm)
    out = {}
    for name, loss in inductance_functions(cfg, prob).items():
        out[name] = [
            taylor(loss, fill, random_direction(fill.shape, s), study.eps_fill)[1] for s in range(3)
        ]
    return {"gap_mm": study.second_gap_mm, "r1_rates": out}


def check_independent():
    cfg, prob, fill = setup()
    out = {}
    for name, loss in inductance_functions(cfg, prob).items():
        _, c = slv._resolve_solver_options(slv._with_traceable(build_solver_options(name, cfg)))

        def native(f, c=c):
            # The same solve with the preconditioner frozen; JAX and lineax
            # then differentiate it by their own rules, not the custom adjoint.
            prob.set_params(f)
            dofs = jnp.zeros(prob.num_total_dofs_all_vars)
            res_list, jac = prob.newton_update(prob.unflatten_fn_sol_list(dofs))
            b = -slv.apply_bc_vec(ravel_pytree(res_list)[0], dofs, prob)
            a = slv.get_a_bcoo(prob, jac)
            linear = c.get("linear", {})
            pre = jax.lax.stop_gradient(slv._make_preconditioner(a, slv._backend_options(linear)))
            inc = slv.linear_solver(a, b, jnp.zeros_like(b), linear, pre)
            return prob.compute_inductance(prob.unflatten_fn_sol_list(dofs + inc))

        g_adj = np.asarray(jax.jit(jax.grad(loss))(fill)).ravel()
        g_nat = np.asarray(jax.jit(jax.grad(native))(fill)).ravel()
        out[name] = {
            "relative_difference": float(np.linalg.norm(g_nat - g_adj) / np.linalg.norm(g_adj)),
            "cosine": float(g_nat @ g_adj / (np.linalg.norm(g_nat) * np.linalg.norm(g_adj))),
        }
    return out


def check_regions():
    cfg, prob, fill = setup()
    r_e, z_e = (np.asarray(e) for e in prob._edges)
    n_r, n_z = len(r_e) - 1, len(z_e) - 1
    rc, zc = np.meshgrid(0.5 * (r_e[:-1] + r_e[1:]), 0.5 * (z_e[:-1] + z_e[1:]), indexing="ij")
    rc, zc = rc.ravel(), zc.ravel()
    rects, mask = np.asarray(prob.core_rects_padded), np.asarray(prob.rect_mask)
    post = sorted((r[1], r[3], r[2]) for r, m in zip(rects, mask) if m and r[0] == 0.0 and r[2] > 0)
    gap_lo, gap_hi, r_post = next((a[1], b[0], a[2]) for a, b in zip(post, post[1:]) if b[0] > a[1])
    prim = np.asarray(prob.primary_rect)
    edge = np.zeros((n_r, n_z), bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    regions = {
        "domain edge": edge.ravel(),
        "axis strip r < 0.3 mm": rc < 3e-4,
        "air gap": (rc < r_post) & (zc > gap_lo) & (zc < gap_hi),
        "core": np.asarray(fill)[:, 0, 0] > 0.99,
        "coil window": (rc > prim[0]) & (rc < prim[2]) & (zc > prim[1]) & (zc < prim[3]),
    }
    loss = inductance_functions(cfg, prob)["lineax"]
    out = {}
    for name, m in regions.items():
        # Relative to each cell's own permeability: 1 in air, mur in the core.
        v = random_direction(fill.shape, 0) * m[:, None, None] * np.asarray(fill)
        r0, r1, _ = taylor(loss, fill, v, study.eps_fill)
        out[name] = {"cells": int(m.sum()), "r0_rate": r0, "r1_rate": r1}
    return out


def check_dirichlet():
    cfg, prob, fill = setup()
    bc = np.asarray(slv._bc_dofs(prob))
    opts = build_solver_options("lineax", cfg)
    solve = slv.ad_wrapper(prob, opts, dict(opts["newton"]["linear"]), traceable=True)
    sol = jax.jit(solve)(fill)
    seed = ravel_pytree(jax.grad(prob.compute_inductance)(sol))[0]

    def plain(f):
        return prob.compute_inductance(solve(f))

    def seeded(f):  # equal to L: the solution is zero on every Dirichlet DOF
        s = solve(f)
        return prob.compute_inductance(s) + 1e-3 * jnp.sum(ravel_pytree(s)[0][bc])

    g1 = np.asarray(jax.jit(jax.grad(plain))(fill)).ravel()
    g2 = np.asarray(jax.jit(jax.grad(seeded))(fill)).ravel()
    return {
        "dirichlet_dofs": int(bc.size),
        "seed_nonzero_on_dirichlet": int(np.count_nonzero(np.asarray(seed)[bc])),
        "gradient_difference": float(np.linalg.norm(g2 - g1) / np.linalg.norm(g1)),
        "seeded_r1_rate": taylor(seeded, fill, random_direction(fill.shape, 0), study.eps_fill)[1],
    }


def check_chain():
    import objective as ob

    ob.config.lineax_opts = {**ob.config.lineax_opts, "rtol": study.tol, "atol": study.tol}
    spec = ob.apply_config(ob.load_core_spec(study.core))
    _, _, _, _, _, loss_and_l, _ = ob.setup(spec, "lineax")
    x_ds = np.asarray(ob.x0_from_spec(spec), dtype=float)
    l_of_x = lambda x: loss_and_l(x)[1]  # noqa: E731
    f_rel = ob.make_relative_loss(loss_and_l, spec)
    loss_of_u = lambda u: f_rel(u)[0]  # noqa: E731

    # Compiled once: every call below has the same shapes.
    l_pair = (jax.jit(l_of_x), jax.jit(jax.value_and_grad(l_of_x)))
    loss_pair = (jax.jit(loss_of_u), jax.jit(jax.value_and_grad(loss_of_u)))
    out = {}
    for start, x0 in (("datasheet (on grid)", x_ds), ("off grid", x_ds + study.off_grid_m)):
        u0 = np.asarray(ob.to_relative(x0, spec))
        rows = {}
        for i, key in enumerate(ob.config.vary_keys):
            for sign in (+1, -1):
                e = sign * np.eye(len(x_ds))[i]
                rows[f"{key} {'+' if sign > 0 else '-'}"] = {
                    "inductance": taylor(
                        l_of_x, jnp.asarray(x0), e * x_ds, study.eps_design, l_pair
                    )[1],
                    "loss": taylor(loss_of_u, jnp.asarray(u0), e, study.eps_design, loss_pair)[1],
                }
        out[start] = rows
    return out


checks = {
    "geometry2": check_geometry2,
    "independent": check_independent,
    "regions": check_regions,
    "dirichlet": check_dirichlet,
    "chain": check_chain,
}


# ---- MAIN ----


def main():
    """Run the checks, each in a fresh process, and write verification.json."""
    os.makedirs(study.out_dir, exist_ok=True)
    path = os.path.join(study.out_dir, "verification.json")

    if sys.argv[1:2] == ["--child"]:
        name = sys.argv[2]
        record = json.load(open(path)) if os.path.exists(path) else {}
        record[name] = checks[name]()
        with open(path, "w") as f:
            json.dump(record, f, indent=1)
        print(json.dumps({name: record[name]}, indent=1), flush=True)
        return

    for name in sys.argv[1:] or list(checks):
        print(f"=== {name} ===", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--child", name], check=True)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
