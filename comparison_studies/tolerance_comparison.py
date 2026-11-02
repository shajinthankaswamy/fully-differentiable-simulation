"""
CG Tolerance Study
==================
Differentiates the datasheet core with lineax and feax at every CG tolerance
and measures the inductance error, the gradient error and the forward +
adjoint time.  Each backend is compared with its own reference solve at
``tol_study.ref_tol``.  Core, backends and output folder come from ``study``
in solver_comparison.py.

    python comparison_studies/tolerance_comparison.py          # run the sweep
    python comparison_studies/tolerance_comparison.py --plot   # redraw from the saved CSV
"""

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from solver_comparison import append_csv, line_plot, read_csv, setup, study, style

# solver_comparison puts the engine on sys.path, so it is imported first.
from new_fem_engine.run_forward import (
    backend_label,
    build_solver_options,
    cg_iterations,
    complete_solve,
    inductance_functions,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class ToleranceStudy:
    """Settings of the tolerance sweep; edit here to change a run."""

    mesh_mm:    float = 0.1                                  # working mesh          [mm]
    tolerances: tuple = tuple(10.0**-k for k in range(0, 11))  # rtol = atol, 1 to 1e-10
    ref_tol:    float = 1e-12                                # each backend's reference solve
# fmt: on


tol_study = ToleranceStudy()


# ---- HELPERS ----


def rel_error(g, g_ref):
    """Return ||g - g_ref|| / ||g_ref||."""
    return float(np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref))


def with_tol(cfg, tol):
    """Return a copy of ``cfg`` with rtol = atol = ``tol`` on every backend."""
    return dataclasses.replace(
        cfg,
        lineax={**cfg.lineax, "rtol": tol, "atol": tol},
        feax={**cfg.feax, "rtol": tol, "atol": tol},
    )


# ---- SWEEP ----


def reference_point(base, name):
    """Compute backend ``name``'s reference inductance and gradient at ``ref_tol``.

    Args:
        base: (cfg, prob, fill) from setup at ``tol_study.mesh_mm``.
        name: Backend to solve with.

    Returns:
        (reference record, reference gradient as a flat array).

    Raises:
        RuntimeError: if CG stopped at its iteration cap instead of converging.
    """
    cfg, prob, fill = base
    cfg = dataclasses.replace(with_tol(cfg, tol_study.ref_tol), backends=(name,))
    ref = complete_solve(inductance_functions(cfg, prob), jnp.asarray(fill))[name]
    prob.set_params(fill)
    iters = cg_iterations(build_solver_options(name, cfg), prob)
    # feax stops silently at its cap, so convergence is checked here.
    cap = cfg.feax["maxiter"] if name == "feax" else cfg.lineax["max_steps"]
    if iters >= cap:
        raise RuntimeError(f"reference {name} at tol {tol_study.ref_tol:g} hit the {cap}-step cap")
    record = {
        "ref_backend": name,
        "ref_tol": tol_study.ref_tol,
        "mesh_mm": tol_study.mesh_mm,
        "cg_iters": iters,
        "l_nh": ref["l"] * 1e9,
    }
    return record, ref["grad"]


def tolerance_rows(base, tol, refs):
    """Run every backend's complete solve at CG tolerance ``tol``.

    Args:
        refs: {backend: (reference L in nH, reference gradient)}.

    Returns:
        One row per backend: inductance, inductance and gradient error against
        that backend's own reference, and the time of the forward + adjoint solve.
    """
    cfg, prob, fill = base
    solves = complete_solve(inductance_functions(with_tol(cfg, tol), prob), jnp.asarray(fill))
    rows = []
    for name in study.backends:
        s = solves[name]
        l_ref, g_ref = refs[name]
        rows.append(
            {
                "tol": tol,
                "backend": name,
                "mesh_mm": tol_study.mesh_mm,
                "l_nh": s["l"] * 1e9,
                "l_err": abs(s["l"] * 1e9 - l_ref) / l_ref,
                "grad_err": rel_error(s["grad"], g_ref),
                "grad_s": s["grad_s"],
            }
        )
        r = rows[-1]
        print(
            f"{tol:7.0e} {name:<8} {r['l_nh']:11.4f} {r['grad_err']:11.3e} {r['grad_s']:8.3f}",
            flush=True,
        )
    return rows


def tolerance_sweep(tolerances):
    """Build the problem once, compute each backend's reference, then run every tolerance.

    Rows are appended to tolerance_sweep.csv as each tolerance finishes.

    Returns:
        One row per (tolerance, backend).
    """
    sweep_csv = os.path.join(study.out_dir, "tolerance_sweep.csv")
    if os.path.exists(sweep_csv):
        os.remove(sweep_csv)
    cfg, _, prob, *_, fill, _ = setup(tol_study.mesh_mm)
    base = (cfg, prob, fill)

    records, refs = {}, {}
    for name in study.backends:
        records[name], g_ref = reference_point(base, name)
        refs[name] = (records[name]["l_nh"], g_ref)
        print(
            f"reference: {name} at tol {tol_study.ref_tol:.0e}, "
            f"{records[name]['cg_iters']} CG iterations, L = {records[name]['l_nh']:.4f} nH"
        )
    with open(os.path.join(study.out_dir, "tolerance_reference.json"), "w") as f:
        json.dump(records, f, indent=1)
    print(f"{'tol':>7} {'backend':<8} {'L[nH]':>11} {'grad err':>11} {'grad[s]':>8}", flush=True)
    rows = []
    for tol in sorted(tolerances):
        for r in tolerance_rows(base, tol, refs):
            append_csv(sweep_csv, r)
            rows.append(r)
        # Each tolerance compiles its own programs; drop them before the next.
        jax.clear_caches()
    print(f"  saved: {sweep_csv}")
    return rows


# ---- PLOTS ----


def plot_tolerance_sweep(rows):
    """Draw inductance, the inductance and gradient errors and gradient time against the tolerance."""
    tols = sorted({r["tol"] for r in rows}, reverse=True)
    labels = [f"{t:.0e}" for t in tols]
    by = {
        n: [next(r for r in rows if r["backend"] == n and r["tol"] == t) for t in tols]
        for n in study.backends
    }
    tiny = 1e-16  # keeps an exact zero on the log axis
    where = f"({study.core}, {tol_study.mesh_mm} mm)"

    def err(n):
        return [max(r["grad_err"], tiny) for r in by[n]]

    def secs(n):
        return [r["grad_s"] for r in by[n]]

    line_plot(
        labels,
        {backend_label(n): ([r["l_nh"] for r in by[n]], style[n]) for n in study.backends},
        "CG tolerance (rtol = atol)",
        "Inductance [nH]",
        f"Inductance vs CG tolerance {where}",
        os.path.join(study.out_dir, "tolerance_vs_inductance.png"),
        ten_steps=True,
    )
    errors = {}
    l_err = {n: [max(r["l_err"], tiny) for r in by[n]] for n in study.backends}
    for i, n in enumerate(study.backends):
        marker = f"C{i}{style[n][0]}"  # one colour per backend
        errors[f"{backend_label(n)} inductance"] = (l_err[n], marker + "-")
        errors[f"{backend_label(n)} gradient"] = (err(n), marker + ":")
    line_plot(
        labels,
        errors,
        "CG tolerance (rtol = atol)",
        "relative error against the backend's own reference",
        f"Inductance and gradient error vs CG tolerance {where}",
        os.path.join(study.out_dir, "tolerance_vs_error.png"),
        logy=True,
    )
    line_plot(
        labels,
        {backend_label(n): (secs(n), style[n]) for n in study.backends},
        "CG tolerance (rtol = atol)",
        "gradient time, forward + adjoint [s]",
        f"Gradient time vs CG tolerance {where}",
        os.path.join(study.out_dir, "tolerance_vs_time.png"),
    )

    plot_tolerance_choice(tols, labels, l_err, err, secs, where)


def plot_tolerance_choice(tols, labels, l_err, err, secs, where):
    """Draw the gradient error, the inductance error and the time against the tolerance, with the plateau marked.

    The floor is the round-off level of L: the median error of the three
    tightest tolerances, taken for the backend that sits highest.  The
    plateau starts at the loosest tolerance from which every backend's L
    error stays within one decade of that floor.
    """
    floor = max(float(np.median(l_err[n][-3:])) for n in study.backends)
    start = next(  # tols run loose to tight
        i
        for i in range(len(tols))
        if all(e <= 10.0 * floor for n in study.backends for e in l_err[n][i:])
    )
    x = range(len(tols))

    fig, (ax_g, ax_l, ax_t) = plt.subplots(3, 1, figsize=study.fig_size, sharex=True)
    for n in study.backends:
        ax_g.plot(x, err(n), style[n], lw=1.8, ms=4, label=backend_label(n))
        ax_l.plot(x, l_err[n], style[n], lw=1.8, ms=4, label=backend_label(n))
        ax_t.plot(x, secs(n), style[n], lw=1.8, ms=4, label=backend_label(n))
    for ax in (ax_g, ax_l, ax_t):
        ax.axvline(start, color="C2", lw=1.5)
        ax.grid(True, alpha=0.3)
    for ax in (ax_g, ax_l):
        ax.set_yscale("log")
        ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16][: 4 if ax is ax_g else 5])
    ax_g.set_ylabel("gradient\nerror", fontsize=9)
    ax_l.set_ylabel("inductance\nerror", fontsize=9)
    t = {n: secs(n)[start] for n in study.backends}
    fast, slow = sorted(study.backends, key=t.get)
    ax_g.set_title(
        f"CG tolerance {where}: plateau from {tols[start]:.0e}, where "
        f"{backend_label(fast)} is {100 * (t[slow] / t[fast] - 1):.0f} % faster",
        fontsize=10,
    )
    ax_g.legend(loc="upper right", fontsize=7)
    ax_t.set_ylim(bottom=0.0)
    ax_t.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax_t.set_ylabel("time [s]", fontsize=9)
    ax_t.set_xticks(list(x))
    ax_t.set_xticklabels(labels, rotation=45, ha="right")
    ax_t.set_xlabel("CG tolerance (rtol = atol)")
    fig.tight_layout()
    path = os.path.join(study.out_dir, "tolerance_choice.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}  (L round-off floor {floor:.1e}, plateau from {tols[start]:.0e})")


# ---- MAIN ----


def main():
    """Run the tolerance sweep and write its table and figures to ``study.out_dir``.

    With ``--plot`` only the figures are redrawn from the CSV of an earlier
    run.
    """
    t0 = time.time()
    os.makedirs(study.out_dir, exist_ok=True)
    if "--plot" in sys.argv:
        plot_tolerance_sweep(read_csv(os.path.join(study.out_dir, "tolerance_sweep.csv")))
        return

    print(
        f"=== CG tolerance sweep | {study.core} | mesh {tol_study.mesh_mm} mm | "
        f"backends {', '.join(study.backends)} | cg + multigrid ==="
    )
    plot_tolerance_sweep(tolerance_sweep(tol_study.tolerances))
    print(f"\nDone in {time.time() - t0:.0f} s; outputs under {study.out_dir}/")


if __name__ == "__main__":
    main()
