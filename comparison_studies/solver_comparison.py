"""
Mesh Study of One Core
======================
Solves and differentiates the datasheet core with lineax and feax at every
mesh size in ``study.mesh_mm``, records the inductance, the solve time and the
gradient time, and fits L(h) = L0 + C h^p.  tolerance_comparison.py reuses the
helpers here.

    python comparison_studies/solver_comparison.py          # run the sweep
    python comparison_studies/solver_comparison.py --plot   # redraw from the saved CSV
"""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from new_fem_engine.run_forward import (  # noqa: E402
    ExperimentConfig,
    apply_config,
    backend_label,
    build_problem,
    build_solver_options,
    geometry_fill,
    gradient_functions,
    load_core_spec,
    output_root,
    run_backend,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class Study:
    """Settings shared by the mesh and tolerance sweeps; edit here to change a run."""

    core:         str   = "pq_40x40"         # core YAML under cores/
    backends:     tuple = ("lineax", "feax")
    mesh_mm:      tuple = (0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05)  # coarse to fine [mm]
    grad_repeats: int   = 3                  # timed gradient calls per point
    out_dir:      str   = os.path.join(output_root, "solver_comparison")
    fig_size:     tuple = (7.5, 5.0)         # every figure [in]
# fmt: on


study = Study()
style = {"lineax": "s-", "feax": "^--"}


# ---- HELPERS ----


def setup(mesh_mm, tol=None):
    """Build the core problem at ``mesh_mm``; return (cfg, spec, prob, n_r, n_z, h, fill, f_core).

    ``tol`` sets rtol = atol on every backend; None keeps the defaults.
    """
    cfg = ExperimentConfig(core_name=study.core, mesh_size=mesh_mm * 1e-3)
    cfg.backends = study.backends
    if tol is not None:
        cfg.lineax = {**cfg.lineax, "rtol": tol, "atol": tol}
        cfg.feax = {**cfg.feax, "rtol": tol, "atol": tol}
    spec = load_core_spec(study.core)
    apply_config(spec, cfg)
    prob, n_r, n_z, h = build_problem(spec)
    cfg.grid_shape = (n_r + 1, n_z + 1)
    fill, (f_core, _, _) = geometry_fill(prob, n_r, n_z, h)
    return cfg, spec, prob, n_r, n_z, h, fill, f_core


def read_csv(path):
    """Read a CSV written by append_csv back into a list of dicts with numbers as floats."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def line_plot(
    x_labels,
    series,
    xlabel,
    ylabel,
    title,
    path,
    logy=False,
    ten_steps=False,
):
    """Plot ``series`` ({label: (ys, fmt)}) over evenly spaced categorical x positions.

    With ``ten_steps`` the y axis is ten equal steps (1 nH unless the data
    need more) centred on the data; otherwise a linear axis starts at 0.
    """
    fig, ax = plt.subplots(figsize=study.fig_size)
    x = range(len(x_labels))
    for label, (ys, fmt) in series.items():
        ax.plot(x, ys, fmt, lw=1.8, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 6 else 0, ha="right")
    if logy:
        ax.set_yscale("log")
    elif ten_steps:
        vals = [v for ys, _ in series.values() for v in ys]
        step = max(1.0, math.ceil((max(vals) - min(vals)) / 10.0))
        centre = round(0.5 * (max(vals) + min(vals)) / step) * step
        ax.set_ylim(centre - 5 * step, centre + 5 * step)
        ax.set_yticks(np.arange(centre - 5 * step, centre + 5.5 * step, step))
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def gradient_time(grad_fn, fill_j):
    """Return (median gradient seconds, gradient as a flat array) for one backend."""
    jax.block_until_ready(grad_fn(fill_j))
    times = []
    for _ in range(study.grad_repeats):
        t0 = time.perf_counter()
        _, grad = grad_fn(fill_j)
        grad.block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), np.asarray(grad, dtype=float).ravel()


def run_points(script, flag, values, csv_path):
    """Run ``script flag value backend`` for every point in its own process, appending each row to ``csv_path``.

    A fresh process per point, because JAX on CPU keeps its peak memory until
    the process exits and the fine meshes would crash WSL in one process.

    Returns:
        The rows in the order they ran.
    """
    if os.path.exists(csv_path):
        os.remove(csv_path)
    rows = []
    with tempfile.TemporaryDirectory() as work_dir:
        for value in values:
            for name in study.backends:
                rows.append(run_child(script, [flag, repr(value), name], work_dir))
                append_csv(csv_path, rows[-1])
    print(f"  saved: {csv_path}")
    return rows


def run_child(script, args, work_dir):
    """Run ``script *args work_dir part`` in a fresh process and return the JSON it wrote to part."""
    part = os.path.join(work_dir, "part.json")
    subprocess.run([sys.executable, os.path.abspath(script), *args, work_dir, part], check=True)
    with open(part) as f:
        return json.load(f)


def append_csv(path, row):
    """Append one dict as a CSV row, writing the header if the file is new."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---- MESH SWEEP ----


def mesh_row(mm, name):
    """Return one row with the solve time, gradient time and inductance of one backend at one mesh size."""
    cfg, spec, prob, n_r, n_z, h, fill, f_core = setup(mm)
    res = run_backend(
        build_solver_options(name, cfg),
        None,
        prob,
        fill,
        f_core,
        n_r,
        n_z,
        h,
        spec["params"]["mur"],
        save_plots=False,
    )
    grad_s, _ = gradient_time(gradient_functions(cfg, prob)[name], jnp.asarray(fill))
    r = {
        "mesh_mm": mm,
        "cells": n_r * n_z,
        "backend": name,
        "assemble_s": res["assemble_s"],
        "solve_s": res["solve_s"],
        "grad_s": grad_s,
        "l_nh": res["l"] * 1e9,
    }
    print(
        f"{mm:9.2f} {r['cells']:8d} {name:<10} {r['assemble_s']:12.4f} "
        f"{r['solve_s']:9.4f} {r['grad_s']:9.4f} {r['l_nh']:10.3f}",
        flush=True,
    )
    return r


def mesh_sweep(mesh_mm):
    """Run mesh_row at every (mesh size, backend), saving each row to sweep.csv as it finishes."""
    print(
        f"{'mesh[mm]':>9} {'cells':>8} {'backend':<10} {'assemble[s]':>12} "
        f"{'solve[s]':>9} {'grad[s]':>9} {'L[nH]':>10}",
        flush=True,
    )
    return run_points(__file__, "--mesh", mesh_mm, os.path.join(study.out_dir, "sweep.csv"))


def convergence_fit(hs, ls):
    """Fit L(h) = L0 + C h^p by scanning p; return (p, l0)."""
    best = None
    for p in np.arange(0.3, 3.005, 0.005):
        design = np.stack([np.ones_like(hs), hs**p], axis=1)
        coef, *_ = np.linalg.lstsq(design, ls, rcond=None)
        resid = float(np.sum((design @ coef - ls) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(p), float(coef[0]))
    return best[1], best[2]


def report_mesh_sweep(rows):
    """Save the convergence fit of the mesh sweep, then draw its figures."""
    ref = [r for r in rows if r["backend"] == study.backends[0]]
    hs, ls = np.array([r["mesh_mm"] for r in ref]), np.array([r["l_nh"] for r in ref])
    p, l0 = convergence_fit(hs, ls)
    finest = int(np.argmin(hs))
    fit = {
        "backend": study.backends[0],
        "order_p": p,
        "l0_nh": l0,
        "finest_mm": float(hs[finest]),
        "finest_l_nh": float(ls[finest]),
        "finest_error_nh": float(abs(l0 - ls[finest])),
    }
    with open(os.path.join(study.out_dir, "convergence_fit.json"), "w") as f:
        json.dump(fit, f, indent=1)
    print(
        f"  fit L(h) = L0 + C h^p: p = {p:.2f}, L0 = {l0:.4f} nH, finest {hs[finest]:g} mm is {fit['finest_error_nh']:.4f} nH from L0"
    )
    plot_mesh_sweep(rows)


def plot_mesh_sweep(rows):
    """Draw inductance against mesh size."""
    meshes = sorted({r["mesh_mm"] for r in rows})
    labels = [f"{m:g}" for m in meshes]
    by = {
        name: [next(r for r in rows if r["backend"] == name and r["mesh_mm"] == m) for m in meshes]
        for name in study.backends
    }
    line_plot(
        labels,
        {backend_label(n): ([r["l_nh"] for r in by[n]], style[n]) for n in study.backends},
        "Mesh size [mm]",
        "Inductance [nH]",
        f"Inductance vs mesh size ({study.core})",
        os.path.join(study.out_dir, "mesh_vs_inductance.png"),
        ten_steps=True,
    )


# ---- MAIN ----


def main():
    """Run the mesh sweep and write its table and figures to ``study.out_dir``.

    With ``--plot`` only the figures are redrawn from the CSV of an earlier
    run.  ``--mesh MM BACKEND WORK_DIR PART`` runs one point and writes its row
    to PART; the sweep starts these child processes.
    """
    if sys.argv[1:2] == ["--mesh"]:
        value, name, _work_dir, part = sys.argv[2:6]
        with open(part, "w") as f:
            json.dump(mesh_row(float(value), name), f)
        return

    t0 = time.time()
    os.makedirs(study.out_dir, exist_ok=True)

    if "--plot" in sys.argv:
        plot_mesh_sweep(read_csv(os.path.join(study.out_dir, "sweep.csv")))
        return

    print(
        f"=== Mesh sweep | {study.core} | backends {', '.join(study.backends)} | {min(study.mesh_mm)}-{max(study.mesh_mm)} mm ==="
    )
    report_mesh_sweep(mesh_sweep(study.mesh_mm))
    print(f"\nDone in {time.time() - t0:.0f} s; outputs under {study.out_dir}/")


if __name__ == "__main__":
    main()
