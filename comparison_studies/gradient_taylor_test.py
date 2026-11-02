"""
Taylor Test of the Adjoint Gradient
===================================
For a random direction v and step eps the remainders

    r0 = |L(f + eps v) - L(f)|              shrink like eps    (first order)
    r1 = |L(f + eps v) - L(f) - eps g.v|    shrink like eps^2  (second order)

and r1 falls as eps^2 only if the adjoint gradient g is the true derivative
(Farrell et al., 2013).  The step is halved from 1e-2 until r1 reaches the
round-off floor of double precision, and the test is repeated at the working
CG tolerance to show where an inexact solve ends the eps^2 trend, following
the presentation of Ghelichkhan et al. (2024, GMD 17, 5057), Figs. 4 and 9.
Run on the datasheet core at the working mesh and gap.

    python comparison_studies/gradient_taylor_test.py          # run the test
    python comparison_studies/gradient_taylor_test.py --plot   # redraw from the saved CSV
"""

import csv
import os
import sys
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402

from new_fem_engine.run_forward import (  # noqa: E402
    ExperimentConfig,
    apply_config,
    backend_label,
    build_problem,
    geometry_fill,
    inductance_functions,
    load_core_spec,
    output_root,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class Study:
    """Settings of the Taylor test; edit here to change a run."""

    core:      str   = "pq_40x40"                        # core YAML under cores/
    # (backend, CG rtol = atol) pairs: both libraries at a tight tolerance,
    # and lineax again at the working tolerance of the optimization runs.
    runs:      tuple = (("lineax", 1e-10), ("feax", 1e-10), ("lineax", 1e-6))
    seeds:     tuple = (0, 1, 2)                         # one random direction per seed
    # Halving from 1e-2 down to the round-off floor; a step is an absolute
    # change of the fill's laminate permeabilities (1 in air, mur in ferrite).
    eps:       tuple = tuple(1e-2 / 2**k for k in range(25))
    # eps ranges of the reported rates: r1 before it reaches round-off,
    # r0 where the first-order term dominates the curvature of L.
    fit_r1:    tuple = (1e-4, 1e-2)
    fit_r0:    tuple = (1e-8, 1e-5)
    plateau_below: float = 1e-7                          # eps under which r1 is round-off
    out_dir:   str   = os.path.join(output_root, "gradient_taylor_test")
    fig_size:  tuple = (3.4, 2.9)                        # each panel           [in]
    font_size: int   = 8
# fmt: on


study = Study()


# ---- TEST ----


def taylor_rows():
    """Return one row per (backend, tolerance, direction, eps) with the two remainders."""
    spec = load_core_spec(study.core)
    rows = []
    for name, tol in study.runs:
        cfg = ExperimentConfig(core_name=study.core, backends=(name,))
        cfg.lineax = {**cfg.lineax, "rtol": tol, "atol": tol}
        cfg.feax = {**cfg.feax, "rtol": tol, "atol": tol}
        apply_config(spec, cfg)
        prob, n_r, n_z, h = build_problem(spec)
        cfg.grid_shape = (n_r + 1, n_z + 1)
        fill, _ = geometry_fill(prob, n_r, n_z, h)
        loss = inductance_functions(cfg, prob)[name]
        forward = jax.jit(loss)
        l0, g = jax.jit(jax.value_and_grad(loss))(fill)
        l0, g = float(l0), np.asarray(g)
        print(f"{backend_label(name)}, CG tol {tol:g}: L = {l0 * 1e9:.6f} nH", flush=True)

        for seed in study.seeds:
            v = np.random.default_rng(seed).uniform(-1.0, 1.0, fill.shape)
            gv = float(np.sum(g * v))
            for eps in study.eps:
                l_eps = float(forward(fill + eps * v))
                rows.append(
                    {
                        "backend": name,
                        "tol": tol,
                        "seed": seed,
                        "eps": eps,
                        "r0": abs(l_eps - l0) / abs(l0),
                        "r1": abs(l_eps - l0 - eps * gv) / abs(l0),
                    }
                )
            mine = rows[-len(study.eps) :]
            print(
                f"   direction {seed}: r0 slope {slope(mine, 'r0'):.3f} | r1 slope {slope(mine, 'r1'):.3f}",
                flush=True,
            )
    return rows


def slope(rows, key):
    """Return the slope of log(key) against log(eps) inside its fit range."""
    lo, hi = study.fit_r1 if key == "r1" else study.fit_r0
    sel = [r for r in rows if lo <= r["eps"] <= hi]
    x = np.log([r["eps"] for r in sel])
    y = np.log([r[key] for r in sel])
    return float(np.polyfit(x, y, 1)[0])


def select(rows, name, tol, seed=None):
    """Return the rows of one run (and one direction), sorted by decreasing eps."""
    mine = [
        r
        for r in rows
        if r["backend"] == name and np.isclose(r["tol"], tol) and (seed is None or r["seed"] == seed)
    ]
    return sorted(mine, key=lambda r: -r["eps"])


# ---- CSV ----


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        for k in ("tol", "eps", "r0", "r1"):
            r[k] = float(r[k])
    return rows


# ---- PLOT ----


def _axes():
    """Return a figure and axes in the style of Ghelichkhan et al. (2024): eps decreasing to the right."""
    plt.rcParams.update({"font.size": study.font_size, "legend.fontsize": study.font_size - 1})
    fig, ax = plt.subplots(figsize=study.fig_size)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlabel(r"step $\varepsilon$")
    ax.set_ylabel(r"remainder / $L$")
    return fig, ax


def plateau(rows):
    """Return the median r1 over the smallest steps: the round-off level of the remainder."""
    tiny = [r["r1"] for r in rows if r["eps"] < study.plateau_below]
    return float(np.median(tiny))


def _guides(ax, rows, key_point, first_order=True):
    """Draw the O(eps) and O(eps^2) guides through the first point and the round-off plateau."""
    e = np.array([max(study.eps), min(study.eps)])
    r0_first, r1_first = key_point
    if first_order:
        ax.plot(e, r0_first * (e / e[0]), "k:", lw=0.9, label=r"$O(\varepsilon)$")
    ax.plot(e, r1_first * (e / e[0]) ** 2, "k--", lw=0.9, label=r"$O(\varepsilon^2)$")
    level = plateau(rows)
    ax.axhline(level, color="0.55", lw=1.0, label=f"round-off plateau ({level:.0e})".replace("e-", "e−"))


def _save(fig, ax, name):
    ax.set_ylim(1e-14, 1e0)
    fig.tight_layout()
    path = os.path.join(study.out_dir, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_libraries(rows):
    """Panel (a): r0 and r1 of both libraries at the tight tolerance, three directions each."""
    fig, ax = _axes()
    style = {"lineax": dict(color="C0", marker="o", ms=3.5), "feax": dict(color="C3", marker="x", ms=3.5, mew=0.9)}
    tight = min(t for _, t in study.runs)
    first = None
    for name in ("lineax", "feax"):
        st = style[name]
        for seed in study.seeds:
            mine = select(rows, name, tight, seed)
            eps = [r["eps"] for r in mine]
            ax.plot(eps, [r["r0"] for r in mine], lw=0, alpha=0.45, **st)
            ax.plot(eps, [r["r1"] for r in mine], lw=0, **st)
            first = first or (mine[0]["r0"], mine[0]["r1"])
        ax.plot([], [], lw=0, label=backend_label(name), **st)
    _guides(ax, select(rows, "lineax", tight), first)
    ax.text(1e-6, 4e-7, r"$r_0$", fontsize=study.font_size + 1)
    ax.text(3e-4, 3e-10, r"$r_1$", fontsize=study.font_size + 1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    _save(fig, ax, "taylor_test_libraries.png")


def plot_tolerance(rows):
    """Panel (b): r1 of lineax at the tight and at the working CG tolerance."""
    fig, ax = _axes()
    tols = sorted({t for n, t in study.runs if n == "lineax"})
    styles = (dict(color="C0", marker="o", ms=3.5), dict(color="C1", marker="x", ms=3.5, mew=0.9))
    first = None
    for tol, st in zip(tols, styles):
        for seed in study.seeds:
            mine = select(rows, "lineax", tol, seed)
            ax.plot([r["eps"] for r in mine], [r["r1"] for r in mine], lw=0, **st)
            first = first or (mine[0]["r0"], mine[0]["r1"])
        ax.plot([], [], lw=0, label=f"CG tolerance {tol:.0e}".replace("e-0", "e−").replace("e-", "e−"), **st)
    _guides(ax, select(rows, "lineax", min(tols)), first, first_order=False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    _save(fig, ax, "taylor_test_tolerance.png")


def report(rows):
    """Print the slopes and the eps below which each run leaves the eps^2 trend."""
    for name, tol in study.runs:
        s1, s0, leave = [], [], []
        for seed in study.seeds:
            mine = select(rows, name, tol, seed)
            s1.append(slope(mine, "r1"))
            s0.append(slope(mine, "r0"))
            ref = mine[0]["r1"]
            off = [r["eps"] for r in mine if r["r1"] > 3.0 * ref * (r["eps"] / mine[0]["eps"]) ** 2]
            leave.append(max(off) if off else 0.0)
        print(
            f"{backend_label(name)}, tol {tol:g}: r1 rate {min(s1):.3f}-{max(s1):.3f}, "
            f"r0 rate {min(s0):.3f}-{max(s0):.3f}, leaves eps^2 below eps = {max(leave):.1e}"
        )


# ---- MAIN ----


def main():
    """Run the Taylor test and write its CSV and figures to ``study.out_dir``."""
    t0 = time.time()
    os.makedirs(study.out_dir, exist_ok=True)
    csv_path = os.path.join(study.out_dir, "taylor_test.csv")

    if "--plot" not in sys.argv:
        print(
            f"=== Taylor test | {study.core} | runs {study.runs} | "
            f"eps {min(study.eps):.2g}-{max(study.eps):.2g} ==="
        )
        write_csv(csv_path, taylor_rows())
    rows = read_csv(csv_path)
    report(rows)
    plot_libraries(rows)
    plot_tolerance(rows)
    print(f"\nDone in {time.time() - t0:.0f} s; outputs under {study.out_dir}/")


if __name__ == "__main__":
    main()
