"""
Cross-Optimizer Comparison Figures
==================================
Drawn only from output/comparison/<label>/run.json and history.csv, which
main_run_optimizer.py maintains, so the figures can be redrawn without
re-running anything.  An optimizer that crashed leaves no record and is
skipped.

    python optimizers/compare_plots.py                      # default label
    python optimizers/compare_plots.py pq_40x40_target250.0nH_N87_mesh0.100mm
"""

import csv
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config  # noqa: E402

# ---- SETTINGS ----

# fmt: off
fig_size       = (7.5, 5.0)   # every figure                  [in]
fig_panel_size = (16.0, 9.0)  # the six-panel parameter figure [in]
neutral_color  = "#5C6673"    # colour of non-optimizer marks
# fmt: on

opt_colors = {
    "adam": "#20558A",
    "lbfgs": "#4E8FC4",
    "cmaes": "#1D6B4F",
    "ga": "#A2552A",
    "bayesian": "#7A5C99",
}
opt_order = ["adam", "lbfgs", "cmaes", "ga", "bayesian"]
opt_label = {
    "adam": "Adam",
    "lbfgs": "L-BFGS",
    "cmaes": "CMA-ES",
    "ga": "GA",
    "bayesian": "Bayesian",
}

param_label = {
    "leg_inner_diameter": "leg inner dia.",
    "center_post_diameter": "center post dia.",
    "leg_height": "leg height",
    "window_height": "window height",
    "coil_clearance": "coil clearance",
    "gap_size": "gap size",
}

# Scalar record keys carried into the per-seed rows.
record_keys = (
    "target_nh",
    "l_sim_nh",
    "l_analytical_nh",
    "abs_error_nh",
    "converge_tol_nh",
    "core_volume_mm3",
    "datasheet_volume_mm3",
    "volume_ratio",
    "best_loss",
    "n_evals",
    "n_solves",
    "opt_time_s",
    "first_eval_s",
    "mesh_size_mm",
    "cells",
)


# ---- READING THE RECORD ----


def run_path(out_root, core_label):
    """Return the path of the consolidated run.json for a study label."""
    return os.path.join(out_root, "comparison", core_label, "run.json")


def history_path(run_json):
    """Return the path of the history.csv beside a run.json."""
    return os.path.join(os.path.dirname(os.path.abspath(run_json)), "history.csv")


def read_run(path):
    """Read run.json into one flat row per (optimizer, seed).

    Scalar keys are floats or None; the six free parameters are carried as
    ``<key>_start_mm``, ``<key>_opt_mm`` and ``<key>_change_mm``.  Records
    without a protocol stamp count as version 1.
    """
    with open(path) as f:
        doc = json.load(f)

    rows = []
    for entry in doc.get("optimizers", {}).values():
        for rec in entry.get("seeds", []):
            row = {
                "optimizer": entry["optimizer"],
                "backend": entry["backend"],
                "seed": float(rec.get("seed", 0)),
            }
            for k in record_keys:
                v = rec.get(k)
                row[k] = None if v in ("", None) else float(v)
            cfg = entry.get("config") or {}
            row["constraint_weight"] = cfg.get("constraint_weight")
            row["reached_target"] = bool(rec.get("reached_target"))
            row["feasible"] = rec.get("feasible")
            row["protocol_version"] = int(
                rec.get("protocol_version") or cfg.get("protocol_version") or 1
            )
            for k in config.vary_keys:
                for suffix, src in (
                    ("start", "x_start_mm"),
                    ("opt", "x_opt_mm"),
                    ("change", "x_change_mm"),
                ):
                    v = (rec.get(src) or {}).get(k)
                    row[f"{k}_{suffix}_mm"] = None if v is None else float(v)
            rows.append(row)

    rows.sort(
        key=lambda r: (
            opt_order.index(r["optimizer"]) if r["optimizer"] in opt_order else 99,
            r["seed"],
        )
    )
    return rows


def datasheet_geometry(path):
    """Return the datasheet geometry in mm from any seed-0 record, or None."""
    with open(path) as f:
        doc = json.load(f)
    for entry in doc.get("optimizers", {}).values():
        for rec in entry.get("seeds", []):
            if int(rec.get("seed", 0)) == 0 and rec.get("x_start_mm"):
                return {k: float(v) for k, v in rec["x_start_mm"].items()}
    return None


def select_protocol(rows, wanted=None):
    """Keep only the rows made under one protocol version and name the rest.

    Runs under different finish-line rules are not comparable, so mixed
    versions are dropped rather than averaged; the newest version is kept
    unless ``wanted`` is given.
    """
    versions = sorted({int(r.get("protocol_version") or 1) for r in rows})
    if not versions:
        return rows
    keep = max(versions) if wanted is None else int(wanted)
    if len(versions) > 1:
        dropped = sorted(
            {
                f"{r['optimizer']}#seed{int(r.get('seed') or 0)}"
                for r in rows
                if int(r.get("protocol_version") or 1) != keep
            }
        )
        print(f"  ! runs from protocol versions {versions} found in one study.")
        print(
            f"    Plotting version {keep} only.  Re-run under the current "
            f"rules: {', '.join(dropped)}"
        )
    return [r for r in rows if int(r.get("protocol_version") or 1) == keep]


def aggregate(rows):
    """Collapse per-seed rows to one row per optimizer with mean and s.e.m.

    A single run of a deterministic optimizer has s.e.m. 0; a single run of a
    stochastic one has unknown spread and gets None.
    """
    by_opt = {}
    for r in rows:
        by_opt.setdefault(r["optimizer"], []).append(r)

    out = []
    for name, group in by_opt.items():
        merged = {"optimizer": name, "backend": group[0]["backend"], "n_runs": len(group)}
        keys = [k for k in group[0] if k not in ("optimizer", "backend", "seed", "n_runs")]
        for k in keys:
            vals = [g[k] for g in group if g.get(k) is not None]
            if not vals:
                merged[k] = None
                merged[f"{k}_sem"] = None
                continue
            arr = np.asarray(vals, dtype=float)
            merged[k] = float(arr.mean())
            if len(arr) > 1:
                merged[f"{k}_sem"] = float(arr.std(ddof=1) / np.sqrt(len(arr)))
            elif name in config.stochastic:
                merged[f"{k}_sem"] = None
            else:
                merged[f"{k}_sem"] = 0.0
        out.append(merged)

    out.sort(key=lambda r: opt_order.index(r["optimizer"]) if r["optimizer"] in opt_order else 99)
    return out


# ---- BAR CHARTS ----


def _names(rows):
    return [r["optimizer"] for r in rows]


def _colors(rows):
    return [opt_colors.get(n, neutral_color) for n in _names(rows)]


def _labels(rows):
    return [opt_label.get(n, n) for n in _names(rows)]


def _annotate(ax, bars, values, fmt="{:.0f}", fontsize=8):
    """Write each bar's value above (or below, if negative) the bar."""
    for b, v in zip(bars, values):
        if v is None:
            continue
        ax.annotate(
            fmt.format(v),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=fontsize,
            xytext=(0, 2 if v >= 0 else -2),
            textcoords="offset points",
        )


def quality_costs(run_json, keep=None):
    """Return ({optimizer: cost}, target) scored from history.csv on one common task.

    Every optimizer's proposal is its best feasible design so far (in the band,
    every condition within tolerance).  The target is the largest of the final
    best volumes, a core every optimizer delivered; an optimizer's cost is the
    FE solves and wall time until its proposal first reaches it.  Each cost is
    a dict with ``solves``, ``time_s``, ``best`` (its final best volume ratio)
    and the curve ``s``, ``v``.  Returns ({}, None) for records without
    per-design columns, or when an optimizer never found a feasible core.
    """
    path = history_path(run_json)
    if not os.path.isfile(path):
        return {}, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or "feasible" not in rows[0]:
        return {}, None

    curves = {}
    for row in rows:
        opt = row.get("optimizer")
        try:
            seed_i = int(float(row.get("seed") or 0))
        except ValueError:
            seed_i = 0
        if opt not in opt_order or (keep is not None and (opt, seed_i) not in keep):
            continue
        feasible = row.get("feasible") == "1" and row.get("volume_ratio")
        v = float(row["volume_ratio"]) if feasible else np.inf
        c = curves.setdefault(opt, {"s": [], "t": [], "v": []})
        best = min(v, c["v"][-1]) if c["v"] else v
        c["s"].append(float(row.get("solves") or len(c["s"]) + 1))
        c["t"].append(float(row.get("elapsed_s") or "nan"))
        c["v"].append(best)

    finals = [c["v"][-1] for c in curves.values()]
    if not finals or not all(np.isfinite(finals)):
        return {}, None
    target = max(finals)
    costs = {}
    for opt, c in curves.items():
        hit = next((i for i, v in enumerate(c["v"]) if v <= target), None)
        costs[opt] = {
            "solves": None if hit is None else c["s"][hit],
            "time_s": None if hit is None else c["t"][hit],
            "best": c["v"][-1],
            "s": c["s"],
            "v": c["v"],
        }
    return costs, target


def _cost_bars(rows, costs, target, key, fmt, ylabel, title, name, out_dir):
    """Bar chart of each optimizer's cost to the common quality target."""
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(rows))
    values = [costs.get(r["optimizer"], {}).get(key) for r in rows]
    heights = [0.0 if v is None else float(v) for v in values]
    b = ax.bar(x, heights, color=_colors(rows))
    _annotate(ax, b, values, fmt)
    top = max(heights) or 1.0
    for xi, v, r in zip(x, values, rows):
        if v is None:
            best = costs.get(r["optimizer"], {}).get("best", np.inf)
            note = "not reached" + (f"\n(best {best:.3f})" if np.isfinite(best) else "")
            ax.annotate(note, (xi, 0.02 * top), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15 * top)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\nto a feasible core with $V/V_e \\leq$ {target:.3f}, reached by all",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved: {p}")


def plot_time(rows, out_dir, costs=None, target=None):
    """Bar chart of wall time to the common quality target (to the finish line for old records)."""
    if costs:
        _cost_bars(
            rows,
            costs,
            target,
            "time_s",
            "{:.0f}s",
            "wall-clock time [s]",
            "Time by optimizer",
            "optimizer_vs_time.png",
            out_dir,
        )
        return
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(rows))
    total = np.array([r["opt_time_s"] for r in rows], dtype=float)
    b = ax.bar(x, total, color=_colors(rows))
    _annotate(ax, b, total, "{:.0f}s")
    ax.set_ylim(0, 1.15 * (max(total) or 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("wall-clock time [s]")
    ax.set_title("Optimization time by optimizer (to the finish line)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(out_dir, "optimizer_vs_time.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved: {p}")


def plot_simulations(rows, out_dir, costs=None, target=None):
    """Bar chart of FE solves to the common quality target (to the finish line for old records).

    Solves, not evaluations: a gradient step costs a forward and an adjoint
    solve, a population member one, so solves is the count that compares.
    """
    if costs:
        _cost_bars(
            rows,
            costs,
            target,
            "solves",
            "{:.0f}",
            "FEM simulations",
            "Simulations by optimizer",
            "optimizer_vs_simulations.png",
            out_dir,
        )
        return
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(rows))
    solves = [r["n_solves"] for r in rows]
    b = ax.bar(x, solves, color=_colors(rows))
    _annotate(ax, b, solves)
    ax.set_ylim(0, 1.15 * (max(solves) or 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("FEM simulations")
    ax.set_title("Simulations by optimizer (to the finish line)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(out_dir, "optimizer_vs_simulations.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved: {p}")


def plot_parameters(rows, out_dir, datasheet_mm=None):
    """2 x 3 panel of each free parameter's change from the datasheet design.

    One panel per parameter because the changes span three decades.  With
    ``datasheet_mm`` the change is measured from the YAML geometry for every
    seed; without it the record's own change from its start point is used.
    """
    fig, axes = plt.subplots(2, 3, figsize=fig_panel_size)
    axes = axes.ravel()
    x = np.arange(len(rows))

    for i, key in enumerate(config.vary_keys):
        ax = axes[i]
        if datasheet_mm and key in datasheet_mm:
            vals = [r[f"{key}_opt_mm"] - datasheet_mm[key] for r in rows]
        else:
            vals = [r[f"{key}_change_mm"] for r in rows]
        b = ax.bar(x, vals, color=_colors(rows))
        _annotate(ax, b, vals, "{:+.3f}", fontsize=10)
        ax.axhline(0, color=neutral_color, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(_labels(rows), rotation=45, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(param_label[key], fontsize=10)
        ax.set_ylabel("change [mm]", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.18 * (hi - lo))

    fig.suptitle("Geometry change from the datasheet design, by optimizer", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out_dir, "optimizer_vs_parameter_change.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved: {p}")


def plot_analytical(rows, out_dir):
    """Paired bars of simulated and analytical inductance at each final design."""
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(rows))
    w = 0.38
    sim = [r["l_sim_nh"] for r in rows]
    ana = [r["l_analytical_nh"] for r in rows]
    target = rows[0]["target_nh"]

    b1 = ax.bar(x - w / 2, sim, w, color=_colors(rows))
    b2 = ax.bar(x + w / 2, ana, w, color=_colors(rows), alpha=0.45, hatch="//")
    _annotate(ax, b1, sim, "{:.1f}")
    _annotate(ax, b2, ana, "{:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("inductance [nH]")
    ax.set_ylim(0, 1.3 * target)
    ax.set_title("FEM vs analytical inductance at each optimized design")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=neutral_color, label="simulation"),
        Patch(facecolor=neutral_color, alpha=0.45, hatch="//", label="analytical"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    fig.tight_layout()
    p = os.path.join(out_dir, "optimizer_vs_analytical.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved: {p}")


# ---- CURVES ----


def _incumbent_curve(solves, err, grid):
    """Return the best |L - target| reached after each solve count in ``grid``.

    A running minimum over every evaluation, so the curve is a step function
    that changes only when a design closer to the target is evaluated.
    """
    order = np.argsort(solves)
    s_sorted = np.asarray(solves, dtype=float)[order]
    best = np.minimum.accumulate(np.asarray(err, dtype=float)[order])
    idx = np.searchsorted(s_sorted, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    out[idx >= 0] = best[idx[idx >= 0]]
    return out


def _read_histories(run_json, keep=None):
    """Read history.csv into {optimizer: [(solves, |L - target| in nH) per seed]}.

    ``keep`` is the {(optimizer, seed)} set the bar charts draw, so curves
    and bars always describe the same runs.
    """
    path = history_path(run_json)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    with open(run_json) as f:
        doc = json.load(f)
    target_nh = next(
        (
            float(rec["target_nh"])
            for entry in doc.get("optimizers", {}).values()
            for rec in entry.get("seeds", [])
            if rec.get("target_nh")
        ),
        None,
    )
    if target_nh is None:
        return {}

    groups = {}
    for row in rows:
        try:
            err = abs(float(row["l_nh"]) - target_nh)
        except (TypeError, ValueError, KeyError):
            continue
        raw = row.get("solves") or row.get("evals") or ""
        try:
            seed_i = int(float(row.get("seed") or 0))
        except ValueError:
            seed_i = 0
        if keep is not None and (row.get("optimizer"), seed_i) not in keep:
            continue
        key = (row.get("optimizer"), row.get("backend"), row.get("seed"))
        s_list, e_list = groups.setdefault(key, ([], []))
        try:
            n = float(raw)
        except ValueError:
            n = len(s_list) + 1
        s_list.append(n)
        e_list.append(err)

    runs = {}
    for (opt, _backend, _seed), (s_list, e_list) in groups.items():
        if s_list and opt in opt_order:
            runs.setdefault(opt, []).append((s_list, e_list))
    return runs


def plot_incumbent(run_json, out_dir, keep=None):
    """Best inductance error reached against FEM solves, mean over seeds per optimizer.

    Curves are extended flat past the point where an optimizer stopped; the
    s.e.m. band is drawn only where more than one seed exists.
    """
    runs = _read_histories(run_json, keep=keep)
    if not runs:
        print("  no history.csv files found -- incumbent plot skipped.")
        return

    budget = max(max(s) for group in runs.values() for s, _ in group)
    grid = np.arange(1.0, budget + 1.0)

    fig, ax = plt.subplots(figsize=fig_size)
    for opt in opt_order:
        if opt not in runs:
            continue
        curves = np.vstack([_incumbent_curve(s, e, grid) for s, e in runs[opt]])
        n = np.sum(~np.isnan(curves), axis=0)
        valid = n > 0
        if not valid.any():
            continue
        g = grid[valid]
        block = curves[:, valid]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(block, axis=0)
        colour = opt_colors.get(opt, neutral_color)
        label = opt_label.get(opt, opt)
        if block.shape[0] > 1:
            with np.errstate(invalid="ignore"):
                sd = np.nanstd(block, axis=0, ddof=1)
            sem = sd / np.sqrt(np.maximum(n[valid], 1))
            ax.fill_between(
                g, np.maximum(mean - sem, 1e-30), mean + sem, color=colour, alpha=0.20, linewidth=0
            )
        ax.plot(g, mean, color=colour, lw=1.8, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("number of simulations")
    ax.set_ylabel("incumbent inductance error [nH]")
    ax.set_title("Incumbent error vs number of simulations")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "optimizer_vs_incumbent.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_volume(rows, out_dir):
    """Bar chart of the final core volume per optimizer."""
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(rows))
    vol = [r.get("core_volume_mm3") for r in rows]
    if not any(v is not None for v in vol):
        plt.close(fig)
        print("  no core_volume_mm3 in summary -- volume plot skipped.")
        return
    vol = [0.0 if v is None else v for v in vol]

    b = ax.bar(x, vol, color=_colors(rows))
    _annotate(ax, b, vol, "{:.0f}")

    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("core volume [mm$^3$]")
    # Fixed scale so studies at different targets compare.
    ax.set_ylim(0, 20000)
    ax.set_title("Core volume by optimizer")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, "optimizer_vs_volume.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def _rows_per_iteration(optimizer, settings):
    """Return how many history rows make one iteration of ``optimizer``."""
    if optimizer in ("cmaes", "ga"):
        return int((settings or {}).get("pop_size") or 1)
    return 1


def plot_convergence(history, target_nh, out_path, title, rows_per_iteration=1):
    """Draw one optimizer's raw loss and inductance against its own iterations.

    One raw curve, no smoothing and no best-so-far.  For population methods
    a generation's best member is its proposal.  ``history`` is an in-memory
    list or history.csv rows, with keys ``loss`` and ``l_nh``.
    """
    loss_all = np.asarray([float(h["loss"]) for h in history])
    l_all = np.asarray([float(h["l_nh"]) for h in history])
    k = max(int(rows_per_iteration), 1)
    if k > 1:
        n = len(loss_all) // k
        loss_g = loss_all[: n * k].reshape(n, k)
        j = np.argmin(loss_g, axis=1)
        loss = loss_g[np.arange(n), j]
        l_nh = l_all[: n * k].reshape(n, k)[np.arange(n), j]
    else:
        loss, l_nh = loss_all, l_all
    it = np.arange(len(loss))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(it, loss, lw=1.0, color="C0", marker=".", ms=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{title} convergence")

    axes[1].plot(it, l_nh, lw=1.0, color="C0", marker=".", ms=3)
    axes[1].axhline(target_nh, color="gray", ls="--", lw=1, label=f"target ({target_nh:.1f} nH)")
    axes[1].set_xlabel("iteration" if k == 1 else "generation")
    axes[1].set_ylabel("inductance L [nH]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def replot_convergence(run_json, out_root=None):
    """Redraw every optimizer's winning-seed convergence.png from history.csv."""
    with open(run_json) as f:
        doc = json.load(f)
    label = doc.get("core_label") or os.path.basename(os.path.dirname(run_json))
    out_root = out_root or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(run_json)))
    )
    path = history_path(run_json)
    if not os.path.isfile(path):
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for entry in doc.get("optimizers", {}).values():
        win = next((s for s in entry.get("seeds", []) if s.get("winner")), None)
        if win is None:
            continue
        opt, backend, seed = entry["optimizer"], entry["backend"], int(win["seed"])
        hist = [
            r
            for r in rows
            if r.get("optimizer") == opt
            and r.get("backend") == backend
            and int(float(r.get("seed") or 0)) == seed
            and r.get("loss")
        ]
        if not hist:
            continue
        target_nh = float(win["target_nh"])
        settings = (entry.get("config") or {}).get("optimizer_settings") or {}
        out_dir = os.path.join(out_root, opt, label, backend)
        if not os.path.isdir(out_dir):
            continue
        plot_convergence(
            hist,
            target_nh,
            os.path.join(out_dir, "convergence.png"),
            opt_label.get(opt, opt),
            rows_per_iteration=_rows_per_iteration(opt, settings),
        )
        print(f"  saved: {os.path.join(out_dir, 'convergence.png')}")


# ---- ALL FIGURES ----


def stale_runs(run_json):
    """List optimizer output that the consolidated record has not seen.

    A leftover seed<N>/ folder or any file newer than run.json means an
    optimizer was run on its own, outside the driver; convergence.png is
    exempt because replot_convergence rewrites it.
    """
    if not os.path.isfile(run_json):
        return []
    t_run = os.path.getmtime(run_json)
    comparison_dir = os.path.dirname(os.path.abspath(run_json))
    core_label = os.path.basename(comparison_dir)
    out_root = os.path.dirname(os.path.dirname(comparison_dir))

    stale = []
    for opt in opt_order:
        opt_dir = os.path.join(out_root, opt, core_label)
        if not os.path.isdir(opt_dir):
            continue
        for backend in sorted(os.listdir(opt_dir)):
            root = os.path.join(opt_dir, backend)
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if name.startswith("seed") and os.path.isdir(path):
                    stale.append(f"{opt}/{backend}/{name} (un-absorbed run)")
                elif name == "convergence.png":
                    continue
                elif os.path.isfile(path) and os.path.getmtime(path) > t_run:
                    stale.append(f"{opt}/{backend} ({name} newer than run.json)")
                    break
    return stale


def make_all(run_json, out_dir=None, allow_stale=False):
    """Draw every comparison figure beside the run.json it was read from.

    Refuses when any optimizer output is newer than run.json, unless
    ``allow_stale``: the figures would otherwise mix two versions of a run.

    Returns:
        The aggregated per-optimizer rows, or [] when nothing was plotted.
    """
    stale = stale_runs(run_json)
    if stale and not allow_stale:
        raise SystemExit(
            f"  REFUSING TO PLOT: {len(stale)} run(s) are newer than "
            f"{os.path.basename(run_json)}:\n"
            + "".join(f"      {s}\n" for s in stale)
            + "  Those runs are not in the consolidated record, so the figures\n"
            "  would describe a different version of them than the output folder\n"
            "  holds.\n"
            "  Fix: re-run them through optimizers/main_run_optimizer.py, which\n"
            "  updates run.json as it goes.\n"
            "  Override (only if you know the rows still match): "
            "compare_plots.py <label> --allow-stale"
        )
    out_dir = out_dir or os.path.dirname(run_json)
    raw = read_run(run_json)
    if not raw:
        print(f"  {run_json} has no runs -- nothing to plot.")
        return []
    raw = select_protocol(raw)
    if not raw:
        print(f"  {run_json} has no runs under the current protocol -- nothing to plot.")
        return []
    rows = aggregate(raw)
    missing = [o for o in opt_order if o not in _names(rows)]
    if missing:
        print(
            f"  note: no results for {', '.join(missing)} -- "
            "those optimizers are omitted, not plotted as zero."
        )
    os.makedirs(out_dir, exist_ok=True)
    keep = {(r["optimizer"], int(r.get("seed") or 0)) for r in raw}
    costs, target = quality_costs(run_json, keep=keep)
    plot_time(rows, out_dir, costs, target)
    plot_simulations(rows, out_dir, costs, target)
    plot_parameters(rows, out_dir, datasheet_geometry(run_json))
    plot_analytical(rows, out_dir)
    plot_volume(rows, out_dir)
    plot_incumbent(run_json, out_dir, keep=keep)
    replot_convergence(run_json)
    return rows


# ---- MAIN ----


def main():
    """Draw the comparison figures for the label or run.json path on the command line."""
    args = [a for a in sys.argv[1:] if a != "--allow-stale"]
    allow_stale = "--allow-stale" in sys.argv
    if args and os.path.isfile(args[0]):
        path = args[0]
    else:
        label = args[0] if args else config.core_name
        path = run_path(config.output_dir, label)
    if not os.path.isfile(path):
        raise SystemExit(
            f"No run record at {path}\n"
            "Run optimizers/main_run_optimizer.py first, or pass the path to a "
            "run.json as the first argument."
        )
    print(f"=== comparison plots from {path} ===")
    make_all(path, allow_stale=allow_stale)


if __name__ == "__main__":
    main()
