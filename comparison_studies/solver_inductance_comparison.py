"""
Gap Study of the Solve Time
===========================
At the 0.1 mm working mesh, times the forward, backward and total solve of
lineax and feax at every gap in ``study.gap_mm``, with the same timing
run_forward.py prints for its single 1 mm gap.

    python comparison_studies/solver_inductance_comparison.py          # run the sweep
    python comparison_studies/solver_inductance_comparison.py --plot   # redraw from the saved CSV
"""

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_fem_engine.run_forward import (  # noqa: E402
    ExperimentConfig,
    apply_config,
    backend_label,
    build_problem,
    complete_solve,
    geometry_fill,
    inductance_functions,
    load_core_spec,
    output_root,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class Study:
    """Settings of the gap sweep; edit here to change a run."""

    core:     str   = "pq_40x40"                          # core YAML under cores/
    backends: tuple = ("lineax", "feax")
    mesh_mm:  float = 0.1                                 # working mesh       [mm]
    gap_mm:   tuple = tuple(0.5 * k for k in range(1, 11))  # 0.5 to 5          [mm]
    n_calls:  int   = 10                                  # timed calls per backend per gap; median kept
    out_dir:  str   = os.path.join(output_root, "solver_inductance_comparison")
    fig_size: tuple = (7.5, 5.0)                          # every figure       [in]
# fmt: on


study = Study()
style = {"lineax": "s-", "feax": "^--"}


# ---- SWEEP ----


def gap_rows(gap_mm):
    """Return one row per backend with the forward, backward and total time at one gap."""
    cfg = ExperimentConfig(
        core_name=study.core,
        mesh_size=study.mesh_mm * 1e-3,
        gap_size=gap_mm * 1e-3,
        backends=study.backends,
    )
    spec = load_core_spec(study.core)
    apply_config(spec, cfg)
    prob, n_r, n_z, h = build_problem(spec)
    cfg.grid_shape = (n_r + 1, n_z + 1)
    fill, _ = geometry_fill(prob, n_r, n_z, h)
    solves = complete_solve(inductance_functions(cfg, prob), fill, study.n_calls)

    rows = []
    for name in study.backends:
        s = solves[name]
        rows.append(
            {
                "gap_mm": gap_mm,
                "backend": name,
                "forward_s": s["forward_s"],
                "backward_s": s["backward_s"],
                "total_s": s["grad_s"],
                "l_nh": s["l"] * 1e9,
            }
        )
        print(
            f"{gap_mm:7.1f} {name:<8} {s['forward_s']:11.4f} {s['backward_s']:12.4f} "
            f"{s['grad_s']:9.4f} {s['l'] * 1e9:10.3f}",
            flush=True,
        )
    return rows


# ---- CSV ----


def append_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("gap_mm", "forward_s", "backward_s", "total_s", "l_nh"):
            r[k] = float(r[k])
    return rows


# ---- PLOTS ----


def plot_gap_sweep(rows):
    """Draw forward, backward and total time against gap size, one figure each."""
    for key, what in (
        ("forward_s", "Forward"),
        ("backward_s", "Backward"),
        ("total_s", "Total (forward + backward)"),
    ):
        fig, ax = plt.subplots(figsize=study.fig_size)
        for name in study.backends:
            mine = sorted((r for r in rows if r["backend"] == name), key=lambda r: r["gap_mm"])
            ax.plot(
                [r["gap_mm"] for r in mine],
                [r[key] for r in mine],
                style[name],
                lw=1.8,
                label=backend_label(name),
            )
        ax.set_xticks(study.gap_mm)
        ax.set_ylim(0.0, max(1.0, ax.get_ylim()[1]))  # at least 1 s, never cutting a point
        ax.set_xlabel("Gap size [mm]")
        ax.set_ylabel("time [s]")
        ax.set_title(f"{what} time vs gap size ({study.core}, {study.mesh_mm:g} mm mesh)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(study.out_dir, f"gap_vs_{key[:-2]}_time.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved: {path}")


# ---- MAIN ----


def main():
    """Run the gap sweep and write its CSV and figures to ``study.out_dir``.

    Each gap runs in a fresh process: in one long process the same gap gets
    slower after a few gaps (measured 1.25 s -> 1.46 s at 2.0 mm, memory
    2.0 -> 2.9 GB), which drew a false hump.  ``--gap MM CSV`` runs one gap
    and appends its rows to CSV.
    """
    if sys.argv[1:2] == ["--gap"]:
        gap, csv_path = float(sys.argv[2]), sys.argv[3]
        append_csv(csv_path, gap_rows(gap))
        return

    t0 = time.time()
    os.makedirs(study.out_dir, exist_ok=True)
    csv_path = os.path.join(study.out_dir, "sweep.csv")

    if "--plot" in sys.argv:
        plot_gap_sweep(read_csv(csv_path))
        return

    print(
        f"=== Gap sweep | {study.core} | {study.mesh_mm:g} mm mesh | "
        f"{min(study.gap_mm):g}-{max(study.gap_mm):g} mm gap ==="
    )
    print(
        f"{'gap[mm]':>7} {'backend':<8} {'forward[s]':>11} {'backward[s]':>12} "
        f"{'total[s]':>9} {'L[nH]':>10}",
        flush=True,
    )
    if os.path.exists(csv_path):
        os.remove(csv_path)
    for gap in study.gap_mm:
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--gap", repr(gap), csv_path], check=True
        )

    plot_gap_sweep(read_csv(csv_path))
    print(f"\nDone in {time.time() - t0:.0f} s; outputs under {study.out_dir}/")


if __name__ == "__main__":
    main()
