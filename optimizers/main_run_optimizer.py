"""
Five-Optimizer Race
===================
Runs every seed of every optimizer from its start design to the finish line
in objective.py, folds each seed into one run.json and history.csv under
output/comparison/<label>/, promotes the winning seed's figures to
output/<optimizer>/<label>/<backend>/, and draws the comparison figures.
Each optimizer seed runs in its own process, so JAX's memory is freed.

    python optimizers/main_run_optimizer.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time

_optimizers_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _optimizers_dir)
sys.path.insert(0, os.path.dirname(_optimizers_dir))

from settings import config  # noqa: E402

import compare_plots  # noqa: E402

_module_names = ("adam", "lbfgs", "cmaes", "ga", "bayesian")


# ---- OUTPUT FILES ----


run_images = (
    "aphi.png",
    "bfield.png",
    "convergence.png",
    "energy_density.png",
    "geometry.png",
    "geometry_mesh.png",
    "material_current.png",
    "psi.png",
)
run_files = run_images
stale_files = ("result.json", "history.csv", "seeds.json", "seeds_history.csv")


# ---- RUNNING ONE SEED ----


def _load_module(name):
    import importlib

    if name not in _module_names:
        raise KeyError(f"Unknown optimizer {name!r}; choices: {_module_names}")
    return importlib.import_module(name)


def _seeds_for(opt_name):
    """Return the seed list for an optimizer; every optimizer gets the same seeds."""
    return list(range(config.n_seeds))


def _load_core_spec(target_nh):
    import objective as ob

    return ob.apply_config(
        ob.load_core_spec(config.core_name),
        target_l_nh=target_nh,
        mesh_size_mm=config.mesh_size_mm,
        material=config.material,
    )


def _output_label(target_nh):
    """Return the study label, e.g. pq_40x40_target250.0nH_N87_mesh0.100mm."""
    bits = []
    if target_nh is not None:
        bits.append(f"target{target_nh:.1f}nH")
    if config.material is not None:
        bits.append(config.material)
    if config.mesh_size_mm is not None:
        bits.append(f"mesh{config.mesh_size_mm:.3f}mm")
    if not bits:
        return config.core_name
    return f"{config.core_name}_" + "_".join(bits)


def _run_in_process(opt_name, backend, target_nh, core_label, seed):
    """Run one optimizer seed in this process."""

    # The driver prints one summary per optimizer after all its seeds.
    config.print_design = False
    mod = _load_module(opt_name)
    core_spec = _load_core_spec(target_nh)
    mod.run_one_backend(core_spec, backend, core_label, seed=seed)


def _run_isolated(opt_name, backend, target_nh, core_label, seed):
    """Run one optimizer seed in a child process; return True if it exited cleanly.

    stdout is inherited so progress streams live.  The child writes its own
    result.json, so a child that dies simply leaves no record.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--run",
        opt_name,
        backend,
        repr(float(target_nh)),
        core_label,
        str(int(seed)),
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(
            f"    ! {opt_name} seed {seed} exited with code {proc.returncode} "
            f"(killed by the OS if negative -- most likely out of memory). "
            f"Continuing with the remaining optimizers."
        )
    return proc.returncode == 0


# ---- RECORDS ----


def run_record_path(core_label):
    """Return the path of the one run.json for a study."""
    return os.path.join(config.output_dir, "comparison", core_label, "run.json")


def run_history_path(core_label):
    """Return the path of the one history.csv for a study."""
    return os.path.join(config.output_dir, "comparison", core_label, "history.csv")


def _load_run(core_label):
    path = run_record_path(core_label)
    if not os.path.isfile(path):
        return {"core_label": core_label, "optimizers": {}}
    with open(path) as f:
        return json.load(f)


def _save_run(core_label, doc):
    path = run_record_path(core_label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def _pick_winner(records):
    """Return (record, reason): the smallest core among seeds that reached the
    target, or the closest seed when none did."""

    def _loss(rec):
        v = rec.get("best_loss")
        return float("inf") if v is None else float(v)

    def _vol(rec):
        v = rec.get("core_volume_mm3")
        return float("inf") if v is None else float(v)

    reached = [r for r in records if r.get("reached_target")]
    if reached:
        return min(reached, key=_vol), "smallest core among seeds that reached the target"
    return min(records, key=_loss), "closest to target (no seed reached it)"


def _reset_seed_records(opt_name, backend, core_label):
    """Drop everything a previous run of this optimizer left in the study."""
    doc = _load_run(core_label)
    if doc.get("optimizers", {}).pop(f"{opt_name}/{backend}", None) is not None:
        _save_run(core_label, doc)

    hist = run_history_path(core_label)
    if os.path.isfile(hist):
        with open(hist) as f:
            rows = [
                r
                for r in csv.DictReader(f)
                if not (r.get("optimizer") == opt_name and r.get("backend") == backend)
            ]
        if rows:
            with open(hist, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        else:
            os.remove(hist)

    root = os.path.join(config.output_dir, opt_name, core_label, backend)
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.startswith("seed") and os.path.isdir(os.path.join(root, name)):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        for name in stale_files:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                os.remove(p)


def _drop_older_protocols(doc, version, core_label):
    """Remove optimizer entries recorded under an older protocol than ``version``.

    Their numbers are not comparable with the new run and their record keys
    may no longer be the ones the figures read.
    """
    stale = [
        key
        for key, entry in doc.get("optimizers", {}).items()
        if any(int(s.get("protocol_version") or 1) < version for s in entry.get("seeds", []))
    ]
    for key in stale:
        del doc["optimizers"][key]
    if stale:
        print(
            f"    dropped {', '.join(stale)} from {core_label}/run.json: recorded under an "
            f"older protocol than {version}; re-run them for a comparable study"
        )


def _history_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f), None)


def _absorb_seed(opt_name, backend, core_label, seed):
    """Fold one finished seed into run.json and history.csv and delete its folder.

    Called after every seed so the optimizer's folder always holds the best
    design so far.  Returns False when the seed left no result.json.
    """
    root = os.path.join(config.output_dir, opt_name, core_label, backend)
    seed_dir = os.path.join(root, f"seed{int(seed)}")
    result = os.path.join(seed_dir, "result.json")
    if not os.path.isfile(result):
        return False

    with open(result) as f:
        r = json.load(f)
    hist = os.path.join(seed_dir, "history.csv")

    # The first row's elapsed time is the evaluation that paid the compile.
    first_eval_s = ""
    rows = []
    if os.path.isfile(hist):
        with open(hist) as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0].get("elapsed_s"):
            first_eval_s = rows[0]["elapsed_s"]

    doc = _load_run(core_label)
    doc.setdefault("optimizers", {})
    _drop_older_protocols(doc, int(r.get("protocol_version") or 1), core_label)
    key = f"{opt_name}/{backend}"
    entry = doc["optimizers"].setdefault(
        key, {"optimizer": opt_name, "backend": backend, "seeds": []}
    )
    entry["config"] = r.pop("config", entry.get("config"))
    r["seed"] = int(seed)
    r["first_eval_s"] = first_eval_s
    entry["seeds"] = [x for x in entry["seeds"] if x.get("seed") != int(seed)] + [r]

    winner, why = _pick_winner(entry["seeds"])
    for rec in entry["seeds"]:
        rec["winner"] = rec.get("seed") == winner.get("seed")
    entry["seeds"].sort(key=lambda x: x.get("seed", 0))
    entry["winner_seed"] = winner.get("seed")
    entry["n_seeds"] = len(entry["seeds"])
    entry["selected_by"] = why
    _save_run(core_label, doc)

    if rows:
        stacked = run_history_path(core_label)
        os.makedirs(os.path.dirname(stacked), exist_ok=True)
        fieldnames = ["optimizer", "backend", "seed"] + list(rows[0])
        exists = os.path.isfile(stacked) and _history_header(stacked) == fieldnames
        with open(stacked, "a" if exists else "w", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            for row in rows:
                w.writerow({"optimizer": opt_name, "backend": backend, "seed": int(seed), **row})

    promoted = winner.get("seed") == int(seed)
    if promoted:
        for name in run_files:
            src = os.path.join(seed_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(root, name))
    shutil.rmtree(seed_dir, ignore_errors=True)

    vol = r.get("core_volume_mm3")
    hit = "reached" if r.get("reached_target") else "missed"
    size = "" if vol is None else f", {vol:.0f} mm3"
    verdict = "promoted" if promoted else f"kept seed {winner.get('seed')}"
    print(f"    seed {seed} absorbed ({hit}{size}) -> {verdict}")
    return True


# ---- SUMMARY ----


def _check_lines(checks, indent="    "):
    """Format the design_checks dict of a record; mirrors objective._check_lines."""
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
        f"({checks['copper_area_needed_mm2'] * checks['copper_fill_factor']:.1f} "
        f"mm2 of copper at {checks['copper_fill_factor']:g} fill) "
        f"({ok('copper_fit')})",
        f"{indent}winding: coil area {checks['coil_area_mm2']:.1f} mm2, datasheet winding "
        f"{checks['winding_area_mm2']:.1f} mm2 ({ok('winding')}); clearance "
        f"{checks['clearance_mm']:.3f} mm, datasheet {checks['datasheet_clearance_mm']:.3f} mm "
        f"({ok('clearance')})",
        f"{indent}fringing: gap / window height = {checks['gap_to_window']:.3f}, "
        f"limit {checks['gap_to_window_max']:g} ({ok('fringing')})",
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


def _print_optimizer_summary(opt_name, backend, core_label, target_nh):
    """Print every seed of one optimizer on a line, then its winning design."""
    root = os.path.join(config.output_dir, opt_name, core_label, backend)
    d = _load_run(core_label).get("optimizers", {}).get(f"{opt_name}/{backend}")
    if not d:
        return
    seeds = d.get("seeds", [])
    if not seeds:
        return

    n = len(seeds)
    print(
        f"=== {opt_name} / {backend} | target {target_nh:.1f} nH | "
        f"{n} seed{'s' if n != 1 else ''} ==="
    )
    print(f"   {'seed':>4} {'loss':>10} {'L (nH)':>10} {'solves':>8} {'volume (mm3)':>14}")
    for r in sorted(seeds, key=lambda x: x.get("seed", 0)):
        mark = "  <-- best" if r.get("winner") else ""
        print(
            f"   {r.get('seed', 0):>4} {r.get('best_loss', float('nan')):>10.3e} "
            f"{r.get('l_sim_nh', float('nan')):>10.3f} "
            f"{int(r.get('n_solves') or 0):>8} "
            f"{r.get('core_volume_mm3', float('nan')):>14.0f}{mark}"
        )

    win = next((r for r in seeds if r.get("winner")), None)
    if win is not None:
        print(f"\n  best design (seed {win.get('seed')}) -- {d.get('selected_by', '')}:")
        mm = win.get("x_opt_mm", {})
        for k in config.vary_keys:
            if k in mm:
                print(f"    {k:22s} = {mm[k]:9.4f} mm")
        print(
            f"    L = {win.get('l_sim_nh', float('nan')):.3f} nH | "
            f"analytical = {win.get('l_analytical_nh', float('nan')):.2f} nH | "
            f"target = {target_nh:.2f} nH"
        )
        ratio = win.get("volume_ratio")
        if ratio is not None:
            print(
                f"    core volume = {win.get('core_volume_mm3', 0):.0f} mm3 "
                f"= {ratio:.3f}x the datasheet V_e"
            )
        for line in _check_lines(win.get("design_checks")):
            print(line)
        if n > 1:
            vols = [r.get("core_volume_mm3") for r in seeds if r.get("core_volume_mm3")]
            hits = sum(1 for r in seeds if r.get("reached_target"))
            if vols:
                print(
                    f"    across the {n} seeds: {hits} reached target, "
                    f"volume {min(vols):.0f}-{max(vols):.0f} mm3 "
                    f"({max(vols) / min(vols):.2f}x spread)"
                )
    print(f"  saved under {root}/\n", flush=True)


def _normalize_names(value):
    """Return a flat list of names from a string or list, split on commas."""
    items = [value] if isinstance(value, str) else list(value)
    names = []
    for item in items:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    return names


# ---- MAIN ----


def main():
    """Run the whole study: every target, optimizer, backend and seed, then the figures."""
    targets = (
        list(config.target_l_nh)
        if isinstance(config.target_l_nh, (list, tuple))
        else [config.target_l_nh]
    )
    optimizers = _normalize_names(config.optimizers)
    backends = _normalize_names(config.backends)

    print(
        f"=== main_run_optimizer | core {config.core_name} | targets {targets} nH | "
        f"optimizers {optimizers} | backends {backends} ===\n"
    )

    total = 0
    print("run plan:")
    for tnh in targets:
        per = [(o, len(_seeds_for(o))) for o in optimizers]
        n = sum(c for _, c in per) * len(backends)
        total += n
        print(
            f"  {tnh:.1f} nH | race to the finish line of objective.py: the inductance "
            f"band, plus every physical condition when config.constraint_weight > 0"
        )
        print(
            "      "
            + ", ".join(f"{o} x {c} seed{'s' if c != 1 else ''}" for o, c in per)
            + f"  ->  {n} run(s)"
        )
    print(f"  TOTAL: {total} run(s)\n")

    print(
        f"process isolation: {'on -- one child per optimizer' if config.isolate_processes else 'OFF'}\n"
    )

    labels = []
    failed = []
    run_start = time.time()
    for target_nh in targets:
        core_label = _output_label(target_nh)
        if core_label not in labels:
            labels.append(core_label)
        for opt_name in optimizers:
            for backend in backends:
                _reset_seed_records(opt_name, backend, core_label)
                for seed in _seeds_for(opt_name):
                    n_rep = len(_seeds_for(opt_name))
                    tag = f" | seed {seed}" if n_rep > 1 else ""
                    print(
                        f"--- target {target_nh:.1f}nH | {opt_name} / {backend}{tag} ---",
                        flush=True,
                    )
                    if config.isolate_processes:
                        _run_isolated(opt_name, backend, target_nh, core_label, seed)
                    else:
                        _run_in_process(opt_name, backend, target_nh, core_label, seed)
                    if not _absorb_seed(opt_name, backend, core_label, seed):
                        failed.append(f"{opt_name}/{backend}@{target_nh:.1f}nH#seed{seed}")
                    print(flush=True)
                _print_optimizer_summary(opt_name, backend, core_label, target_nh)

    print("=== Comparison data + figures ===")
    for core_label in labels:
        path = run_record_path(core_label)
        if not os.path.isfile(path):
            print(f"  {core_label}: no finished runs, nothing to summarise.")
            continue
        compare_plots.make_all(path)
    print()

    print("=== Final summary ===")
    print(
        f"{'optimizer':10s} {'backend':8s} {'seed':>5s} {'target (nH)':>12s} "
        f"{'L_sim (nH)':>11s} {'L_an (nH)':>10s} {'|err| (nH)':>11s} "
        f"{'volume (mm3)':>13s} {'V/V_ds':>7s} {'solves':>8s} {'time (s)':>9s}"
    )
    for core_label in labels:
        for entry in _load_run(core_label).get("optimizers", {}).values():
            for r in entry.get("seeds", []):
                mark = "  <-- best" if r.get("winner") else ""
                print(
                    f"{entry['optimizer']:10s} {entry['backend']:8s} "
                    f"{r.get('seed', 0):5d} {r['target_nh']:12.2f} "
                    f"{r['l_sim_nh']:11.2f} "
                    f"{r.get('l_analytical_nh', float('nan')):10.2f} "
                    f"{r['abs_error_nh']:11.4f} "
                    f"{r.get('core_volume_mm3', float('nan')):13.0f} "
                    f"{r.get('volume_ratio', float('nan')):7.3f} "
                    f"{int(r['n_solves']):8d} {r['opt_time_s']:9.1f}{mark}"
                )
    if failed:
        print(f"\nDid not finish ({len(failed)}): {', '.join(failed)}")
        print("Those leave no record and are absent from the comparison figures.")
    print(f"\nTotal wall time: {time.time() - run_start:.1f}s")
    print(
        f"Per-optimizer outputs: {os.path.join(config.output_dir, '<optimizer>', '<core_label>')}/<backend>/"
    )
    print("    the winning seed's figures -- numbers live in run.json")
    print(f"Whole run:      {os.path.join(config.output_dir, 'comparison', '<core_label>')}/")
    print("    run.json     -- every optimizer, every seed, every number")
    print("    history.csv  -- every optimizer, every seed, every evaluation")
    print("    plus the comparison figures")
    print("Redraw the comparison figures without re-running anything:")
    print("    python optimizers/compare_plots.py <core_label>")


if __name__ == "__main__":
    # "--run <optimizer> <backend> <target_nh> <core_label> <seed>" is the
    # child invocation from _run_isolated.
    if "--run" in sys.argv:
        i = sys.argv.index("--run")
        _opt, _backend, _target, _label, _seed = sys.argv[i + 1 : i + 6]
        _run_in_process(_opt, _backend, float(_target), _label, int(_seed))
    else:
        main()
