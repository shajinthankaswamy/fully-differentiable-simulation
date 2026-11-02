"""The record contract: what save_run writes is what the driver folds into
run.json and what the comparison figures read.  No FEM solve is needed --
the inductance of a finished design is a number the record stores."""

import csv
import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)

import compare_plots as cp  # noqa: E402
import main_run_optimizer as drv  # noqa: E402
import objective as ob  # noqa: E402

run_mesh_mm = 0.1
run_target_nh = 250.0
N_R, N_Z = 350, 600
# Keys the driver, its summary and compare_plots read from a seed record.
record_keys = (
    "optimizer",
    "backend",
    "seed",
    "protocol_version",
    "target_nh",
    "mesh_size_mm",
    "n_r",
    "n_z",
    "x_opt_mm",
    "x_start_mm",
    "core_volume_mm3",
    "volume_ratio",
    "l_sim_nh",
    "l_analytical_nh",
    "abs_error_nh",
    "reached_target",
    "design_checks",
    "best_loss",
    "n_evals",
    "n_solves",
    "opt_time_s",
    "run_solves",
    "run_time_s",
    "convergence",
    "config",
)


# Every history row carries its design, its volume ratio and whether it is feasible.
history_header = (
    list(ob.config.history_fields)
    + [f"{k}_mm" for k in ob.config.vary_keys]
    + ["volume_ratio", "feasible"]
)


def _spec():
    return ob.apply_config(
        ob.load_core_spec("pq_40x40"),
        target_l_nh=run_target_nh,
        mesh_size_mm=run_mesh_mm,
        material="N87",
    )


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    """Point the driver at an empty output tree."""
    monkeypatch.setattr(drv.config, "output_dir", str(tmp_path))
    return str(tmp_path)


def _design(spec, yoke_scale):
    """Datasheet design with the yoke thickness scaled through the leg height: a
    thinner yoke shrinks the core without touching the winding or the outline,
    and 0.65 and above keeps the yoke under B_MAX_T at 250 nH (0.914x V_e at 0.7)."""
    x = ob.x0_from_spec(spec)
    i_leg, i_win = (
        ob.config.vary_keys.index("leg_height"),
        ob.config.vary_keys.index("window_height"),
    )
    yoke = 0.5 * (x[i_leg] - x[i_win])
    x[i_leg] = x[i_win] + 2.0 * yoke * yoke_scale
    return x


def _write_seed(output_dir, opt, seed, l_nh, yoke_scale=0.7, on_target=True):
    """Write one finished seed the way an optimizer module does; return its record.

    ``yoke_scale`` sets the design (see _design), so seeds differ in
    volume; ``on_target`` decides whether the stopping rule saw the finish line.
    """
    spec = _spec()
    label = drv._output_label(run_target_nh)
    x_ds = ob.x0_from_spec(spec)
    x_opt = _design(spec, yoke_scale)
    l_sim = l_nh * 1e-9
    history = []
    stopper = ob.TargetReached(spec["target_l"], core_spec=spec)
    for i in range(ob.config.converge_rounds + 2):
        l_i = (l_nh if on_target else l_nh + 20.0) * 1e-9
        loss = ((l_i - spec["target_l"]) / spec["target_l"]) ** 2
        history.append(
            {
                "iteration": i,
                "loss": loss,
                "l_nh": l_i * 1e9,
                "evals": i + 1,
                "solves": i + 1,
                "t_s": float(i),
            }
        )
        stopper.update(l_i, solves=i + 1, x=x_opt, loss=loss)
    out_dir = os.path.join(output_dir, opt, label, "lineax", f"seed{seed}")
    os.makedirs(out_dir)
    for name in drv.run_images:
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(b"png")
    checks = ob.design_checks(spec, x_opt, l_sim)
    assert checks["feasible"], "the synthetic design must meet every condition"
    return ob.save_run(
        out_dir,
        opt,
        "lineax",
        spec,
        N_R,
        N_Z,
        run_mesh_mm * 1e-3,
        x_opt,
        history,
        12.5,
        l_sim,
        2.6e-7,
        settings={"note": "synthetic"},
        stopper=stopper,
        seed=seed,
        x_start=x_ds,
        checks=checks,
    )


def test_result_json_carries_every_key_the_driver_and_figures_read(output_dir):
    r = _write_seed(output_dir, "adam", 0, run_target_nh)
    for key in record_keys:
        assert key in r, key
    label = drv._output_label(run_target_nh)
    seed_dir = os.path.join(output_dir, "adam", label, "lineax", "seed0")
    with open(os.path.join(seed_dir, "result.json")) as f:
        on_disk = json.load(f)
    assert (
        on_disk["l_sim_nh"] == r["l_sim_nh"]
        and on_disk["protocol_version"] == ob.config.protocol_version
    )
    with open(os.path.join(seed_dir, "history.csv"), newline="") as f:
        header = next(csv.reader(f))
    assert header == history_header
    assert r["reached_target"] and r["convergence"]["converged"]
    assert r["n_solves"] == r["convergence"]["converged_solves"] < r["run_solves"]


def test_absorb_seed_folds_seeds_into_run_json_and_promotes_the_smallest_core(output_dir):
    label = drv._output_label(run_target_nh)
    _write_seed(output_dir, "adam", 0, run_target_nh, yoke_scale=0.7)
    _write_seed(output_dir, "adam", 1, run_target_nh, yoke_scale=0.65)
    assert drv._absorb_seed("adam", "lineax", label, 0)
    assert drv._absorb_seed("adam", "lineax", label, 1)

    doc = drv._load_run(label)
    entry = doc["optimizers"]["adam/lineax"]
    assert entry["n_seeds"] == 2 and entry["winner_seed"] == 1
    assert [s["seed"] for s in entry["seeds"]] == [0, 1]
    assert [s["winner"] for s in entry["seeds"]] == [False, True]
    assert "config" in entry and "config" not in entry["seeds"][0]

    root = os.path.join(output_dir, "adam", label, "lineax")
    assert not os.path.isdir(os.path.join(root, "seed0"))
    assert not os.path.isdir(os.path.join(root, "seed1"))
    assert all(os.path.isfile(os.path.join(root, name)) for name in drv.run_images)

    with open(drv.run_history_path(label), newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["optimizer", "backend", "seed"] + history_header
    assert {r["seed"] for r in rows} == {"0", "1"} and len(rows) == 2 * (
        ob.config.converge_rounds + 2
    )


def test_absorb_seed_reports_a_seed_that_left_no_record(output_dir):
    assert not drv._absorb_seed("ga", "lineax", drv._output_label(run_target_nh), 0)


def test_pick_winner_prefers_the_smallest_core_that_reached_the_target():
    reached_big = {
        "seed": 0,
        "reached_target": True,
        "core_volume_mm3": 15000.0,
        "best_loss": 0.1,
    }
    reached_small = {
        "seed": 1,
        "reached_target": True,
        "core_volume_mm3": 12000.0,
        "best_loss": 0.4,
    }
    closest_miss = {
        "seed": 2,
        "reached_target": False,
        "core_volume_mm3": 9000.0,
        "best_loss": 0.01,
    }
    winner, why = drv._pick_winner([reached_big, reached_small, closest_miss])
    assert winner is reached_small and "smallest core" in why
    far_miss = {"seed": 3, "reached_target": False, "core_volume_mm3": 8000.0, "best_loss": 5.0}
    winner, why = drv._pick_winner([closest_miss, far_miss])
    assert winner is closest_miss and "closest" in why


def test_reset_seed_records_drops_only_that_optimizer(output_dir):
    label = drv._output_label(run_target_nh)
    _write_seed(output_dir, "adam", 0, run_target_nh)
    _write_seed(output_dir, "cmaes", 0, run_target_nh)
    drv._absorb_seed("adam", "lineax", label, 0)
    drv._absorb_seed("cmaes", "lineax", label, 0)
    _write_seed(output_dir, "adam", 1, run_target_nh)  # a leftover seed folder

    drv._reset_seed_records("adam", "lineax", label)

    assert set(drv._load_run(label)["optimizers"]) == {"cmaes/lineax"}
    with open(drv.run_history_path(label), newline="") as f:
        assert {r["optimizer"] for r in csv.DictReader(f)} == {"cmaes"}
    assert not os.path.isdir(os.path.join(output_dir, "adam", label, "lineax", "seed1"))


def test_older_protocol_entries_are_dropped_with_a_message(capsys):
    doc = {
        "optimizers": {
            "adam/lineax": {"seeds": [{"seed": 0, "protocol_version": ob.config.protocol_version}]},
            "ga/lineax": {
                "seeds": [{"seed": 0, "protocol_version": ob.config.protocol_version - 1}]
            },
        }
    }
    drv._drop_older_protocols(doc, ob.config.protocol_version, "label")
    assert set(doc["optimizers"]) == {"adam/lineax"}
    assert "ga/lineax" in capsys.readouterr().out


def test_comparison_figures_draw_from_the_record(output_dir):
    label = drv._output_label(run_target_nh)
    for opt, seeds in (("adam", (0, 1)), ("cmaes", (0, 1))):
        for seed in seeds:
            _write_seed(
                output_dir, opt, seed, run_target_nh + 0.2 * seed, yoke_scale=0.7 - 0.05 * seed
            )
            assert drv._absorb_seed(opt, "lineax", label, seed)

    rows = cp.make_all(drv.run_record_path(label))

    assert {r["optimizer"] for r in rows} == {"adam", "cmaes"}
    comparison_dir = os.path.dirname(drv.run_record_path(label))
    for name in (
        "optimizer_vs_time.png",
        "optimizer_vs_simulations.png",
        "optimizer_vs_parameter_change.png",
        "optimizer_vs_analytical.png",
        "optimizer_vs_volume.png",
        "optimizer_vs_incumbent.png",
    ):
        assert os.path.isfile(os.path.join(comparison_dir, name)), name
    for opt in ("adam", "cmaes"):
        assert os.path.isfile(os.path.join(output_dir, opt, label, "lineax", "convergence.png"))


def test_optimizer_summary_prints_every_seed_and_the_winner(output_dir, capsys):
    label = drv._output_label(run_target_nh)
    _write_seed(output_dir, "ga", 0, run_target_nh, yoke_scale=0.7)
    _write_seed(output_dir, "ga", 1, run_target_nh, yoke_scale=0.65)
    drv._absorb_seed("ga", "lineax", label, 0)
    drv._absorb_seed("ga", "lineax", label, 1)
    capsys.readouterr()

    drv._print_optimizer_summary("ga", "lineax", label, run_target_nh)

    out = capsys.readouterr().out
    assert "2 seeds" in out and "<-- best" in out and "best design (seed 1)" in out
    assert "across the 2 seeds" in out and "reached target" in out


def test_quality_costs_score_every_optimizer_on_its_best_feasible_design(tmp_path, monkeypatch):
    """Cost is the solves until the best feasible design first reaches a core every optimizer delivered."""
    run_json = tmp_path / "run.json"
    run_json.write_text("{}")
    fields = ["optimizer", "backend", "seed", "solves", "elapsed_s", "volume_ratio", "feasible"]
    rows = [
        # a infeasible at 0.70 (ignored), feasible 0.90 at 2, feasible 0.80 at 3
        ("a", 1, 1.0, 0.70, 0),
        ("a", 2, 2.0, 0.90, 1),
        ("a", 3, 3.0, 0.80, 1),
        # b feasible 0.805 at 5, then worse designs that must not raise its best
        ("b", 5, 5.0, 0.805, 1),
        ("b", 6, 6.0, 0.95, 1),
        # c ends at 0.85, the largest final core, so it sets the common target
        ("c", 1, 1.0, 0.95, 1),
        ("c", 4, 4.0, 0.85, 1),
    ]
    with open(tmp_path / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for opt, solves, t, v, ok in rows:
            w.writerow([opt, "lineax", 0, solves, t, v, ok])

    monkeypatch.setattr(cp, "opt_order", ["a", "b", "c"])
    costs, target = cp.quality_costs(str(run_json))
    assert target == 0.85
    assert costs["a"]["solves"] == 3 and costs["a"]["time_s"] == 3.0
    assert costs["b"]["solves"] == 5 and costs["b"]["best"] == 0.805
    assert costs["c"]["solves"] == 4 and costs["c"]["best"] == 0.85
