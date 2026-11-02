"""The rules that keep one stored study internally comparable: one protocol
version per figure, and the same seeds and start designs for all five."""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)
import compare_plots as cp  # noqa: E402
import main_run_optimizer as drv  # noqa: E402
import objective as ob  # noqa: E402


def _row(opt, seed, version, solves=100):
    return {
        "optimizer": opt,
        "backend": "lineax",
        "seed": float(seed),
        "protocol_version": version,
        "n_solves": solves,
        "reached_target": True,
    }


def test_mixed_protocol_versions_are_not_plotted_together(capsys):
    rows = [_row("adam", 0, 2, 144)] + [_row(o, 0, 1, 13) for o in ("lbfgs", "cmaes")]
    kept = cp.select_protocol(rows)
    assert [r["optimizer"] for r in kept] == ["adam"]
    out = capsys.readouterr().out
    assert "lbfgs" in out and "cmaes" in out, "the message must name what to re-run"


def test_single_protocol_passes_through_untouched():
    rows = [_row(o, 0, 2) for o in ("adam", "lbfgs", "cmaes", "ga", "bayesian")]
    assert len(cp.select_protocol(rows)) == 5


def test_missing_stamp_counts_as_version_one():
    rows = [
        {"optimizer": "ga", "seed": 0.0, "n_solves": 300},
        {"optimizer": "adam", "seed": 0.0, "n_solves": 144, "protocol_version": 2},
    ]
    kept = cp.select_protocol(rows)
    assert [r["optimizer"] for r in kept] == ["adam"]


def test_every_optimizer_runs_the_same_seeds():
    seeds = {o: drv._seeds_for(o) for o in drv._module_names}
    assert len({tuple(v) for v in seeds.values()}) == 1, (
        "all five must repeat the same starting conditions"
    )
    assert len(seeds["adam"]) == drv.config.n_seeds


def test_start_points_are_shared_deterministic_and_relative_to_the_datasheet():
    spec = ob.apply_config(ob.load_core_spec("pq_40x40"), target_l_nh=350.0, mesh_size_mm=1.0)
    x_ds = ob.x0_from_spec(spec)
    x_ref = ob.start_point(spec, 0)
    i_gap = ob.config.vary_keys.index("gap_size")
    keep = [i for i in range(len(x_ds)) if i != i_gap]
    assert np.allclose(x_ref[keep], x_ds[keep]), "seed 0 is the datasheet core, re-gapped"
    from new_fem_engine.run_forward import analytical_inductance

    l_ref = analytical_inductance({"params": ob.params_dict_from_x(x_ref, spec)})
    assert abs(l_ref - spec["target_l"]) < 1e-6 * spec["target_l"], "re-gapped to the target"
    for s in range(1, drv.config.n_seeds):
        a, b = ob.start_point(spec, s), ob.start_point(spec, s)
        assert np.array_equal(a, b), "a seed must name one design, always"
        assert np.all(a > 0.0)
        du = np.abs(a / x_ref - 1.0)
        assert np.all(du <= ob.config.start_perturbation + 1e-9)
    assert not np.allclose(ob.start_point(spec, 1), ob.start_point(spec, 2))


def test_log_coordinates_round_trip_and_have_no_floor():
    spec = ob.apply_config(ob.load_core_spec("pq_40x40"), mesh_size_mm=1.0)
    x_ds = ob.x0_from_spec(spec)
    x = ob.start_point(spec, 3)
    assert np.allclose(np.asarray(ob.from_relative(ob.to_relative(x, spec), spec)), x)
    assert np.allclose(ob.to_relative(x_ds, spec), 0.0)
    # Every real coordinate is a positive length; the exponent cap only keeps floats finite.
    far = np.asarray(ob.from_relative(np.array([-50.0, 50.0, -1e9, 1e9, 0.0, 0.0]), spec))
    assert np.all(far > 0.0) and np.all(np.isfinite(far))
    pop = np.asarray(ob.from_relative(np.zeros((3, len(x_ds))), spec))
    assert pop.shape == (3, len(x_ds)) and np.allclose(pop[0], x_ds)
    assert not hasattr(ob, "design_floor")


def test_every_optimizer_searches_in_log_coordinates():
    """A method left in raw metres would measure its steps in a different geometry."""
    import importlib
    import inspect

    for name in ("adam", "lbfgs", "cmaes", "ga", "bayesian"):
        src = inspect.getsource(importlib.import_module(name))
        assert "start_point(core_spec, seed)" in src, name
        assert "make_relative_loss" in src or "to_relative" in src, name
