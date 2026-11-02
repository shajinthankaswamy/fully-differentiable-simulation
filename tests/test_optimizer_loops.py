"""Every optimizer's search loop actually runs at the working mesh: a few
rounds each, checking the history it records, the stopping rule it consults,
the design it returns and that its seed is what makes runs repeat."""

import gc
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)

import adam  # noqa: E402
import bayesian  # noqa: E402
import cmaes  # noqa: E402
import ga  # noqa: E402
import lbfgs  # noqa: E402
import objective as ob  # noqa: E402

run_mesh_mm = 0.1
run_target_nh = 250.0
history_keys = ("iteration", "loss", "l_nh", "evals", "solves", "t_s")


def _spec():
    return ob.apply_config(
        ob.load_core_spec("pq_40x40"),
        target_l_nh=run_target_nh,
        mesh_size_mm=run_mesh_mm,
        material="N87",
    )


@pytest.fixture(autouse=True)
def _release_between_optimizers():
    """Each optimizer compiles its own 0.1 mm programs; free them before the next."""
    yield
    jax.clear_caches()
    gc.collect()


def _check_run(out, spec, n_rows):
    """Common checks on a run_* return tuple; returns (history, stopper)."""
    prob, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start = out
    assert (n_r, n_z) == (350, 600) and abs(mesh_size - run_mesh_mm * 1e-3) < 1e-12
    assert len(history) == n_rows, len(history)
    for h in history:
        assert all(k in h for k in history_keys), sorted(h)
        assert np.isfinite(h["loss"]) and np.isfinite(h["l_nh"]) and h["loss"] >= 0.0
    solves = [h["solves"] for h in history]
    assert solves == sorted(solves) and solves[-1] >= n_rows
    assert history[-1]["t_s"] <= opt_time
    x_opt = np.asarray(x_opt, dtype=float)
    assert x_opt.shape == (len(ob.config.vary_keys),)
    assert np.all(x_opt > 0.0)
    assert np.allclose(x_start, ob.start_point(spec, 0))
    assert stopper.rounds >= 1 and stopper.solves == solves[-1]
    return history, stopper


@pytest.mark.slow
def test_adam_runs_and_records_two_solves_per_iteration():
    spec = _spec()
    history, stopper = _check_run(
        adam.run_adam(spec, "lineax", seed=0, max_iters=3, verbose=False), spec, 3
    )
    assert [h["solves"] for h in history] == [2, 4, 6]
    assert stopper.rounds == 3
    assert abs(history[0]["l_nh"] - run_target_nh) < 0.06 * run_target_nh, (
        "iteration 0 is the re-gapped datasheet core"
    )


@pytest.mark.slow
def test_lbfgs_runs_and_counts_line_search_evaluations():
    spec = _spec()
    history, _ = _check_run(
        lbfgs.run_lbfgs(spec, "lineax", seed=0, max_iters=3, verbose=False), spec, 3
    )
    evals = [h["evals"] for h in history]
    assert evals == sorted(evals) and evals[-1] >= 3
    assert [h["solves"] for h in history] == [2 * e for e in evals]
    losses = [h["loss"] for h in history]
    assert losses[-1] < losses[0], "three L-BFGS iterations must reduce the loss"


@pytest.mark.slow
def test_cmaes_runs_one_solve_per_sample():
    spec = _spec()
    history, _ = _check_run(
        cmaes.run_cmaes(spec, "lineax", max_gens=2, pop_size=4, seed=0, verbose=False), spec, 8
    )
    assert [h["solves"] for h in history] == list(range(1, 9))


@pytest.mark.slow
def test_ga_evaluates_the_initial_population_then_each_generation():
    spec = _spec()
    history, _ = _check_run(
        ga.run_ga(spec, "lineax", max_gens=1, pop_size=4, seed=0, verbose=False), spec, 8
    )
    assert [h["solves"] for h in history] == list(range(1, 9))
    assert abs(history[0]["l_nh"] - run_target_nh) < 0.06 * run_target_nh, (
        "member 0 is the shared start design"
    )


@pytest.mark.slow
def test_bayesian_runs_its_initial_design_then_the_surrogate_loop():
    spec = _spec()
    history, _ = _check_run(
        bayesian.run_bayesian(spec, "lineax", n_init=3, n_iters=2, seed=0, verbose=False),
        spec,
        5,
    )
    assert all("x" in h and len(h["x"]) == len(ob.config.vary_keys) for h in history)


@pytest.mark.slow
def test_seed_repeats_a_run_and_a_different_seed_changes_it():
    spec = _spec()

    def losses(seed):
        out = bayesian.run_bayesian(spec, "lineax", n_init=3, n_iters=0, seed=seed, verbose=False)
        return [h["loss"] for h in out[5]]

    first, again, other = losses(1), losses(1), losses(2)
    assert first == again
    assert first != other


def test_lbfgs_init_state_has_no_weak_types_so_the_step_compiles_once():
    tx = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=lbfgs.hyper.ls_max_steps, initial_guess_strategy="one"
        )
    )
    x = jnp.ones(len(ob.config.vary_keys))
    state = lbfgs._init_state(tx, x)
    assert not any(getattr(a, "weak_type", False) for a in jax.tree_util.tree_leaves(state))

    def fun(u):
        return jnp.sum((u - 1.3) ** 2 * jnp.arange(1, len(ob.config.vary_keys) + 1))

    @jax.jit
    def step(grad, state, x, value):
        upd, state = tx.update(grad, state, x, value=value, grad=grad, value_fn=fun)
        return x + upd, state

    value, grad = jax.value_and_grad(fun)(x)
    for _ in range(3):
        x, state = step(grad, state, x, value)
        value = optax.tree.get(state, "value")
        grad = optax.tree.get(state, "grad")
    assert step._cache_size() == 1
