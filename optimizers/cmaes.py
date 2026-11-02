"""
CMA-ES Optimizer
================
CMA-ES (evosax) over the six core design variables.
The objective, finish line and record are shared in objective.py.
"""

import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import CMA_ES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config  # noqa: E402

import compare_plots  # noqa: E402
import objective as ob  # noqa: E402
from new_fem_engine.run_forward import analytical_inductance  # noqa: E402
from objective import (  # noqa: E402
    _print_design,
    apply_config,
    load_core_spec,
    make_population_evaluator,
    params_dict_from_x,
    report_design,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class CmaesSettings:
    """Search settings; edit here to change a run."""

    max_gens: int   = 300   # runaway guard, not a budget: the run ends at the finish line
    pop_size: int   = 8     # designs per generation
    seed:     int   = 42    # random stream
    sigma0:   float = 0.15  # initial step, fraction of each datasheet value
# fmt: on


hyper = CmaesSettings()


# ---- SEARCH ----


def run_cmaes(
    core_spec,
    backend,
    max_gens=hyper.max_gens,
    pop_size=hyper.pop_size,
    seed=hyper.seed,
    verbose=True,
):
    """Run CMA-ES from the start design of ``seed`` until the finish line.

    The proposal the stopping rule judges is the best design so far.  One
    generation costs ``pop_size`` FEM solves.

    Returns:
        (problem, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start)
        with x_opt the best design so far [m].
    """
    prob, n_r, n_z, mesh_size, geometry_from_x, loss_fn, traceable = ob.setup(core_spec, backend)
    target_l = float(core_spec["target_l"])
    eval_population = make_population_evaluator(loss_fn, traceable)

    x_ds = ob.x0_from_spec(core_spec)
    x0 = ob.start_point(core_spec, seed)
    stopper = ob.TargetReached(target_l, core_spec=core_spec)
    dim = len(x_ds)
    pop = int(pop_size) if pop_size is not None else 4 + int(3 * np.log(max(dim, 1)))

    es = CMA_ES(population_size=pop, solution=jnp.zeros(dim, dtype=jnp.float64))
    params = es.default_params.replace(std_init=hyper.sigma0)
    key = jax.random.PRNGKey(seed)
    state = es.init(key, jnp.asarray(ob.to_relative(x0, core_spec)), params)

    history = []
    best_loss = float("inf")
    best_l = float("nan")
    best_x = np.asarray(x0)

    if verbose:
        print(
            f"=== CMA-ES optimizer | core {core_spec.get('name', core_spec.get('type', '?'))} | "
            f"backend {backend} | target {target_l * 1e9:.1f} nH | "
            f"mesh {mesh_size * 1e3:.3f} mm | pop {pop} | "
            f"sigma0 {hyper.sigma0} (log coordinates) | seed {seed} | "
            f"{max_gens} gens ==="
        )
        print(
            "  (first generation compiles the FEM solve -- "
            "about half a minute at 0.1 mm with nothing printed)",
            flush=True,
        )

    t_start = time.time()
    for gen in range(max_gens):
        key, ask_key, tell_key = jax.random.split(key, 3)
        u_pop, state = es.ask(ask_key, state, params)
        u_pop = np.asarray(u_pop, dtype=float).reshape(pop, dim)
        designs = np.asarray(ob.from_relative(u_pop, core_spec), dtype=float)
        losses, l_values = eval_population(designs)

        for k in range(pop):
            history.append(
                {
                    "iteration": len(history),
                    "loss": float(losses[k]),
                    "l_nh": float(l_values[k]) * 1e9,
                    "evals": len(history) + 1,
                    "solves": len(history) + 1,
                    "t_s": time.time() - t_start,
                    "x": designs[k].tolist(),
                }
            )
            if losses[k] < best_loss:
                best_loss = float(losses[k])
                best_l = float(l_values[k])
                best_x = designs[k].copy()

        state, _ = es.tell(tell_key, jnp.asarray(u_pop), jnp.asarray(losses), state, params)

        if stopper.update(best_l, solves=history[-1]["solves"], x=best_x, loss=best_loss):
            if verbose:
                print(f"  generation {gen}: {stopper.stop_message()}")
            break

        if verbose:
            print(
                f"  gen {gen:3d}: best-so-far loss = {best_loss:.3e}  "
                f"sigma = {float(jnp.mean(state.std)):.3e}",
                flush=True,
            )
    opt_time = time.time() - t_start

    if verbose:
        print(
            f"  done in {opt_time:.1f}s, {len(history)} evals, "
            f"best loss = {best_loss:.3e} "
            f"(|L - target| = {abs(best_l - target_l) * 1e9:.3f} nH)"
        )

    return (prob, n_r, n_z, mesh_size, best_x, history, opt_time, stopper, x0)


# ---- ONE RUN ----


def run_one_backend(core_spec, backend, core_name, seed=None):
    """Run one seed, verify and report the finished design, and save the record.

    Results go to config.output_dir/cmaes/<core_name>/<backend>/seed<k>/; ``seed``
    selects the start design and seeds the random stream.
    """
    seed_used = 0 if seed is None else int(seed)

    target_nh = core_spec["target_l"] * 1e9
    (prob, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start) = run_cmaes(
        core_spec, backend, seed=seed_used
    )

    out_dir = os.path.join(config.output_dir, "cmaes", core_name, backend, f"seed{seed_used}")
    l_sim, checks = report_design(core_spec, prob, n_r, n_z, mesh_size, x_opt, out_dir)
    compare_plots.plot_convergence(
        history,
        target_nh,
        os.path.join(out_dir, "convergence.png"),
        "CMA-ES",
        rows_per_iteration=hyper.pop_size,
    )
    p_opt = params_dict_from_x(x_opt, core_spec)
    l_an = analytical_inductance({"params": p_opt})
    ob.save_run(
        out_dir,
        "cmaes",
        backend,
        core_spec,
        n_r,
        n_z,
        mesh_size,
        x_opt,
        history,
        opt_time,
        l_sim,
        l_an,
        settings={
            "max_gens": hyper.max_gens,
            "pop_size": hyper.pop_size,
            "sigma0_relative": hyper.sigma0,
            "coordinates": "log-relative to datasheet",
            "seed": seed_used,
        },
        stopper=stopper,
        seed=seed_used,
        x_start=x_start,
        checks=checks,
    )

    if config.print_design:
        _print_design(p_opt, l_sim, l_an, target_nh, opt_time, out_dir, checks)

    return {
        "backend": backend,
        "l_nh": l_sim * 1e9,
        "l_analytical_nh": l_an * 1e9,
        "time_s": opt_time,
    }


# ---- MAIN ----


def main():
    """Run CMA-ES on every backend for the core named on the command line."""
    core_name = sys.argv[1] if len(sys.argv) > 1 else config.core_name
    core_spec = apply_config(load_core_spec(core_name))
    target_nh = core_spec["target_l"] * 1e9

    print(
        f"=== CMA-ES x {len(config.backends)} backends | core {core_name} | target {target_nh:.1f} nH ==="
    )
    rows = []
    for backend in config.backends:
        print(f"\n--- backend: {backend} ---")
        rows.append(run_one_backend(core_spec, backend, core_name))

    print("\n=== Cross-backend summary ===")
    print(
        f"{'backend':10s} {'target (nH)':>12s} {'L_sim (nH)':>11s} {'L_an (nH)':>11s} "
        f"{'time (s)':>9s}"
    )
    for r in rows:
        print(
            f"{r['backend']:10s} {target_nh:12.2f} {r['l_nh']:11.2f} {r['l_analytical_nh']:11.2f} "
            f"{r['time_s']:9.1f}"
        )
    print(
        f"\nAll outputs saved under {os.path.join(config.output_dir, 'cmaes', core_name)}/<backend>/"
    )


if __name__ == "__main__":
    main()
