"""
Genetic Algorithm
=================
Genetic algorithm (evosax SimpleGA) over the six core design variables.
The objective, finish line and record are shared in objective.py.
"""

import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import SimpleGA

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


# Each gene mutates with probability 1 / dim by a normal step with its own
# std, adapted by Rechenberg's 1/5 success rule: the inductance band allows
# only small post and gap steps while the window and yoke can move far.
# fmt: off
@dataclass
class GaSettings:
    """Search settings; edit here to change a run."""

    max_gens:       int   = 150   # runaway guard, not a budget: the run ends at the finish line
    pop_size:       int   = 20    # designs per generation
    seed:           int   = 42    # random stream
    mut_std0:       float = 0.20  # initial mutation std, fraction of each datasheet value
    mut_std_factor: float = 0.82  # std shrink factor; 1 / factor grows it
    success_rate:   float = 0.2   # the 1/5 rule's success threshold
    min_trials:     int   = 5     # mutated children seen before a gene's std adapts
    crossover_rate: float = 0.9   # uniform crossover
# fmt: on


hyper = GaSettings()


# ---- SEARCH ----


def run_ga(
    core_spec,
    backend,
    max_gens=hyper.max_gens,
    pop_size=hyper.pop_size,
    seed=hyper.seed,
    verbose=True,
):
    """Run the GA from the start design of ``seed`` until the finish line.

    Member 0 of the initial population is the shared start design, the rest
    are mutated copies of it.  The library selects and crosses; mutation is
    per gene (see the settings).  Elitist: the best design so far replaces
    the worst child before each tell and is the proposal the stopping rule
    judges.  One generation costs ``pop_size`` FEM solves, and the initial
    population is counted too.

    Returns:
        (problem, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start)
        with x_opt the best design so far [m].
    """
    prob, n_r, n_z, mesh_size, geometry_from_x, loss_fn, traceable = ob.setup(core_spec, backend)
    target_l = float(core_spec["target_l"])
    eval_population = make_population_evaluator(loss_fn, traceable)

    x_ds = ob.x0_from_spec(core_spec)
    dim = len(x_ds)
    pop = int(pop_size)
    stopper = ob.TargetReached(target_l, core_spec=core_spec)

    history = []
    t_start = time.time()

    def to_x(u_pop):
        """Map a log-coordinate population to metres."""
        return np.asarray(ob.from_relative(np.asarray(u_pop, dtype=float), core_spec), dtype=float)

    def evaluate(u_pop):
        """Evaluate a population, append it to the history and return (losses, l_values)."""
        designs = to_x(u_pop)
        losses, l_values = eval_population(designs)
        for k in range(designs.shape[0]):
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
        return losses, l_values

    rng = np.random.default_rng(seed)
    x_start = ob.start_point(core_spec, seed)
    u_start = ob.to_relative(x_start, core_spec)
    u_pop = u_start + rng.normal(0.0, hyper.mut_std0, size=(pop, dim))
    u_pop[0] = u_start
    std = np.full(dim, hyper.mut_std0)
    successes = np.zeros(dim, dtype=int)
    trials = np.zeros(dim, dtype=int)

    def mutate(bred):
        """Mutate each gene with probability 1 / dim by its own std; return (children, mask)."""
        mask = rng.random(size=(pop, dim)) < 1.0 / dim
        return bred + mask * std * rng.normal(size=(pop, dim)), mask

    if verbose:
        print(
            f"=== GA optimizer | core {core_spec.get('name', core_spec.get('type', '?'))} | "
            f"backend {backend} | target {target_l * 1e9:.1f} nH | "
            f"mesh {mesh_size * 1e3:.3f} mm | pop {pop} | "
            f"max {max_gens} gens ==="
        )
        print(
            "  (first generation compiles the FEM solve -- "
            "about half a minute at 0.1 mm with nothing printed)",
            flush=True,
        )

    losses, l_values = evaluate(u_pop)
    best_i = int(np.argmin(losses))
    best_loss = float(losses[best_i])
    best_l = float(l_values[best_i])
    best_x = to_x(u_pop)[best_i].copy()
    best_u = u_pop[best_i].copy()

    # The library's own mutation is switched off (std 0); it selects the top
    # half and crosses genomes, and mutate() above does the rest.
    es = SimpleGA(population_size=pop, solution=jnp.zeros(dim), std_schedule=lambda step: 0.0)
    params = es.default_params.replace(crossover_rate=hyper.crossover_rate)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state = es.init(init_key, jnp.asarray(u_pop), jnp.asarray(losses), params)

    stop = stopper.update(best_l, solves=len(history), x=best_x, loss=best_loss)

    for gen in range(max_gens):
        if stop:
            break
        key, ask_key, tell_key = jax.random.split(key, 3)
        bred, state = es.ask(ask_key, state, params)
        u_pop, mask = mutate(np.asarray(bred, dtype=float).reshape(pop, dim))
        losses, l_values = evaluate(u_pop)

        better = np.asarray(losses) < best_loss
        successes += (mask & better[:, None]).sum(axis=0)
        trials += mask.sum(axis=0)
        i = int(np.argmin(losses))
        if better[i]:
            best_loss = float(losses[i])
            best_l = float(l_values[i])
            best_x = to_x(u_pop)[i].copy()
            best_u = u_pop[i].copy()
        judged = trials >= hyper.min_trials
        grow = judged & (successes > hyper.success_rate * trials)
        shrink = judged & (successes == 0)
        std = np.where(
            grow, std / hyper.mut_std_factor, np.where(shrink, std * hyper.mut_std_factor, std)
        )
        successes[judged] = 0
        trials[judged] = 0

        # Elitism: the incumbent takes the worst child's place in the bred
        # population (its loss is known, so no solve); the history keeps only
        # the designs actually evaluated.
        told_u, told_losses = u_pop.copy(), np.asarray(losses, dtype=float).copy()
        worst = int(np.argmax(told_losses))
        told_u[worst], told_losses[worst] = best_u, best_loss
        state, _ = es.tell(tell_key, jnp.asarray(told_u), jnp.asarray(told_losses), state, params)

        stop = stopper.update(best_l, solves=len(history), x=best_x, loss=best_loss)
        if stop and verbose:
            print(f"  generation {gen}: {stopper.stop_message()}")
        elif verbose:
            stds = " ".join(f"{v * 100:.2f}" for v in std)
            print(
                f"  gen {gen:3d}: best-so-far loss = {best_loss:.3e}  "
                f"better children = {int(better.sum())}  stds % = [{stds}]",
                flush=True,
            )

    opt_time = time.time() - t_start

    if verbose:
        print(
            f"  done in {opt_time:.1f}s, {len(history)} evals, "
            f"best loss = {best_loss:.3e} "
            f"(|L - target| = {abs(best_l - target_l) * 1e9:.3f} nH)"
        )

    return (prob, n_r, n_z, mesh_size, best_x, history, opt_time, stopper, x_start)


# ---- ONE RUN ----


def run_one_backend(core_spec, backend, core_name, seed=None):
    """Run one seed, verify and report the finished design, and save the record.

    Results go to config.output_dir/ga/<core_name>/<backend>/seed<k>/; ``seed``
    selects the start design and seeds the random stream.
    """
    seed_used = 0 if seed is None else int(seed)

    target_nh = core_spec["target_l"] * 1e9
    (prob, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start) = run_ga(
        core_spec, backend, seed=seed_used
    )

    out_dir = os.path.join(config.output_dir, "ga", core_name, backend, f"seed{seed_used}")
    l_sim, checks = report_design(core_spec, prob, n_r, n_z, mesh_size, x_opt, out_dir)
    compare_plots.plot_convergence(
        history,
        target_nh,
        os.path.join(out_dir, "convergence.png"),
        "GA",
        rows_per_iteration=hyper.pop_size,
    )
    p_opt = params_dict_from_x(x_opt, core_spec)
    l_an = analytical_inductance({"params": p_opt})
    ob.save_run(
        out_dir,
        "ga",
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
            "mut_std0_relative": hyper.mut_std0,
            "mutation": (
                f"each gene with probability 1/dim; per-gene std by the 1/5 rule over "
                f"{hyper.min_trials} trials, x{1 / hyper.mut_std_factor:.2f} / x{hyper.mut_std_factor:g}"
            ),
            "crossover_rate": hyper.crossover_rate,
            "elitism": "the best design so far replaces the worst child before every tell",
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
    """Run the GA on every backend for the core named on the command line."""
    core_name = sys.argv[1] if len(sys.argv) > 1 else config.core_name
    core_spec = apply_config(load_core_spec(core_name))
    target_nh = core_spec["target_l"] * 1e9

    print(
        f"=== GA x {len(config.backends)} backends | core {core_name} | target {target_nh:.1f} nH ==="
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
        f"\nAll outputs saved under {os.path.join(config.output_dir, 'ga', core_name)}/<backend>/"
    )


if __name__ == "__main__":
    main()
