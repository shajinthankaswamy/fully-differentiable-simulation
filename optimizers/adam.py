"""
Adam Optimizer
==============
Adam (optax) over the six core design variables.
The objective, finish line and record are shared in objective.py.
"""

import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config  # noqa: E402

import compare_plots  # noqa: E402
import objective as ob  # noqa: E402
from new_fem_engine.run_forward import analytical_inductance  # noqa: E402
from objective import (  # noqa: E402
    _print_design,
    apply_config,
    from_relative,
    load_core_spec,
    make_relative_loss,
    params_dict_from_x,
    report_design,
    start_point,
    to_relative,
)

# ---- SETTINGS ----


# fmt: off
@dataclass
class AdamSettings:
    """Search settings; edit here to change a run."""

    max_iters:     int   = 300   # runaway guard, not a budget: the run ends at the finish line
    learning_rate: float = 0.01  # log coordinates: moves every variable by ~1 % per step
# fmt: on


hyper = AdamSettings()


# ---- SEARCH ----


def run_adam(
    core_spec, backend, seed=0, max_iters=hyper.max_iters, lr=hyper.learning_rate, verbose=True
):
    """Run Adam from the start design of ``seed`` until the finish line.

    One iteration costs two FEM solves, the forward and the adjoint.

    Returns:
        (problem, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start)
        with x_opt the last iterate [m], the design the stopping rule judged.
    """
    prob, n_r, n_z, mesh_size, _geom, loss_and_l, traceable = ob.setup(core_spec, backend)
    target_l = float(core_spec["target_l"])

    f_rel = make_relative_loss(loss_and_l, core_spec)
    vg = jax.value_and_grad(f_rel, has_aux=True)
    value_and_grad = jax.jit(vg) if traceable else vg

    x_start = start_point(core_spec, seed)
    u = jnp.asarray(to_relative(x_start, core_spec))
    stopper = ob.TargetReached(target_l, core_spec=core_spec)

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(u)

    history = []
    best_loss = float("inf")
    u_last = np.asarray(u)

    if verbose:
        print(
            f"=== Adam optimizer | core {core_spec.get('name', core_spec.get('type', '?'))} | "
            f"backend {backend} | target {target_l * 1e9:.1f} nH | "
            f"mesh {mesh_size * 1e3:.3f} mm | lr {lr:g} (log coordinates) | "
            f"seed {seed} | {max_iters} iters ==="
        )
        print(
            "  (first evaluation compiles the FEM solve and its adjoint -- "
            "about half a minute at 0.1 mm with nothing printed)",
            flush=True,
        )

    t_start = time.time()
    for i in range(max_iters):
        (loss, l), grad = value_and_grad(u)
        loss_f, l_f = float(loss), float(l)
        if not np.isfinite(loss_f):
            # Every geometry has a finite loss; a non-finite one is a solver failure.
            if verbose:
                print(f"  iteration {i}: non-finite loss, stopping at the last finite iterate")
            u = jnp.asarray(u_last)
            stopper.settle()
            break
        best_loss = min(best_loss, loss_f)
        u_last = np.asarray(u)
        history.append(
            {
                "iteration": i,
                "loss": loss_f,
                "l_nh": l_f * 1e9,
                "evals": i + 1,
                "solves": 2 * (i + 1),
                "t_s": time.time() - t_start,
                "x": np.asarray(from_relative(u, core_spec)).tolist(),
            }
        )

        if verbose and (i % 20 == 0 or i == max_iters - 1):
            print(
                f"  iter {i:4d}: L = {l_f * 1e9:8.2f} nH  "
                f"loss = {loss_f:.3e}  best = {best_loss:.3e}",
                flush=True,
            )

        if stopper.update(
            l_f, solves=2 * (i + 1), x=from_relative(np.asarray(u), core_spec), loss=loss_f
        ):
            if verbose:
                print(f"  iteration {i}: {stopper.stop_message()}")
            break

        updates, opt_state = tx.update(grad, opt_state, u)
        u = optax.apply_updates(u, updates)
    opt_time = time.time() - t_start

    if verbose and history:
        print(
            f"  done in {opt_time:.1f}s, {len(history)} iters, "
            f"final loss = {history[-1]['loss']:.3e} "
            f"(|L - target| = {abs(history[-1]['l_nh'] - target_l * 1e9):.3f} nH)"
        )

    return (
        prob,
        n_r,
        n_z,
        mesh_size,
        np.asarray(from_relative(u_last, core_spec), dtype=float),
        history,
        opt_time,
        stopper,
        x_start,
    )


# ---- ONE RUN ----


def run_one_backend(core_spec, backend, core_name, seed=None):
    """Run one seed, verify and report the finished design, and save the record.

    Results go to config.output_dir/adam/<core_name>/<backend>/seed<k>/; ``seed``
    selects the start design.
    """
    seed_used = 0 if seed is None else int(seed)

    target_nh = core_spec["target_l"] * 1e9
    (prob, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start) = run_adam(
        core_spec, backend, seed=seed_used
    )

    out_dir = os.path.join(config.output_dir, "adam", core_name, backend, f"seed{seed_used}")
    l_sim, checks = report_design(core_spec, prob, n_r, n_z, mesh_size, x_opt, out_dir)
    compare_plots.plot_convergence(
        history,
        target_nh,
        os.path.join(out_dir, "convergence.png"),
        "Adam",
        rows_per_iteration=1,
    )
    p_opt = params_dict_from_x(x_opt, core_spec)
    l_an = analytical_inductance({"params": p_opt})
    ob.save_run(
        out_dir,
        "adam",
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
            "max_iters": hyper.max_iters,
            "learning_rate": hyper.learning_rate,
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
    """Run Adam on every backend for the core named on the command line."""
    core_name = sys.argv[1] if len(sys.argv) > 1 else config.core_name
    core_spec = apply_config(load_core_spec(core_name))
    target_nh = core_spec["target_l"] * 1e9

    print(
        f"=== Adam x {len(config.backends)} backends | core {core_name} | target {target_nh:.1f} nH ==="
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
        f"\nAll outputs saved under {os.path.join(config.output_dir, 'adam', core_name)}/<backend>/"
    )


if __name__ == "__main__":
    main()
