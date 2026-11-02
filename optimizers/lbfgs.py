"""
L-BFGS Optimizer
================
L-BFGS (optax) over the six core design variables.
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
class LbfgsSettings:
    """Search settings; edit here to change a run."""

    max_iters:    int   = 300  # runaway guard, not a budget: the run ends at the finish line
    ls_max_steps: int   = 20   # optax's zoom line-search cap; an exhausted search ends the run
# fmt: on


hyper = LbfgsSettings()


# ---- SEARCH ----


def _init_state(tx, x):
    """Return tx.init(x) with every weak-typed leaf cast to its own dtype, so the jitted step compiles once."""
    return jax.tree.map(
        lambda a: jnp.asarray(a, dtype=a.dtype) if getattr(a, "weak_type", False) else a,
        tx.init(x),
    )


def run_lbfgs(core_spec, backend, seed=0, max_iters=hyper.max_iters, verbose=True):
    """Run optax L-BFGS from the start design of ``seed`` until the finish line.

    Every evaluation inside the compiled line search is recorded through a
    host callback, so solves are counted, not inferred.  A line search that
    exhausts its trial points ends the run, and the stopping rule settles
    the design in hand.

    Returns:
        (problem, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start).
    """
    prob, n_r, n_z, mesh_size, _geom, loss_and_l, traceable = ob.setup(core_spec, backend)
    target_l = float(core_spec["target_l"])
    f_rel = make_relative_loss(loss_and_l, core_spec)
    x_start = start_point(core_spec, seed)
    x = jnp.asarray(to_relative(x_start, core_spec))
    stopper = ob.TargetReached(target_l, core_spec=core_spec)

    evaluations = []

    def _record(loss, l):
        evaluations.append((float(loss), float(l)))

    def fun(u):
        loss, l = f_rel(u)
        jax.debug.callback(_record, loss, l, ordered=True)
        return loss

    # optax.lbfgs's own default line search, restated so the loop knows its step cap.
    tx = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=hyper.ls_max_steps, initial_guess_strategy="one"
        )
    )
    opt_state = _init_state(tx, x)
    value_and_grad = jax.jit(jax.value_and_grad(fun))

    @jax.jit
    def step(grad, state, x, value):
        upd, state = tx.update(grad, state, x, value=value, grad=grad, value_fn=fun)
        return x + upd, state

    def l_of(value):
        """Return the inductance recorded with this loss value."""
        v = float(value)
        return next(l for (loss_value, l) in reversed(evaluations) if loss_value == v)

    history = []
    if verbose:
        print(
            f"=== L-BFGS optimizer | core {core_spec.get('name', core_spec.get('type', '?'))} | "
            f"backend {backend} | target {target_l * 1e9:.1f} nH | "
            f"mesh {mesh_size * 1e3:.3f} mm | seed {seed} | "
            f"max {max_iters} iters (log coordinates) ==="
        )
        print(
            "  (first evaluation compiles the FEM solve and its adjoint -- "
            "about half a minute at 0.1 mm with nothing printed)",
            flush=True,
        )

    t_start = time.time()
    value, grad = value_and_grad(x)
    x_last = x
    for i in range(max_iters):
        loss_f = float(value)
        if not np.isfinite(loss_f):
            # Every geometry has a finite loss; a non-finite one is a solver failure.
            if verbose:
                print(f"  iteration {i}: non-finite loss, stopping at the last finite iterate")
            x = x_last
            stopper.settle()
            break
        x_last = x
        l_f = l_of(value)
        n_evals = len(evaluations)
        history.append(
            {
                "iteration": i,
                "loss": loss_f,
                "l_nh": l_f * 1e9,
                "evals": n_evals,
                "solves": 2 * n_evals,
                "t_s": time.time() - t_start,
                "x": np.asarray(from_relative(x, core_spec)).tolist(),
            }
        )
        if verbose and (i % 10 == 0 or i == max_iters - 1):
            print(
                f"  iter {i:4d}: L = {l_f * 1e9:8.2f} nH  loss = {loss_f:.3e}  "
                f"solves = {2 * n_evals}",
                flush=True,
            )
        if stopper.update(
            l_f, solves=2 * n_evals, x=from_relative(np.asarray(x), core_spec), loss=loss_f
        ):
            if verbose:
                print(f"  iteration {i}: {stopper.stop_message()}")
            break

        try:
            x_new, opt_state = step(grad, opt_state, x, value)
        except jax.errors.JaxRuntimeError as err:
            # The linear solver gave up inside the line search; keep the
            # iterate in hand rather than the process dying without a record.
            if verbose:
                print(
                    f"  iteration {i}: solver failed in the line search ({err.__class__.__name__}); stopping"
                )
            stopper.settle()
            break
        info = optax.tree.get(opt_state, "info")
        if int(info.num_linesearch_steps) >= hyper.ls_max_steps:
            stopper.settle()
            if verbose:
                print(
                    f"  iteration {i}: line search exhausted its "
                    f"{hyper.ls_max_steps} trial points (loss {loss_f:.3e}); "
                    f"converged to numerical precision, stopping"
                )
            break

        x = x_new
        value = optax.tree.get(opt_state, "value")
        grad = optax.tree.get(opt_state, "grad")
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
        from_relative(np.asarray(x), core_spec),
        history,
        opt_time,
        stopper,
        x_start,
    )


# ---- ONE RUN ----


def run_one_backend(core_spec, backend, core_name, seed=None):
    """Run one seed, verify and report the finished design, and save the record.

    Results go to config.output_dir/lbfgs/<core_name>/<backend>/seed<k>/; ``seed``
    selects the start design.
    """
    seed_used = 0 if seed is None else int(seed)

    target_nh = core_spec["target_l"] * 1e9
    (prob, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start) = run_lbfgs(
        core_spec, backend, seed=seed_used
    )

    out_dir = os.path.join(config.output_dir, "lbfgs", core_name, backend, f"seed{seed_used}")
    l_sim, checks = report_design(core_spec, prob, n_r, n_z, mesh_size, x_opt, out_dir)
    compare_plots.plot_convergence(
        history,
        target_nh,
        os.path.join(out_dir, "convergence.png"),
        "L-BFGS",
        rows_per_iteration=1,
    )
    p_opt = params_dict_from_x(x_opt, core_spec)
    l_an = analytical_inductance({"params": p_opt})
    ob.save_run(
        out_dir,
        "lbfgs",
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
            "optimizer": "optax.lbfgs, library defaults",
            "linesearch": "zoom (strong Wolfe)",
            "ls_max_steps": hyper.ls_max_steps,
            "coordinates": "log-relative to datasheet",
            "bounds": "none; log coordinates clipped at +-log_coord_max",
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
    """Run L-BFGS on every backend for the core named on the command line."""
    core_name = sys.argv[1] if len(sys.argv) > 1 else config.core_name
    core_spec = apply_config(load_core_spec(core_name))
    target_nh = core_spec["target_l"] * 1e9

    print(
        f"=== L-BFGS x {len(config.backends)} backends | core {core_name} | target {target_nh:.1f} nH ==="
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
        f"\nAll outputs saved under {os.path.join(config.output_dir, 'lbfgs', core_name)}/<backend>/"
    )


if __name__ == "__main__":
    main()
