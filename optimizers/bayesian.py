"""
Bayesian Optimization
=====================
Gaussian-process surrogate with expected improvement over the six core design variables.
The objective, finish line and record are shared in objective.py.
"""

import os
import sys
import time
from dataclasses import dataclass

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.stats import norm as jnorm

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
    params_dict_from_x,
    report_design,
)

# ---- SETTINGS ----


# Surrogate (GPJax): RBF kernel with one lengthscale per variable, constant
# mean, Gaussian noise, fitted by Adam on the marginal likelihood.  Trust
# region (TuRBO): candidates are drawn in a box of half-width tr (fraction of
# each datasheet value) around the incumbent; shrinking it is what lets the
# gap land inside the band.
# fmt: off
@dataclass
class BayesianSettings:
    """Search settings; edit here to change a run."""

    n_init:             int   = 50    # initial design, ~8x the dimensionality
    n_iters:            int   = 150   # runaway guard; GP cost is cubic in the observations
    n_candidates:       int   = 6000  # candidate pool per iteration
    ei_xi:              float = 0.01  # expected-improvement exploration margin
    gp_fit_iters:       int   = 200   # Adam steps per GP fit
    gp_learning_rate:   float = 0.05  # Adam learning rate of the GP fit
    gp_obs_stddev_init: float = 1e-2  # initial observation noise
    tr_init:            float = 0.5   # initial trust-region half-width; restart value
    tr_min:             float = 1e-4  # below this the region restarts at tr_init
    tr_max:             float = 1.0   # largest half-width
    tr_shrink:          float = 0.5   # factor after tr_fail_tol failures
    tr_expand:          float = 2.0   # factor after tr_succ_tol successes
    tr_fail_tol:        int   = 4     # consecutive failures before shrinking
    tr_succ_tol:        int   = 3     # consecutive successes before expanding
    seed:               int   = 42    # random stream
# fmt: on


hyper = BayesianSettings()


# ---- SURROGATE ----


def fit_gp(u, z):
    """Fit the GP to the observations and return (posterior, dataset).

    Args:
        u: (n, dim) log-coordinate inputs.
        z: (n,) standardised targets.
    """
    data = gpx.Dataset(X=jnp.asarray(u), y=jnp.asarray(z)[:, None])
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(lengthscale=jnp.ones(data.X.shape[1])),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=data.n, obs_stddev=hyper.gp_obs_stddev_init
    )
    posterior, _ = gpx.fit(
        model=prior * likelihood,
        objective=lambda model, d: -gpx.objectives.conjugate_mll(model, d),
        train_data=data,
        optim=optax.adam(hyper.gp_learning_rate),
        num_iters=hyper.gp_fit_iters,
        key=jax.random.PRNGKey(0),
        verbose=False,
    )
    return posterior, data


def predict_marginals(gp, u_cand):
    """Return the predictive mean and variance, observation noise included, at every candidate."""
    posterior, data = gp
    latent = posterior.predict(u_cand, train_data=data, return_covariance_type="diagonal")
    predictive = posterior.likelihood(latent)
    return predictive.mean, predictive.variance


def expected_improvement(gp, u_cand, z_best, xi=hyper.ei_xi):
    """Return the expected improvement below ``z_best`` at every candidate."""
    mu, variance = predict_marginals(gp, u_cand)
    sigma = jnp.maximum(jnp.sqrt(jnp.maximum(variance, 0.0)), 1e-9)
    imp = z_best - mu - xi
    zscore = imp / sigma
    ei = imp * jnorm.cdf(zscore) + sigma * jnorm.pdf(zscore)
    return jnp.where(sigma > 1e-9, ei, 0.0)


# ---- SEARCH ----


def run_bayesian(
    core_spec,
    backend,
    n_init=hyper.n_init,
    n_iters=hyper.n_iters,
    n_candidates=hyper.n_candidates,
    seed=hyper.seed,
    verbose=True,
):
    """Run Bayesian optimization from the start design of ``seed`` until the finish line.

    Sample 0 of the initial design is the shared start design, the rest are
    uniform in the initial trust region; the stopping rule is checked after
    every sample.  Each iteration fits a GP to log1p(loss), maximises expected
    improvement over candidates in the trust region and evaluates one design,
    one FEM solve.  The proposal the stopping rule judges is the incumbent.

    Returns:
        (problem, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start)
        with x_opt the incumbent [m].
    """
    prob, n_r, n_z, mesh_size, geometry_from_x, loss_fn, traceable = ob.setup(core_spec, backend)
    target_l = float(core_spec["target_l"])
    x_ds = ob.x0_from_spec(core_spec)
    dim = len(x_ds)
    rng = np.random.default_rng(seed)

    eval_x = jax.jit(loss_fn) if traceable else loss_fn

    if verbose:
        print(
            f"=== Bayesian optimizer | "
            f"core {core_spec.get('name', core_spec.get('type', '?'))} | "
            f"backend {backend} | target {target_l * 1e9:.1f} nH | "
            f"mesh {mesh_size * 1e3:.3f} mm | {n_init} init + "
            f"{n_iters} BO iters ==="
        )
        print(
            "  (first evaluation compiles the FEM solve -- "
            "about half a minute at 0.1 mm with nothing printed)",
            flush=True,
        )

    t_start = time.time()

    x_start = ob.start_point(core_spec, seed)
    stopper = ob.TargetReached(target_l, core_spec=core_spec)
    u_start = ob.to_relative(x_start, core_spec)
    u_init = rng.uniform(u_start - hyper.tr_init, u_start + hyper.tr_init, size=(n_init - 1, dim))
    u_init = np.vstack([[u_start], u_init])

    u_obs, y_obs, history = [], [], []

    def record(u, x, loss, l):
        u_obs.append(u)
        y_obs.append(loss)
        history.append(
            {
                "iteration": len(history),
                "loss": loss,
                "l_nh": l * 1e9,
                "evals": len(history) + 1,
                "solves": len(history) + 1,
                "t_s": time.time() - t_start,
                "x": x.tolist(),
            }
        )

    reached = False
    for i in range(n_init):
        u = u_init[i]
        x = np.asarray(ob.from_relative(u, core_spec), dtype=float)
        loss, l = eval_x(jnp.asarray(x))
        loss, l = float(loss), float(l)
        record(u, x, loss, l)
        if verbose:
            print(f"  [init {i + 1:2d}/{n_init}] loss={loss:.4e}  L={l * 1e9:.2f} nH")
        if stopper.update(l, solves=len(history), x=x, loss=loss):
            reached = True
            if verbose:
                print(f"  initial design, evaluation {len(history)}: {stopper.stop_message()}")
            break

    tr = hyper.tr_init
    tr_succ = tr_fail = 0
    n_restarts = 0
    best_seen = min((y for y in y_obs if np.isfinite(y)), default=float("inf"))

    def finite_best():
        """Return the finite-loss history entry with the lowest loss."""
        return min((e for e in history if np.isfinite(e["loss"])), key=lambda e: e["loss"])

    for it in range(0 if reached else n_iters):
        # A non-finite loss is a solver failure; it is left out of the GP
        # and never becomes the incumbent.
        y = np.asarray(y_obs)
        ok = np.isfinite(y)
        # The squared-error loss spans many decades; log1p keeps the ranking
        # and gives the RBF/Gaussian GP residuals it can model.
        y_log = np.log1p(y[ok])
        z = (y_log - y_log.mean()) / (y_log.std() + 1e-12)
        gp = fit_gp(np.stack(u_obs)[ok], z)

        u_best = np.asarray(u_obs[int(np.argmin(np.where(ok, y, np.inf)))], dtype=float)
        u_cand = jnp.asarray(rng.uniform(u_best - tr, u_best + tr, size=(n_candidates, dim)))
        ei = expected_improvement(gp, u_cand, z_best=float(z.min()))
        j = int(jnp.argmax(ei))
        u_next = np.asarray(u_cand[j])
        x_next = np.asarray(ob.from_relative(u_next, core_spec), dtype=float)

        loss, l = eval_x(jnp.asarray(x_next))
        loss, l = float(loss), float(l)
        record(u_next, x_next, loss, l)
        if verbose:
            print(
                f"  [BO {it + 1:3d}/{n_iters}] loss={loss:.4e}  "
                f"L={l * 1e9:.2f} nH  (EI={float(ei[j]):.3e}, tr={tr:.2e})"
            )

        if loss < best_seen:
            best_seen = loss
            tr_succ += 1
            tr_fail = 0
        else:
            tr_fail += 1
            tr_succ = 0
        if tr_succ >= hyper.tr_succ_tol:
            tr = min(tr * hyper.tr_expand, hyper.tr_max)
            tr_succ = 0
        elif tr_fail >= hyper.tr_fail_tol:
            tr *= hyper.tr_shrink
            tr_fail = 0
            if tr < hyper.tr_min:
                tr = hyper.tr_init
                n_restarts += 1
                if verbose:
                    print(
                        f"  trust region collapsed below {hyper.tr_min:g}; "
                        f"restart {n_restarts} at {hyper.tr_init:g} around the incumbent"
                    )

        best_entry = finite_best()
        if stopper.update(
            best_entry["l_nh"] * 1e-9,
            solves=len(history),
            x=np.asarray(best_entry["x"], dtype=float),
            loss=best_entry["loss"],
        ):
            if verbose:
                print(f"  evaluation {len(history)}: {stopper.stop_message()}")
            break

    opt_time = time.time() - t_start

    best = finite_best()
    best_loss = best["loss"]
    best_x = np.asarray(best["x"], dtype=float)

    if verbose:
        print(
            f"  done in {opt_time:.1f}s, {len(history)} evals, "
            f"best loss = {best_loss:.3e} "
            f"(|L - target| = {abs(best['l_nh'] - target_l * 1e9):.3f} nH), "
            f"{n_restarts} trust-region restart{'s' if n_restarts != 1 else ''}"
        )

    return (prob, n_r, n_z, mesh_size, best_x, history, opt_time, stopper, x_start)


# ---- ONE RUN ----


def run_one_backend(core_spec, backend, core_name, seed=None):
    """Run one seed, verify and report the finished design, and save the record.

    Results go to config.output_dir/bayesian/<core_name>/<backend>/seed<k>/; ``seed``
    selects the start design and seeds the random stream.
    """
    seed_used = 0 if seed is None else int(seed)

    target_nh = core_spec["target_l"] * 1e9
    (prob, n_r, n_z, mesh_size, x_opt, history, opt_time, stopper, x_start) = run_bayesian(
        core_spec, backend, seed=seed_used
    )

    out_dir = os.path.join(config.output_dir, "bayesian", core_name, backend, f"seed{seed_used}")
    l_sim, checks = report_design(core_spec, prob, n_r, n_z, mesh_size, x_opt, out_dir)
    compare_plots.plot_convergence(
        history,
        target_nh,
        os.path.join(out_dir, "convergence.png"),
        "Bayesian optimization",
        rows_per_iteration=1,
    )
    p_opt = params_dict_from_x(x_opt, core_spec)
    l_an = analytical_inductance({"params": p_opt})
    ob.save_run(
        out_dir,
        "bayesian",
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
            "n_init": hyper.n_init,
            "n_iters": hyper.n_iters,
            "surrogate": "GPJax RBF GP, constant mean, Gaussian noise; Adam on the marginal likelihood",
            "acquisition": "EI over a TuRBO-style trust region",
            "tr_init": hyper.tr_init,
            "tr_min": hyper.tr_min,
            "tr_max": hyper.tr_max,
            "tr_restart": "reset to tr_init around the incumbent when shrunk below tr_min",
            "coordinates": "log-relative to datasheet",
            "n_candidates": hyper.n_candidates,
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
    """Run Bayesian optimization on every backend for the core named on the command line."""
    core_name = sys.argv[1] if len(sys.argv) > 1 else config.core_name
    core_spec = apply_config(load_core_spec(core_name))
    target_nh = core_spec["target_l"] * 1e9

    print(
        f"=== Bayesian Optimization x {len(config.backends)} backends | "
        f"core {core_name} | target {target_nh:.1f} nH ==="
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
        f"\nAll outputs saved under {os.path.join(config.output_dir, 'bayesian', core_name)}/<backend>/"
    )


if __name__ == "__main__":
    main()
