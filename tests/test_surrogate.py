"""The Bayesian optimizer's Gaussian-process surrogate (GPJax): the fit must
model a smooth function and be uncertain away from the data, the predictive
variance must include the observation noise, and expected improvement must
behave as an acquisition function."""

import os
import sys

import jax.numpy as jnp
import numpy as np
import paramax

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)

import bayesian as bo  # noqa: E402

dim = 6


def _smooth(u):
    return np.sin(3.0 * u[:, 0]) + 0.5 * u[:, 1] ** 2 - 0.3 * u[:, 2]


def _dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.5, 1.5, size=(n, dim))
    z = _smooth(u)
    z = (z - z.mean()) / z.std()
    return u, z


def _hyperparameters(gp):
    """Return (kernel variance, observation noise variance) of a fitted posterior."""
    posterior = paramax.unwrap(gp[0])
    return float(posterior.prior.kernel.variance), float(posterior.likelihood.obs_stddev) ** 2


def test_fit_reproduces_a_smooth_function_and_is_uncertain_away_from_the_data():
    u, z = _dataset(60)
    gp = bo.fit_gp(u, z)
    mean, var = bo.predict_marginals(gp, jnp.asarray(u))
    assert np.sqrt(np.mean((np.asarray(mean) - z) ** 2)) < 0.1
    u_test, z_test = _dataset(40, seed=7)
    mean_test, _ = bo.predict_marginals(gp, jnp.asarray(u_test))
    assert np.mean((np.asarray(mean_test) - z_test) ** 2) < 0.25 * np.var(z_test)
    # Far from every observation the variance returns to the prior plus the noise.
    far = jnp.asarray(np.full((1, dim), 5.0))
    _, var_far = bo.predict_marginals(gp, far)
    kernel_variance, noise_variance = _hyperparameters(gp)
    assert float(np.max(np.asarray(var))) < float(var_far[0])
    assert abs(float(var_far[0]) - (kernel_variance + noise_variance)) < 1e-6


def test_predictive_variance_includes_the_observation_noise():
    u, z = _dataset(40)
    gp = bo.fit_gp(u, z)
    _, var = bo.predict_marginals(gp, jnp.asarray(u))
    _, noise_variance = _hyperparameters(gp)
    assert np.all(np.asarray(var) >= noise_variance * (1.0 - 1e-9))


def test_fit_is_deterministic():
    u, z = _dataset(30)
    a, b = bo.fit_gp(u, z), bo.fit_gp(u, z)
    cand = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, size=(200, dim)))
    for x, y in zip(bo.predict_marginals(a, cand), bo.predict_marginals(b, cand)):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-14)


def test_expected_improvement_is_nonnegative_and_zero_where_nothing_can_improve():
    u, z = _dataset(60)
    gp = bo.fit_gp(u, z)
    cand = jnp.asarray(np.random.default_rng(3).uniform(0.5, 1.5, size=(2000, dim)))
    ei = np.asarray(bo.expected_improvement(gp, cand, z_best=float(z.min())))
    assert np.all(np.isfinite(ei)) and np.all(ei >= 0.0) and ei.max() > 0.0
    # With the incumbent far below anything the model predicts, EI vanishes.
    ei_none = np.asarray(bo.expected_improvement(gp, cand, z_best=float(z.min()) - 50.0))
    assert np.all(ei_none < 1e-12)
