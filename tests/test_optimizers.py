"""The contract the five optimizers share: one objective, one stopping rule,
one geometry map, and a gradient that is the derivative of the discrete
problem."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)

import objective as ob  # noqa: E402

# Coarse on purpose: every check here is about code, not accuracy.
run_mesh_mm = 2.0
run_target_nh = 350.0


def _spec(mesh_mm=run_mesh_mm):
    return ob.apply_config(
        ob.load_core_spec("pq_40x40"),
        target_l_nh=run_target_nh,
        mesh_size_mm=mesh_mm,
        material="N87",
    )


def test_loss_is_volume_plus_band_hinge_when_conditions_are_off():
    """With config.constraint_weight = 0 the loss is v / V_e plus the band hinge on L."""
    spec = _spec()
    old = ob.config.constraint_weight
    ob.config.constraint_weight = 0.0
    try:
        *_, loss_and_l, _ = ob.setup(spec, "lineax")
        x = ob.x0_from_spec(spec)
        loss, l = loss_and_l(jnp.asarray(x))
        target = float(spec["target_l"])
        off = float(l) / target - 1.0
        v_ratio = ob.core_volume(spec, x) / ob.datasheet_volume(spec)
        assert 1.0 < v_ratio < 1.06, "the modelled datasheet geometry is above its V_e"
        expected = v_ratio + ob.config.band_weight * off**2
        assert abs(float(loss) - expected) / expected < 1e-9
        # A third of the way into the band the restoring force is three times
        # the steepest volume gradient in log coordinates (3, uniform scaling).
        third = ob.config.converge_tol_rel / 3.0
        force = 2.0 * ob.config.band_weight * third
        assert abs(force - 9.0) < 1e-9
    finally:
        ob.config.constraint_weight = old


def test_penalty_is_the_weighted_hinge_of_the_violated_conditions():
    """At a 50 nH target the rated amp-turns are 872, and the datasheet core
    (a few hundred nH) then runs its post far above B_MAX: the penalty is
    exactly w times the squared hinges of the violated conditions and nothing else."""
    spec = _spec()
    spec["target_l"] = 50e-9
    x = ob.x0_from_spec(spec)
    old = ob.config.constraint_weight

    ob.config.constraint_weight = 0.0
    try:
        _, _, _, _, geom_fn, loss_pure, _ = ob.setup(spec, "lineax")
        pure, l = loss_pure(jnp.asarray(x))
        pure, l = float(pure), float(l)
    finally:
        ob.config.constraint_weight = old

    w = 1.0
    ob.config.constraint_weight = w
    try:
        *_, loss_pen, _ = ob.setup(spec, "lineax")
        with_pen = float(loss_pen(jnp.asarray(x))[0])
    finally:
        ob.config.constraint_weight = old

    vol = float(geom_fn(jnp.asarray(x))[3])
    g = np.asarray(
        ob.constraint_ratios(
            ob.params_dict_from_x(x, spec),
            l,
            vol,
            ob.operating_point(spec),
            ob.datasheet_volume(spec),
        )
    )
    assert g[ob.config.constraint_names.index("b_post")] > 0, (
        "the datasheet post must saturate at 50 nH"
    )
    expected = w * float(np.sum(np.maximum(g + ob.config.constraint_margin, 0.0) ** 2))
    assert abs((with_pen - pure) - expected) / expected < 1e-6


def test_conditions_are_what_the_report_checks(monkeypatch):
    """The ratios the loss penalises and the ratios design_checks reports are
    the same function of the same design."""
    monkeypatch.setattr(ob.config, "constraint_weight", 1.0)
    spec = _spec()
    _, _, _, _, geom_fn, loss_and_l, _ = ob.setup(spec, "lineax")
    x = ob.x0_from_spec(spec)
    loss, l = loss_and_l(jnp.asarray(x))
    loss, l = float(loss), float(l)
    vol = float(geom_fn(jnp.asarray(x))[3])
    checks = ob.design_checks(spec, x, l)
    g = ob.constraint_ratios(
        ob.params_dict_from_x(x, spec), l, vol, ob.operating_point(spec), ob.datasheet_volume(spec)
    )
    target = float(spec["target_l"])
    band = float(ob.band_penalty(l, target))
    volume_term = vol / ob.datasheet_volume(spec)
    hinge = ob.config.constraint_weight * float(
        np.sum(np.maximum(np.asarray(g) + ob.config.constraint_margin, 0.0) ** 2)
    )
    assert abs((loss - band - volume_term) - hinge) <= 1e-9 + 1e-6 * hinge
    for name, gi in zip(ob.config.constraint_names, np.asarray(g)):
        assert abs(checks["ratios"][name] - float(gi)) < 1e-12, name
    assert checks["met"]["b_post"] and checks["met"]["b_yoke"]
    assert checks["met"]["b_outer_leg"] and checks["met"]["copper_fit"]
    for name in ("winding", "clearance", "envelope"):
        assert checks["met"][name], f"the datasheet core is on the limit of {name}"
        assert abs(checks["ratios"][name]) < 1e-9, name
    # The modelled ferrite volume of the datasheet geometry is above its V_e.
    assert 0.0 < checks["ratios"]["volume"] < 0.06
    assert 0.0 < checks["b_post_t"] < ob.config.b_max_t


def test_traced_volume_matches_the_reporting_volume():
    """The volume inside the jitted loss equals the numpy volume in the record."""
    spec = _spec()
    _, n_r, n_z, _, geom_fn, _, _ = ob.setup(spec, "lineax")
    x_ds = ob.x0_from_spec(spec)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = x_ds * (1.0 + rng.uniform(-0.3, 0.3, size=len(x_ds)))
        traced = float(geom_fn(jnp.asarray(x))[3])
        reported = ob.core_volume(spec, x)
        assert abs(traced - reported) / reported < 1e-9


def test_all_five_optimizers_share_one_loss():
    """Every optimizer obtains its loss from objective and carries no private copy."""
    import importlib
    import inspect

    assert ob.make_loss_fn is ob.make_loss_and_l, (
        "make_loss_fn is no longer an alias of make_loss_and_l"
    )

    for name in ("adam", "lbfgs", "cmaes", "ga", "bayesian"):
        src = inspect.getsource(importlib.import_module(name))
        assert "ob.setup(" in src or "make_loss_fn(" in src, (
            f"{name} does not obtain its loss from objective"
        )
        assert "- target_l)" not in src.replace(" ", ""), (
            f"{name} appears to build its own inductance error term; the five "
            "must minimise one identical objective"
        )


def test_adjoint_gradient_matches_finite_difference():
    """The adjoint gradient of L agrees with central differences in direction and norm.

    Checked where every interface sits at a cell centre: the fill map has
    kinks at cell edges, and the datasheet geometry sits exactly on them.
    """
    d = run_mesh_mm * 1e-3
    spec = _spec()
    _, n_r, n_z, _, _, loss_and_l, _ = ob.setup(spec, "lineax")
    # Post radius 3.5 d, window 6 d wide and 15 d tall, yoke 4 d, gap d, in config.vary_keys order.
    x = np.array([2 * 9.5 * d, 2 * 3.5 * d, 2 * 11.5 * d, 2 * 7.5 * d, 0.0005, 2 * 0.5 * d])

    def l_only(xx):
        return loss_and_l(xx)[1]

    g = np.asarray(jax.grad(l_only)(jnp.asarray(x)))
    assert np.all(np.isfinite(g)), "adjoint gradient contains NaN/Inf"

    # Direction and norm, not per component: three parameters barely move L here.
    fd = np.zeros_like(g)
    for i in range(len(x)):
        h = 0.01 * d
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd[i] = (float(l_only(jnp.asarray(xp))) - float(l_only(jnp.asarray(xm)))) / (2 * h)

    cos = float(np.dot(g, fd) / (np.linalg.norm(g) * np.linalg.norm(fd)))
    assert cos > 0.999, f"adjoint direction is {np.degrees(np.arccos(cos)):.2f} deg off FD"
    nrm = abs(np.linalg.norm(g) - np.linalg.norm(fd)) / np.linalg.norm(fd)
    assert nrm < 0.05, f"adjoint magnitude off by {nrm:.1%}"


def test_race_has_a_finish_line_and_no_budget():
    """Only CONVERGE_ROUNDS consecutive rounds on the finish line stop the run."""
    s = ob.TargetReached(350e-9, weight=0.0)
    assert s.enabled is True
    assert s.update(300e-9, solves=10**9) is False
    for i in range(ob.config.converge_rounds - 1):
        assert s.update(350e-9, solves=i + 1) is False, (
            "a single pass through the band is not convergence"
        )
    assert s.update(350e-9, solves=ob.config.converge_rounds) is True
    assert "converged" in s.stop_message()


def test_conditions_side_of_the_finish_line_is_read_off_the_loss():
    """The penalty loss - volume - band must be at most w * (margin + tol)^2; no loss fails safe."""
    spec = _spec()
    w = ob.config.constraint_weight
    tol, m = ob.config.converge_tol_rel, ob.config.constraint_margin
    s = ob.TargetReached(350e-9, core_spec=spec, weight=w)
    x = ob.x0_from_spec(spec)
    l = (350 + 0.5) * 1e-9
    base = ob.core_volume(spec, x) / ob.datasheet_volume(spec) + float(ob.band_penalty(l, 350e-9))
    assert s.update(l, solves=1, x=x) is False and s.streak == 0, "no loss -> not met"
    s.update(l, solves=2, x=x, loss=base + w * (m + 2 * tol) ** 2)
    assert s.streak == 0, "a 2*tol violation is not on the finish line"
    for i in range(ob.config.converge_rounds):
        stop = s.update(l, solves=3 + i, x=x, loss=base + w * (m + 0.5 * tol) ** 2)
    assert stop is True and s.summary()["converged"] is True
    assert s.summary()["finish_line"] == "inductance band, physical conditions and settled volume"


def test_leaving_the_band_restarts_the_confirmation_streak():
    """A pass through the band resets the streak; cost is charged from the lasting entry."""
    s = ob.TargetReached(350e-9, weight=0.0)
    assert s.update(350e-9, solves=18) is False
    assert s.update(335e-9, solves=20) is False
    assert s.streak == 0
    for i in range(ob.config.converge_rounds):
        stop = s.update(350e-9, solves=100 + 2 * i)
    assert stop is True
    assert s.converged_solves == 100, "charged from the lasting entry"
    assert s.first_round_in_tol == 1, "the fluke touch is still recorded"


def test_settle_accepts_an_in_band_proposal_at_an_optimizer_s_own_stop():
    """An optimizer that stops on its own with an in-band proposal has converged."""
    s = ob.TargetReached(350e-9, weight=0.0)
    s.update(350e-9, solves=12)
    assert s.converged_round is None
    s.settle()
    assert s.converged_solves == 12
    assert s.summary()["converged"] is True


def test_settle_charges_a_self_stopped_run_up_to_where_its_volume_settled():
    """A streak whose core kept shrinking is charged from the round after which
    the volume stayed within the tolerance of the final one, not from its start."""
    spec = _spec()
    s = ob.TargetReached(run_target_nh * 1e-9, core_spec=spec, weight=0.0)
    x = ob.x0_from_spec(spec)
    i_leg, i_win = (
        ob.config.vary_keys.index("leg_height"),
        ob.config.vary_keys.index("window_height"),
    )
    yoke = 0.5 * (x[i_leg] - x[i_win])
    # In band from round 1, the yoke thinning by 5 % a round until round 6.
    for k, scale in enumerate((1.0, 0.95, 0.90, 0.85, 0.80, 0.80, 0.80, 0.80)):
        xk = x.copy()
        xk[i_leg] = x[i_win] + 2.0 * yoke * scale
        s.update(run_target_nh * 1e-9, solves=2 * (k + 1), x=xk, loss=1.0)
    assert s.streak == 8 and s.converged_round is None
    s.settle()
    assert s.converged_round == 5, "the volume last changed between rounds 4 and 5"
    assert s.converged_solves == 10
    assert s.summary()["converged"] is True


def test_tolerance_is_relative_to_the_target():
    for target_nh in (150.0, 350.0, 2000.0):
        s = ob.TargetReached(target_nh * 1e-9)
        assert abs(s.tol_nh - ob.config.converge_tol_rel * target_nh) < 1e-12


def test_tolerance_sits_above_the_discretisation_floor():
    """Refining 0.1 -> 0.08 mm moves L by ~0.4 nH; a tighter band would rank mesh noise."""
    floor_nh = 0.4
    assert ob.config.converge_tol_rel * run_target_nh > floor_nh, (
        f"tolerance {ob.config.converge_tol_rel * run_target_nh:.2f} nH is below the "
        f"{floor_nh} nH discretisation floor at {run_target_nh} nH"
    )


def test_every_log_coordinate_is_a_geometry_with_a_finite_loss():
    """There is no floor: any real log coordinate maps to positive lengths, and
    even a window narrower than its clearances gives a finite loss (no coil,
    no current) rather than a NaN system."""
    spec = _spec()
    *_, loss_and_l, _ = ob.setup(spec, "lineax")
    f_rel = ob.make_relative_loss(loss_and_l, spec)
    rng = np.random.default_rng(0)
    d = len(ob.config.vary_keys)
    narrow = np.zeros(d)
    # An inner leg inside the post: a negative window, which must still be finite.
    narrow[ob.config.vary_keys.index("leg_inner_diameter")] = -1.0
    for v in (
        np.zeros(d),
        np.full(d, -8.0),
        np.full(d, 5.0),
        rng.uniform(-6.0, 3.0, size=d),
        narrow,
        np.full(d, -1e6),  # a line-search trial far outside any geometry
    ):
        x = np.asarray(ob.from_relative(v, spec))
        assert np.all(x > 0.0) and np.all(np.isfinite(x))
        loss, l = f_rel(jnp.asarray(v))
        assert np.isfinite(float(loss)) and np.isfinite(float(l)), v


def test_domain_condition_bites_when_the_core_leaves_the_fem_domain():
    from new_fem_engine.geometry import geometry_settings

    spec = _spec()
    x_ds = ob.x0_from_spec(spec)
    op = ob.operating_point(spec)
    v_ds = ob.datasheet_volume(spec)
    i_domain = ob.config.constraint_names.index("domain")

    def g_domain(x):
        p = ob.params_dict_from_x(x, spec)
        g = ob.constraint_ratios(p, 250e-9, ob.core_volume(spec, x), op, v_ds)
        return float(np.asarray(g)[i_domain])

    assert g_domain(x_ds) < 0, "the datasheet core lies inside the FEM domain"
    r_outer, z_top = ob.core_extent(ob.params_dict_from_x(x_ds, spec))
    assert (
        float(r_outer) < geometry_settings.fixed_bounds[1]
        and float(z_top) < geometry_settings.fixed_bounds[3]
    )
    wide = x_ds.copy()
    wide[ob.config.vary_keys.index("leg_inner_diameter")] *= 2.0
    assert g_domain(wide) > 0, "a core past the outer wall must violate the domain condition"
    tall = x_ds.copy()
    tall[ob.config.vary_keys.index("leg_height")] *= 2.0
    assert g_domain(tall) > 0, "a core past the top wall must violate the domain condition"


def test_datasheet_referenced_conditions_bite_on_the_wrong_side():
    """A smaller window or a thinner clearance than the datasheet's violates
    winding or clearance; a wider or taller core violates envelope; a bigger
    core violates volume.  The opposite moves do not."""
    spec = _spec()
    x_ds = ob.x0_from_spec(spec)
    op = ob.operating_point(spec)
    v_ds = ob.datasheet_volume(spec)

    def g_of(x, name):
        p = ob.params_dict_from_x(x, spec)
        g = ob.constraint_ratios(p, 250e-9, ob.core_volume(spec, x), op, v_ds)
        return float(np.asarray(g)[ob.config.constraint_names.index(name)])

    for key, name, violated_by in (
        ("window_height", "winding", "smaller"),
        ("coil_clearance", "clearance", "smaller"),
        ("leg_inner_diameter", "envelope", "bigger"),
        ("leg_height", "envelope", "bigger"),
        ("leg_height", "volume", "bigger"),
    ):
        smaller, bigger = x_ds.copy(), x_ds.copy()
        smaller[ob.config.vary_keys.index(key)] *= 0.8
        bigger[ob.config.vary_keys.index(key)] *= 1.2
        if name != "volume":
            assert abs(g_of(x_ds, name)) < 1e-9, f"the datasheet core sits on the {name} limit"
        bad, good = (smaller, bigger) if violated_by == "smaller" else (bigger, smaller)
        # The envelope is a max of two ratios, so relaxing one dimension leaves
        # the other on its limit: the good side is <= rather than <.
        assert g_of(bad, name) > g_of(x_ds, name) >= g_of(good, name), name
        assert g_of(bad, name) > 0, name
    # The datasheet geometry itself is above V_e; a thinner yoke brings it under.
    assert g_of(x_ds, "volume") > 0
    thin = x_ds.copy()
    i_leg, i_win = (
        ob.config.vary_keys.index("leg_height"),
        ob.config.vary_keys.index("window_height"),
    )
    thin[i_leg] = x_ds[i_win] + 0.8 * (x_ds[i_leg] - x_ds[i_win])
    assert g_of(thin, "volume") < 0


def test_x0_round_trips_through_the_parameter_dict():
    spec = _spec()
    x = ob.x0_from_spec(spec)
    p = ob.params_dict_from_x(x, spec)
    for i, k in enumerate(ob.config.vary_keys):
        assert abs(p[k] - x[i]) < 1e-15
    assert np.all(x > 0.0)


@pytest.mark.parametrize("name", ["adam", "lbfgs", "cmaes", "ga", "bayesian"])
def test_every_optimizer_exposes_run_one_backend(name):
    import importlib

    mod = importlib.import_module(name)
    fn = getattr(mod, "run_one_backend", None)
    assert callable(fn), f"{name}.run_one_backend is missing"
    import inspect

    params = inspect.signature(fn).parameters
    assert "seed" in params, (
        f"{name}.run_one_backend takes no seed -- main_run_optimizer's repeats "
        "would all be the same run"
    )


@pytest.mark.parametrize("name", ["cmaes", "ga", "bayesian"])
def test_stochastic_optimizers_consume_their_seed(name):
    """A repeat that ignored its seed would give identical runs; checked on the source."""
    import importlib

    mod = importlib.import_module(name)
    src = inspect_source(mod)
    assert "default_rng(seed)" in src or "PRNGKey(seed)" in src or "seed=seed" in src, (
        f"{name} never derives randomness from seed"
    )


def inspect_source(mod):
    import inspect

    return inspect.getsource(mod)


def test_backends_list_matches_what_solver_options_accepts():
    for bk in ob.config.backends:
        fwd, adj, traceable = ob.solver_options_for(bk, (11, 11))
        assert isinstance(fwd, dict) and isinstance(adj, dict)
    with pytest.raises(ValueError):
        ob.solver_options_for("not_a_backend", (11, 11))
