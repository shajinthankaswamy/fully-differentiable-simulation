"""The production configuration as one unit: the shared objective at the
0.1 mm working mesh with multigrid CG, the directional fill and the
physical conditions.  Every number here is a statement about that mesh."""

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

pytestmark = pytest.mark.slow

run_mesh_mm = 0.1
run_target_nh = 250.0
# Inductance of the datasheet PQ40/40 with its 1 mm gap as this code computes
# it on 2026-09-16.  A regression pin, not a physics check: the datasheet
# tabulates no gapped AL for this gap.
datasheet_design_l_nh = 276.51


def _spec():
    return ob.apply_config(
        ob.load_core_spec("pq_40x40"),
        target_l_nh=run_target_nh,
        mesh_size_mm=run_mesh_mm,
        material="N87",
    )


@pytest.fixture(scope="module")
def production():
    spec = _spec()
    prob, n_r, n_z, mesh_size, geometry_from_x, loss_and_l, traceable = ob.setup(spec, "lineax")
    assert traceable and (n_r, n_z) == (350, 600)
    return spec, prob, n_r, n_z, geometry_from_x, loss_and_l


def _l_nh(loss_and_l, x):
    return float(jax.jit(loss_and_l)(jnp.asarray(x))[1]) * 1e9


def test_datasheet_design_inductance_is_pinned(production):
    spec, _, _, _, _, loss_and_l = production
    l_nh = _l_nh(loss_and_l, ob.x0_from_spec(spec))
    assert abs(l_nh - datasheet_design_l_nh) <= 1e-3 * datasheet_design_l_nh


def test_gradient_at_the_working_mesh_is_finite_with_physical_signs(production):
    spec, _, _, _, _, loss_and_l = production
    x0 = jnp.asarray(ob.x0_from_spec(spec))
    (loss, l), grad = jax.jit(jax.value_and_grad(loss_and_l, has_aux=True))(x0)
    assert np.isfinite(float(loss)) and np.isfinite(float(l))
    assert np.all(np.isfinite(np.asarray(grad)))
    dl_dx = np.asarray(jax.jit(jax.grad(lambda x: loss_and_l(x)[1]))(x0))
    i_gap = ob.config.vary_keys.index("gap_size")
    i_post = ob.config.vary_keys.index("center_post_diameter")
    assert dl_dx[i_gap] < 0.0, "a wider gap must lower the inductance"
    assert dl_dx[i_post] > 0.0, "a wider post must raise the inductance"


def test_cg_tolerance_resolves_the_inductance(production):
    spec, prob, n_r, n_z, geometry_from_x, loss_and_l = production
    tight = {
        **ob.config.lineax_opts,
        "rtol": 1e-10,
        "atol": 1e-10,
        "grid_shape": (n_r + 1, n_z + 1),
    }
    loss_ref = ob.make_loss_and_l(
        prob,
        geometry_from_x,
        float(spec["target_l"]),
        {"newton": {"linear": {"lineax_solver": tight}}},
        {"lineax_solver": tight},
        True,
        spec,
    )
    x0 = ob.x0_from_spec(spec)
    l_work, l_ref = _l_nh(loss_and_l, x0), _l_nh(loss_ref, x0)
    # 1e-13 measured on 2026-09-16 across the gap range; the band is 3e-3.
    assert abs(l_work - l_ref) <= 1e-8 * l_ref


def test_sub_cell_gap_follows_the_series_reluctance(production):
    """Gaps below and around one cell are homogenised as a laminate, so L must
    track mu0 A / (g + l_e / mu_e) with no kink at the cell boundary."""
    spec, _, _, _, _, loss_and_l = production
    p = ob.base_params(spec)
    mu0 = 4e-7 * np.pi
    i_gap = ob.config.vary_keys.index("gap_size")
    x0 = ob.x0_from_spec(spec)
    gaps_mm = [0.05, 0.08, 0.10, 0.12, 0.15]
    ratios, l_values = [], []
    for g in gaps_mm:
        x = x0.copy()
        x[i_gap] = g * 1e-3
        l_fem = _l_nh(loss_and_l, x)
        l_model = mu0 * p["a_e"] / (g * 1e-3 + p["l_e"] / p["mur"]) * 1e9
        l_values.append(l_fem)
        ratios.append(l_fem / l_model)
    assert all(b < a for a, b in zip(l_values, l_values[1:])), l_values
    # 1.003-1.013 measured on 2026-09-16; fringing is negligible at these gaps.
    assert all(0.98 <= r <= 1.03 for r in ratios), ratios
    assert all(abs(b - a) < 0.01 for a, b in zip(ratios, ratios[1:])), ratios


def test_population_evaluator_paths_give_the_same_losses(production, monkeypatch):
    spec, _, _, _, _, loss_and_l = production
    x_ds = ob.x0_from_spec(spec)
    designs = np.stack([x_ds, x_ds * 1.05, x_ds * 0.95])

    monkeypatch.setattr(ob.config, "pop_chunk", 1)
    single = ob.make_population_evaluator(loss_and_l, True)(designs)
    monkeypatch.setattr(ob.config, "pop_chunk", 2)
    chunked = ob.make_population_evaluator(loss_and_l, True)(designs)

    for a, b in zip(single, chunked):
        assert np.allclose(a, b, rtol=1e-9, atol=0.0)
