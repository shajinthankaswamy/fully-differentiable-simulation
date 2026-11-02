"""The solve is linear in the source current."""

import numpy as np

import helpers
import new_fem_engine.solver as slv


def _solve(prob, fill):
    prob.set_params(fill)
    sol = slv.solver(prob, helpers.sp)
    return sol, float(prob.compute_inductance(sol))


def test_l_independent_of_current():
    # current is baked into the jitted assembly at the first trace, so each
    # configuration gets a fresh system with the value set before solving.
    prob1, fill1, _ = helpers.build_system()
    _, l1 = _solve(prob1, fill1)

    prob10, fill10, _ = helpers.build_system()
    prob10.current = 10.0
    _, l10 = _solve(prob10, fill10)

    assert helpers.rel_err(l10, l1) < 1e-3, f"L changed with current: {l1:.6e} -> {l10:.6e}"


def test_psi_scales_linearly_with_current():
    prob1, fill1, _ = helpers.build_system()
    sol1, _ = _solve(prob1, fill1)
    psi1 = np.max(np.abs(np.asarray(sol1[0])))

    prob10, fill10, _ = helpers.build_system()
    prob10.current = 10.0
    sol10, _ = _solve(prob10, fill10)
    psi10 = np.max(np.abs(np.asarray(sol10[0])))

    assert psi1 > 0
    assert helpers.rel_err(psi10, 10.0 * psi1) < 1e-2, (
        f"psi not linear in I: {psi1:.6e} -> {psi10:.6e}"
    )
