"""The energy and flux-linkage inductances agree."""

import helpers
import new_fem_engine.solver as slv
from new_fem_engine.run_forward import linkage_inductance


def test_energy_and_linkage_l_agree():
    prob, fill, _ = helpers.build_system()
    l_energy = helpers.solve_l(prob, fill)

    prob.set_params(fill)
    sol = slv.solver(prob, helpers.sp)
    l_link = linkage_inductance(prob, sol)

    assert helpers.rel_err(l_link, l_energy) < 1e-3, (
        f"linkage {l_link:.6e} vs energy {l_energy:.6e}"
    )
