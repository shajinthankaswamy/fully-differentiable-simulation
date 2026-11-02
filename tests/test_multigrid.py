"""The multigrid V-cycle on the tangent the study actually solves: the
0.1 mm production grid, its Galerkin hierarchy and its use inside CG."""

import os
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp
import lineax
import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimizers")
)

import new_fem_engine.solver as slv  # noqa: E402
import objective as ob  # noqa: E402
from new_fem_engine import multigrid  # noqa: E402

run_mesh_mm = 0.1
spd = frozenset({lineax.symmetric_tag, lineax.positive_semidefinite_tag})


@pytest.fixture(scope="module")
def tangent():
    """(a_bcoo, b, grid shape, problem, coil rects) at the datasheet design, 0.1 mm."""
    spec = ob.apply_config(
        ob.load_core_spec("pq_40x40"), target_l_nh=250.0, mesh_size_mm=run_mesh_mm, material="N87"
    )
    prob, n_r, n_z, _, geometry_from_x, _, _ = ob.setup(spec, "lineax")
    fill, prim, sec, _ = geometry_from_x(jnp.asarray(ob.x0_from_spec(spec)))
    prob.set_params((fill, prim, sec))
    dofs = np.zeros(prob.num_total_dofs_all_vars)
    res_list, jac = prob.newton_update(prob.unflatten_fn_sol_list(dofs))
    res = jax.flatten_util.ravel_pytree(res_list)[0]
    b = -slv.apply_bc_vec(res, dofs, prob)
    return slv.get_a_bcoo(prob, jac), np.asarray(b), (n_r + 1, n_z + 1), prob, (prim, sec)


def _mg_cg(a, b, shape, tol):
    """Solve with lineax CG + the multigrid preconditioner; return (x, steps)."""
    opts = {**ob.config.lineax_opts, "grid_shape": shape}
    op = lineax.FunctionLinearOperator(
        slv._SparseMatvec(a),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        tags=spd,
        closure_convert=False,
    )
    precond = lineax.TaggedLinearOperator(
        lineax.FunctionLinearOperator(
            slv._make_preconditioner(a, opts),
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            tags=spd,
            closure_convert=False,
        ),
        spd,
    )
    sol = lineax.linear_solve(
        op,
        b,
        solver=lineax.CG(rtol=tol, atol=tol, max_steps=opts["max_steps"]),
        options={"preconditioner": precond},
        throw=False,
    )
    return sol.value, sol.stats["num_steps"]


def test_stencil_matches_the_sparse_matvec(tangent):
    a, _, shape, _, _ = tangent
    assert multigrid.verify_stencil(a, shape) < 1e-12


def test_hierarchy_coarsens_the_production_grid_to_the_floor(tangent):
    a, _, shape, _, _ = tangent
    _, shapes = multigrid.make_preconditioner(
        a, shape, ob.config.lineax_opts["mg_sweeps"], ob.config.lineax_opts["mg_min_dofs"]
    )
    assert shapes[0] == shape
    sizes = [s[0] * s[1] for s in shapes]
    assert all(b < a_ for a_, b in zip(sizes, sizes[1:])), "every level must be strictly smaller"
    assert sizes[-1] <= ob.config.lineax_opts["mg_min_dofs"]
    assert len(shapes) >= 5, f"only {len(shapes)} levels on a {shape} grid"


def test_vcycle_is_symmetric_for_cg(tangent):
    a, b, shape, _, _ = tangent
    precond, _ = multigrid.make_preconditioner(
        a, shape, ob.config.lineax_opts["mg_sweeps"], ob.config.lineax_opts["mg_min_dofs"]
    )
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(b.shape[0]))
    v = jnp.asarray(rng.standard_normal(b.shape[0]))
    lhs = float(jnp.vdot(u, precond(v)))
    rhs = float(jnp.vdot(precond(u), v))
    assert abs(lhs - rhs) <= 1e-10 * abs(lhs)


def test_vcycle_leaves_dirichlet_dofs_at_zero(tangent):
    a, b, shape, prob, _ = tangent
    precond, _ = multigrid.make_preconditioner(
        a, shape, ob.config.lineax_opts["mg_sweeps"], ob.config.lineax_opts["mg_min_dofs"]
    )
    z = np.asarray(precond(jnp.asarray(b)))
    bc = slv._bc_dofs(prob)
    assert np.all(z[bc] == 0.0)
    assert np.any(z != 0.0)


def test_mg_cg_reaches_the_reference_inductance_in_few_steps(tangent):
    a, b, shape, prob, (prim, sec) = tangent
    solve = jax.jit(lambda a_, b_, tol: _mg_cg(a_, b_, shape, tol), static_argnums=2)
    x_work, steps = solve(a, jnp.asarray(b), ob.config.lineax_opts["rtol"])
    x_ref, _ = solve(a, jnp.asarray(b), 1e-10)

    def l_of(x):
        return float(prob.compute_inductance(prob.unflatten_fn_sol_list(x), prim, sec))

    l_work, l_ref = l_of(x_work), l_of(x_ref)
    assert abs(l_work - l_ref) <= 1e-8 * abs(l_ref)
    # 64-70 steps across the whole gap range on 2026-09-16; Jacobi needs 768.
    assert int(steps) < 150, f"multigrid CG took {int(steps)} steps"
