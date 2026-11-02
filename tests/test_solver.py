"""Sparse assembly, the linear-solver backends and the adjoint on a small problem."""

import sys
from pathlib import Path

import feax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import helpers
import new_fem_engine.solver as slv
from new_fem_engine.generate_mesh import create_mesh, rectangle_mesh
from new_fem_engine.problem import MagnetostaticProblem

core_spec = {
    "params": {
        "center_post_diameter": 0.015,
        "leg_inner_diameter": 0.0275,
        "leg_height": 0.044,
        "window_height": 0.0294,
        "coil_clearance": 0.0005,
        "mur": 1680.0,
        "a_e": 122.6e-6,
        "l_e": 44.4e-3,
        "sigma_l_over_a": 362.0,
        "al": 4300.0,
        "material": "N87",
    }
}


def make_problem(n_r=6, n_z=8):
    msh = rectangle_mesh(n_r, n_z, 0.0, 0.035, -0.03, 0.03)
    mesh = create_mesh(msh, "quad")
    locs = [
        lambda pt: jnp.isclose(pt[0], 0.035),
        lambda pt: jnp.isclose(pt[0], 0.0),
        lambda pt: jnp.isclose(pt[1], 0.03),
        lambda pt: jnp.isclose(pt[1], -0.03),
    ]
    prob = MagnetostaticProblem(
        mesh=mesh,
        vec=1,
        dim=2,
        dirichlet_bc_info=[locs, [0, 0, 0, 0], [lambda pt: 0.0] * 4],
        additional_info=(core_spec, 0.001),
    )
    prob.set_params(jnp.ones((prob.num_cells, 4)))
    return prob


@pytest.fixture(scope="module")
def prob():
    return make_problem()


def reset_params(prob):
    prob.set_params(jnp.ones((prob.num_cells, 4)))


def dirichlet_dofs(prob):
    fe = prob.fes[0]
    return np.concatenate(
        [
            np.asarray(fe.node_inds_list[i]) * fe.vec + np.asarray(fe.vec_inds_list[i])
            for i in range(len(fe.node_inds_list))
        ]
    )


forward_backends = [
    ("lineax-bicgstab", {"newton": {"linear": {"lineax_solver": {"solver": "bicgstab"}}}}),
    ("lineax-cg", {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}}),
    ("lineax-lu", {"newton": {"linear": {"lineax_solver": {"solver": "lu"}}}}),
    ("lineax-auto", {"newton": {"linear": {"lineax_solver": {}}}}),
    ("feax-krylov", {"newton": {"linear": {"feax_solver": {}}}}),
]

adjoint_backends = [
    ("lineax-cg", {"lineax_solver": {"solver": "cg"}}),
    ("feax-krylov", {"feax_solver": {}}),
]


def test_get_a_symmetric_spd(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    a_bcoo = slv.get_a_bcoo(prob)
    assert a_bcoo.shape == (prob.num_total_dofs_all_vars, prob.num_total_dofs_all_vars)
    assert isinstance(a_bcoo, slv.BCOO)
    a_dense = np.asarray(a_bcoo.todense())
    assert np.allclose(a_dense, a_dense.T, atol=1e-6)
    assert np.linalg.eigvalsh(a_dense).min() > 0


def test_get_a_dirichlet_rows_unit(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    a_bcoo = np.asarray(slv.get_a_bcoo(prob).todense())
    identity = np.eye(a_bcoo.shape[0])
    for d in dirichlet_dofs(prob):
        assert np.allclose(a_bcoo[d], identity[d], atol=1e-8)


def test_newton_jacobian_matches_jacfwd(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    n = prob.num_total_dofs_all_vars
    tangent = helpers.dense_tangent(prob, n)

    def raw_res(x):
        return jax.flatten_util.ravel_pytree(prob.compute_residual([jnp.asarray(x)[:, None]]))[0]

    jac = np.asarray(jax.jacfwd(raw_res)(np.zeros(n)))
    np.testing.assert_allclose(tangent, jac, atol=1e-6)


@pytest.mark.parametrize("name,opts", forward_backends)
def test_solver_backend_forward(prob, name, opts):
    reset_params(prob)
    sol = slv.solver(prob, opts)
    psi = np.asarray(sol[0]).ravel()
    assert psi.shape == (prob.fes[0].num_total_nodes,)
    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) > 0
    assert np.allclose(psi[dirichlet_dofs(prob)], 0.0, atol=1e-10)


def test_solver_backends_agree(prob):
    ref = None
    for name, opts in forward_backends:
        reset_params(prob)
        psi = np.asarray(slv.solver(prob, opts)[0]).ravel()
        if ref is None:
            ref = psi
        else:
            np.testing.assert_allclose(psi, ref, atol=1e-8)


def test_solver_empty_options_uses_lineax_default(prob):
    reset_params(prob)
    psi = np.asarray(slv.solver(prob, {})[0]).ravel()
    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) > 0


def test_solver_legacy_flat_options(prob):
    reset_params(prob)
    psi = np.asarray(slv.solver(prob, {"lineax_solver": {"solver": "cg"}})[0]).ravel()
    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) > 0


def test_linear_solver_does_not_mutate_options(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    a_bcoo = slv.get_a_bcoo(prob)
    b = jnp.ones(prob.num_total_dofs_all_vars)
    opts = {}
    slv.linear_solver(a_bcoo, b, None, opts)
    assert opts == {}


def test_lineax_cg_matches_lu(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    a_bcoo = slv.get_a_bcoo(prob)
    b = jnp.ones(prob.num_total_dofs_all_vars)
    xl = np.asarray(slv.bcoo_solve_lineax(a_bcoo, b, None, {"solver": "lu"}))
    xc = np.asarray(slv.bcoo_solve_lineax(a_bcoo, b, None, {"solver": "cg"}))
    np.testing.assert_allclose(xl, xc, atol=1e-6)


def test_feax_solve_matches_lineax(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    a_bcoo = slv.get_a_bcoo(prob)
    b = jnp.ones(prob.num_total_dofs_all_vars)
    xf = np.asarray(slv.bcoo_solve_feax(a_bcoo, b, None, {}))
    xl = np.asarray(slv.bcoo_solve_lineax(a_bcoo, b, None, {"solver": "lu"}))
    np.testing.assert_allclose(xf, xl, atol=1e-6)


def test_feax_direct_options_rejected(prob):
    # A direct feax solver needs a scipy CSR matrix; it must fail loudly.
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    a_bcoo = slv.get_a_bcoo(prob)
    b = jnp.ones(prob.num_total_dofs_all_vars)
    with pytest.raises(NotImplementedError):
        slv.bcoo_solve_feax(a_bcoo, b, None, {"options": feax.DirectSolverOptions(solver="auto")})


@pytest.mark.parametrize("name,adjopts", adjoint_backends)
def test_adjoint_backend_agrees_with_reference(prob, name, adjopts):
    reset_params(prob)
    ref_opts = {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}}
    ref = slv.ad_wrapper(prob, ref_opts, {"lineax_solver": {"solver": "cg"}})
    fwd = slv.ad_wrapper(prob, ref_opts, adjopts)
    fill = jnp.ones((prob.num_cells, 4))

    def total(f):
        return jnp.sum(fwd(f)[0])

    grad = np.asarray(jax.grad(total)(fill))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) > 0

    def total_ref(f):
        return jnp.sum(ref(f)[0])

    grad_ref = np.asarray(jax.grad(total_ref)(fill))
    np.testing.assert_allclose(grad, grad_ref, rtol=1e-6, atol=1e-10)


def test_ad_wrapper_gradient_matches_finite_difference(prob):
    reset_params(prob)
    ref_opts = {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}}
    fwd = slv.ad_wrapper(prob, ref_opts, {"lineax_solver": {"solver": "cg"}})

    def total(fill):
        return jnp.sum(fwd(fill)[0])

    fill = jnp.ones((prob.num_cells, 4))
    grad = np.asarray(jax.grad(total)(fill))
    assert np.all(np.isfinite(grad))

    eps = 1e-4
    for idx in [(0, 0), (5, 2), (7, 3)]:
        e = np.zeros_like(np.asarray(fill))
        e[idx] = eps
        fp = float(total(jnp.asarray(np.asarray(fill) + e)))
        fm = float(total(jnp.asarray(np.asarray(fill) - e)))
        assert np.isclose(fp - fm, 2 * eps * grad[idx], rtol=1e-2)
