"""Problem: assembly structure, kernels, residuals and Jacobians on small meshes."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import helpers
from new_fem_engine.generate_mesh import create_mesh, rectangle_mesh
from new_fem_engine.problem import Problem


class LaplaceProblem(Problem):
    def get_tensor_map(self):
        def tensor_map(u_grad):
            return u_grad

        return tensor_map


class MassProblem(Problem):
    def get_mass_map(self):
        def mass_map(u, x):
            return u

        return mass_map


class ParamLaplaceProblem(Problem):
    def set_params(self, params):
        self.internal_vars = [params]

    def get_tensor_map(self):
        def tensor_map(u_grad, param):
            return param * u_grad

        return tensor_map


class ParamMassProblem(Problem):
    def set_params(self, params):
        self.internal_vars = [params]

    def get_mass_map(self):
        def mass_map(u, x, param):
            return param * u

        return mass_map


class SurfaceMassProblem(MassProblem):
    def get_surface_maps(self):
        def surface_map(u, x):
            return u

        return [surface_map]


class CustomInitProblem(Problem):
    def custom_init(self, *args):
        self.init_args = args


class UniversalLaplaceProblem(Problem):
    def get_universal_kernel(self):
        def universal_kernel(
            cell_sol_flat, physical_quad_points, cell_shape_grads, cell_jxw, cell_v_grads_jxw
        ):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_shape_grads = cell_shape_grads[:, : self.fes[0].num_nodes, :]
            cell_v_grads_jxw = cell_v_grads_jxw[:, : self.fes[0].num_nodes, :, :]
            u_grads = jnp.sum(cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :], axis=1)
            val = jnp.sum(u_grads[:, None, :, :] * cell_v_grads_jxw, axis=(0, -1))
            return jax.flatten_util.ravel_pytree(val)[0]

        return universal_kernel


class UniversalSurfaceMassProblem(MassProblem):
    def get_universal_kernels_surface(self):
        def universal_surface_kernel(
            cell_sol_flat,
            physical_surface_quad_points,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
        ):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, : self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]
            u = jnp.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            val = jnp.sum(
                u[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None],
                axis=0,
            )
            return jax.flatten_util.ravel_pytree(val)[0]

        return [universal_surface_kernel]


class TwoBoundarySurfaceProblem(MassProblem):
    def get_surface_maps(self):
        def surface_map(u, x):
            return u

        return [surface_map, surface_map]


class SurfaceParamMassProblem(MassProblem):
    def set_params(self, params):
        self.internal_vars_surfaces = [(params,)]

    def get_surface_maps(self):
        def surface_map(u, x, param):
            return param * u

        return [surface_map]


domain_area = 0.01 * 0.02  # physical domain area (r in [0,0.01], z in [0,0.02])


def make_mesh(n_r=3, n_z=4):
    msh = rectangle_mesh(n_r, n_z, 0.0, 0.01, 0.0, 0.02)
    return create_mesh(msh, "quad")


def ones_sol(num_nodes=20, vec=1):
    return [jnp.ones((num_nodes, vec))]


def r_sol(mesh):
    r = np.asarray(mesh.points)[:, 0]
    return [jnp.stack([jnp.asarray(r)], axis=1)]


def test_single_var_inputs_wrapped():
    prob = Problem(mesh=make_mesh(), vec=1, dim=2)
    assert prob.num_vars == 1
    assert isinstance(prob.mesh, list)
    assert isinstance(prob.vec, list)
    assert isinstance(prob.ele_type, list)
    assert len(prob.fes) == 1


def test_problem_attributes():
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    assert prob.num_cells == 12
    assert prob.num_total_dofs_all_vars == 20
    assert prob.offset == [0]
    assert prob.cells_flat.shape == (12, 4)
    assert prob.num_nodes_cumsum.tolist() == [0, 4]
    assert prob.boundary_inds_list == []


def test_assembly_inds_no_boundaries():
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    # 12 cells x (4x4 local matrix) = 192 entries, no face contributions
    assert prob.row_inds.shape == (192,)
    assert prob.col_inds.shape == (192,)
    assert prob.row_inds.min() >= 0 and prob.row_inds.max() <= 19
    assert prob.col_inds.min() >= 0 and prob.col_inds.max() <= 19


def test_geometric_quantities():
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    assert prob.jxw.shape == (12, 1, 4)
    assert np.isclose(np.asarray(prob.jxw).sum(), domain_area)
    assert prob.shape_grads.shape == (12, 4, 4, 2)
    assert prob.v_grads_jxw.shape == (12, 4, 4, 1, 2)
    assert prob.physical_quad_points.shape == (12, 4, 2)


def test_multi_var_attributes():
    mesh = make_mesh()
    prob = Problem(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=["QUAD4", "QUAD4"])
    assert prob.num_vars == 2
    assert prob.num_total_dofs_all_vars == 40
    assert prob.offset == [0, 20]
    assert prob.cells_flat.shape == (12, 8)
    assert prob.row_inds.max() == 39


def test_set_params_not_implemented():
    prob = Problem(mesh=make_mesh(), vec=1, dim=2)
    with pytest.raises(NotImplementedError):
        prob.set_params(None)


def test_mass_residual_sum_is_area():
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    res = prob.compute_residual(ones_sol())
    assert res[0].shape == (20, 1)
    assert np.isclose(np.asarray(res[0]).sum(), domain_area)


def test_mass_matrix_symmetric_and_rowsum():
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    res = prob.newton_update(ones_sol())
    tangent = helpers.dense_tangent(prob, 20)
    assert np.allclose(tangent, tangent.T, atol=1e-6)
    assert np.allclose(tangent @ np.ones(20), np.asarray(res[0][0]).reshape(-1), atol=1e-6)


def test_newton_residual_matches_compute_residual():
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    r1 = np.asarray(prob.compute_residual(ones_sol())[0])
    r2 = np.asarray(prob.newton_update(ones_sol())[0][0])
    assert np.allclose(r1, r2, atol=1e-6)


def test_mass_residual_vec_two():
    prob = MassProblem(mesh=make_mesh(), vec=2, dim=2)
    res = prob.compute_residual(ones_sol(vec=2))
    assert res[0].shape == (20, 2)
    assert np.isclose(np.asarray(res[0]).sum(), 2 * domain_area)


def test_laplace_constant_sol_zero():
    prob = LaplaceProblem(mesh=make_mesh(), vec=1, dim=2)
    res = np.asarray(prob.compute_residual(ones_sol())[0])
    assert np.allclose(res, 0.0, atol=1e-6)


def test_laplace_linear_sol_matches_geometric_assembly():
    mesh = make_mesh()
    prob = LaplaceProblem(mesh=mesh, vec=1, dim=2)
    res = np.asarray(prob.compute_residual(r_sol(mesh))[0]).reshape(-1)

    fe = prob.fes[0]
    grad_int = (np.asarray(fe.shape_grads) * np.asarray(fe.jxw)[:, :, None, None]).sum(axis=1)
    cells = np.asarray(fe.cells)
    expected = np.zeros(fe.num_total_nodes)
    for c in range(fe.num_cells):
        for a in range(fe.num_nodes):
            expected[cells[c, a]] += grad_int[c, a, 0]
    assert np.allclose(res, expected, atol=1e-6)


def test_laplace_matrix_symmetric():
    mesh = make_mesh()
    prob = LaplaceProblem(mesh=mesh, vec=1, dim=2)
    prob.newton_update(r_sol(mesh))
    tangent = helpers.dense_tangent(prob, 20)
    assert np.allclose(tangent, tangent.T, atol=1e-6)


def test_param_laplace_equals_plain():
    mesh = make_mesh()
    plain = np.asarray(LaplaceProblem(mesh=mesh, vec=1, dim=2).compute_residual(r_sol(mesh))[0])

    prob = ParamLaplaceProblem(mesh=mesh, vec=1, dim=2)
    prob.set_params(jnp.ones((12, 4)))
    assert np.allclose(np.asarray(prob.compute_residual(r_sol(mesh))[0]), plain, atol=1e-6)

    prob.set_params(jnp.full((12, 4), 2.0))
    assert np.allclose(np.asarray(prob.compute_residual(r_sol(mesh))[0]), 2.0 * plain, atol=1e-6)


def test_param_differentiability_grad_is_jxw():
    mesh = make_mesh()
    prob = ParamMassProblem(mesh=mesh, vec=1, dim=2)

    def total_res(params):
        prob.set_params(params)
        return prob.compute_residual(ones_sol())[0].sum()

    grad = np.asarray(jax.grad(total_res)(jnp.ones((12, 4))))
    jxw = np.asarray(prob.jxw)[:, 0, :]
    assert np.allclose(grad, jxw, atol=1e-6)


def test_surface_boundary_attributes():
    mesh = make_mesh()
    prob = SurfaceMassProblem(
        mesh=mesh, vec=1, dim=2, location_fns=[lambda pt: jnp.isclose(pt[0], 0.0)]
    )
    assert len(prob.boundary_inds_list) == 1
    assert prob.boundary_inds_list[0].shape == (4, 2)
    assert prob.cells_list_face_list[0][0].shape == (4, 4)
    assert prob.physical_surface_quad_points[0].shape == (4, 2, 2)
    assert prob.nanson_scale[0].shape == (4, 1, 2)
    assert prob.selected_face_shape_vals[0].shape == (4, 2, 4)
    assert prob.selected_face_shape_grads[0].shape == (4, 2, 4, 2)


def test_surface_residual_sum():
    mesh = make_mesh()
    prob = SurfaceMassProblem(
        mesh=mesh, vec=1, dim=2, location_fns=[lambda pt: jnp.isclose(pt[0], 0.0)]
    )
    res = prob.compute_residual(ones_sol())
    assert np.isclose(np.asarray(res[0]).sum(), domain_area + 0.02)


def test_custom_init_receives_additional_info():
    prob = CustomInitProblem(mesh=make_mesh(), vec=1, dim=2, additional_info=("hi", 42))
    assert prob.init_args == ("hi", 42)


def test_print_bc_info_no_boundaries(capsys):
    prob = MassProblem(mesh=make_mesh(), vec=1, dim=2)
    prob.print_bc_info()
    out = capsys.readouterr().out
    assert "No surface integral boundary sets found" in out
    assert "No Dirichlet B.C. found" in out


def test_print_bc_info_with_boundaries(capsys):
    prob = SurfaceMassProblem(
        mesh=make_mesh(),
        vec=1,
        dim=2,
        location_fns=[lambda pt: jnp.isclose(pt[0], 0.0)],
        dirichlet_bc_info=[[lambda pt: jnp.isclose(pt[0], 0.0)], [0], [lambda pt: 0.0]],
    )
    assert len(np.asarray(prob.fes[0].node_inds_list[0])) == 5
    prob.print_bc_info()
    out = capsys.readouterr().out
    assert "Surface boundary set 1 information" in out
    assert "Dirichlet Boundary part 1 information" in out


def test_universal_kernel_matches_laplace():
    mesh = make_mesh()
    plain = np.asarray(LaplaceProblem(mesh=mesh, vec=1, dim=2).compute_residual(r_sol(mesh))[0])
    uni = np.asarray(
        UniversalLaplaceProblem(mesh=mesh, vec=1, dim=2).compute_residual(r_sol(mesh))[0]
    )
    assert np.allclose(uni, plain, atol=1e-6)


def test_universal_surface_kernel_matches_surface_mass():
    mesh = make_mesh()
    loc = [lambda pt: jnp.isclose(pt[0], 0.0)]
    expected = np.asarray(
        SurfaceMassProblem(mesh=mesh, vec=1, dim=2, location_fns=loc).compute_residual(ones_sol())[
            0
        ]
    )
    got = np.asarray(
        UniversalSurfaceMassProblem(mesh=mesh, vec=1, dim=2, location_fns=loc).compute_residual(
            ones_sol()
        )[0]
    )
    assert np.allclose(got, expected, atol=1e-6)
    assert np.isclose(got.sum(), domain_area + 0.02)


def test_surface_newton_update():
    mesh = make_mesh()
    prob = SurfaceMassProblem(
        mesh=mesh, vec=1, dim=2, location_fns=[lambda pt: jnp.isclose(pt[0], 0.0)]
    )
    res = prob.newton_update(ones_sol())
    assert prob.jac_values.shape[0] == prob.row_inds.shape[0] == prob.col_inds.shape[0]
    tangent = helpers.dense_tangent(prob, 20)
    assert np.allclose(tangent, tangent.T, atol=1e-6)
    assert np.allclose(tangent @ np.ones(20), np.asarray(res[0][0]).reshape(-1), atol=1e-6)


def test_initialize_geometric_quantities_custom_points():
    mesh = make_mesh()
    prob = MassProblem(mesh=mesh, vec=1, dim=2)
    scaled = [jnp.asarray(np.asarray(mesh.points) * 2.0)]
    prob.initialize_geometric_quantities(scaled)
    assert np.isclose(np.asarray(prob.jxw).sum(), 4 * domain_area)
    pts = np.asarray(prob.physical_quad_points)
    assert pts.min() >= 0.0 and pts.max() <= 0.04


def test_compute_cell_large_mesh():
    mesh = make_mesh(6, 4)  # 24 cells
    prob = MassProblem(mesh=mesh, vec=1, dim=2)
    res = prob.compute_residual(ones_sol(num_nodes=len(mesh.points)))
    assert np.isclose(np.asarray(res[0]).sum(), domain_area)


def test_multiple_boundary_sets():
    mesh = make_mesh()
    prob = TwoBoundarySurfaceProblem(
        mesh=mesh,
        vec=1,
        dim=2,
        location_fns=[lambda pt: jnp.isclose(pt[0], 0.0), lambda pt: jnp.isclose(pt[1], 0.0)],
    )
    assert len(prob.boundary_inds_list) == 2
    assert prob.boundary_inds_list[0].shape == (4, 2)
    assert prob.boundary_inds_list[1].shape == (3, 2)
    res = prob.compute_residual(ones_sol())
    assert np.isclose(np.asarray(res[0]).sum(), domain_area + 0.02 + 0.01)


def test_surface_param_scales_surface_contribution():
    mesh = make_mesh()
    loc = [lambda pt: jnp.isclose(pt[0], 0.0)]
    prob = SurfaceParamMassProblem(mesh=mesh, vec=1, dim=2, location_fns=loc)
    num_faces, num_face_quads = prob.boundary_inds_list[0].shape
    prob.set_params(jnp.ones((num_faces, num_face_quads)))
    assert np.isclose(np.asarray(prob.compute_residual(ones_sol())[0]).sum(), domain_area + 0.02)
    prob.set_params(jnp.full((num_faces, num_face_quads), 2.0))
    assert np.isclose(
        np.asarray(prob.compute_residual(ones_sol())[0]).sum(), domain_area + 2 * 0.02
    )
