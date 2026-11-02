"""Weak-form assembly on a finite-element space, and the axisymmetric
magnetostatic problem built on it.  The generic Problem class is derived from
jax-fem."""

import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp

from . import logger
from .fe import FiniteElement
from .generate_mesh import Mesh
from .geometry import (
    compute_current_density,
    define_core_rectangles,
    intersect_area,
)

# ---- CONSTANTS ----


# fmt: off
@dataclass(frozen=True)
class ProblemSettings:
    """Physical and numerical constants of the problem."""

    mu0:           float = 4 * np.pi * 1e-7  # vacuum permeability [H/m]
    coil_grid_res: int   = 200               # helper grid the tests measure coil areas on
# fmt: on


problem_settings = ProblemSettings()


# ---- GENERIC WEAK-FORM ASSEMBLY ----


@dataclass
class Problem:
    """Assembly of a weak form over one or more finite-element variables.

    Subclasses supply the physics through ``get_tensor_map`` (stiffness),
    ``get_mass_map`` (load) and optional surface or universal kernels, and
    ``custom_init`` for their own setup from ``additional_info``.

    Attributes:
        mesh, vec, dim, ele_type, quadrature_rule, quadrature_order,
        dirichlet_bc_info: Per variable (a single value is wrapped in a list).
        location_fns: Surface-integral boundary selectors.
        additional_info: Positional arguments for ``custom_init``.
    """

    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = "QUAD4"
    quadrature_rule: Any = None
    quadrature_order: int = None
    dirichlet_bc_info: list = None
    location_fns: list = None
    additional_info: tuple = ()

    def __post_init__(self):
        if not isinstance(self.mesh, list):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.quadrature_rule = [self.quadrature_rule]
            self.quadrature_order = [self.quadrature_order]
            self.dirichlet_bc_info = [self.dirichlet_bc_info]

        self.num_vars = len(self.mesh)

        self.fes = [
            FiniteElement(
                mesh=self.mesh[i],
                vec=self.vec[i],
                dim=self.dim,
                ele_type=self.ele_type[i],
                quadrature_rule=self.quadrature_rule[i]
                if isinstance(self.quadrature_rule, list)
                else self.quadrature_rule,
                quadrature_order=self.quadrature_order[i]
                if isinstance(self.quadrature_order, list)
                else self.quadrature_order,
                dirichlet_bc_info=self.dirichlet_bc_info[i]
                if isinstance(self.dirichlet_bc_info, list)
                else self.dirichlet_bc_info,
            )
            for i in range(self.num_vars)
        ]

        self.cells_list = [fe.cells for fe in self.fes]
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0]
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                crt_ind = (
                    self.fes[i].vec * x[i][:, None]
                    + np.arange(self.fes[i].vec)[None, :]
                    + self.offset[i]
                )
                inds.append(crt_ind.reshape(-1))
            return np.hstack(inds)

        # Global (row, col) of every per-cell Jacobian entry, flattened.
        inds = onp.array(jax.vmap(find_ind)(*self.cells_list))
        self.row_inds = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
        self.col_inds = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list]
            inds_face = onp.array(jax.vmap(find_ind)(*cells_list_face))
            row_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
            col_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
            self.row_inds = onp.hstack((self.row_inds, row_face))
            self.col_inds = onp.hstack((self.col_inds, col_face))
            self.cells_list_face_list.append(cells_list_face)

        self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*self.cells_list)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)

        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
        self.num_total_dofs_all_vars = len(dumb_dofs)

        self.num_nodes_cumsum = onp.cumsum([0] + [fe.num_nodes for fe in self.fes])

        self.initialize_geometric_quantities()

        self.internal_vars = ()
        self.internal_vars_surfaces = [() for _ in range(len(self.boundary_inds_list))]
        self.custom_init(*self.additional_info)
        self.pre_jit_fns()

    def initialize_geometric_quantities(self, fes_points=None):
        """Compute shape gradients, quadrature weights and quadrature points.

        Uses the mesh points unless ``fes_points`` (one array per variable)
        is given, which keeps differentiable nodal coordinates as JAX arrays.
        """
        fes_points = [fe.points for fe in self.fes] if fes_points is None else fes_points

        for i, fe in enumerate(self.fes):
            fe.shape_grads, fe.jxw = fe.get_shape_grads(fes_points[i])
            fe.v_grads_jxw = fe.shape_grads[:, :, :, None, :] * fe.jxw[:, :, None, None, None]

        # (num_cells, num_vars, num_quads)
        self.jxw = np.transpose(np.stack([fe.jxw for fe in self.fes]), axes=(1, 0, 2))
        # (num_cells, num_quads, num_nodes + ..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_jxw = np.concatenate([fe.v_grads_jxw for fe in self.fes], axis=2)
        # All variables share the quadrature points of the first.
        self.physical_quad_points = self.fes[0].get_physical_quad_points(fes_points[0])

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for i, fe in enumerate(self.fes):
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(
                    boundary_inds, fes_points[i]
                )
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(
                boundary_inds, fes_points[0]
            )

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            self.physical_surface_quad_points.append(physical_surface_quad_points)

    def custom_init(self):
        """Hook for subclass setup; receives ``additional_info`` unpacked."""
        pass

    def get_laplace_kernel(self, tensor_map):
        """Return the per-cell kernel of the stiffness term for ``tensor_map``.

        Only internal variable 0 feeds ``tensor_map``; later ones belong to
        the mass map.
        """

        def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_jxw, *cell_internal_vars):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, : self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_jxw = cell_v_grads_jxw[:, : self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars[:1]).reshape(
                u_grads.shape
            )
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_jxw, axis=(0, -1))
            return jax.flatten_util.ravel_pytree(val)[0]

        return laplace_kernel

    def get_mass_kernel(self, mass_map):
        """Return the per-cell kernel of the mass (load) term for ``mass_map``."""

        def mass_kernel(cell_sol_flat, x, cell_jxw, *cell_internal_vars):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_jxw = cell_jxw[0]
            u = np.sum(cell_sol[None, :, :] * self.fes[0].shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, x, *cell_internal_vars)
            val = np.sum(
                u_physics[:, None, :]
                * self.fes[0].shape_vals[:, :, None]
                * cell_jxw[:, None, None],
                axis=0,
            )
            return jax.flatten_util.ravel_pytree(val)[0]

        return mass_kernel

    def get_surface_kernel(self, surface_map):
        """Return the per-face kernel of a surface term for ``surface_map``."""

        def surface_kernel(
            cell_sol_flat,
            x,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
            *cell_internal_vars_surface,
        ):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, : self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]

            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)
            val = np.sum(
                u_physics[:, None, :]
                * face_shape_vals[:, :, None]
                * face_nanson_scale[:, None, None],
                axis=0,
            )

            return jax.flatten_util.ravel_pytree(val)[0]

        return surface_kernel

    def pre_jit_fns(self):
        """Build and jit the vmapped cell and face kernels and their Jacobians."""

        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x,))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis,))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(
                cell_sol_flat,
                physical_quad_points,
                cell_shape_grads,
                cell_jxw,
                cell_v_grads_jxw,
                *cell_internal_vars,
            ):
                if hasattr(self, "get_mass_map"):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(
                        cell_sol_flat, physical_quad_points, cell_jxw, *cell_internal_vars
                    )
                else:
                    mass_val = 0.0

                if hasattr(self, "get_tensor_map"):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(
                        cell_sol_flat, cell_shape_grads, cell_v_grads_jxw, *cell_internal_vars
                    )
                else:
                    laplace_val = 0.0

                if hasattr(self, "get_universal_kernel"):
                    universal_kernel = self.get_universal_kernel()
                    universal_val = universal_kernel(
                        cell_sol_flat,
                        physical_quad_points,
                        cell_shape_grads,
                        cell_jxw,
                        cell_v_grads_jxw,
                        *cell_internal_vars,
                    )
                else:
                    universal_val = 0.0

                return laplace_val + mass_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                def kernel_partial(cell_sol_flat):
                    return kernel(cell_sol_flat, *args)

                return value_and_jacfwd(kernel_partial, cell_sol_flat)

            return kernel, kernel_jac

        def get_kernel_fn_face(ind):
            def kernel(
                cell_sol_flat,
                physical_surface_quad_points,
                face_shape_vals,
                face_shape_grads,
                face_nanson_scale,
                *cell_internal_vars_surface,
            ):
                if hasattr(self, "get_surface_maps"):
                    surface_kernel = self.get_surface_kernel(self.get_surface_maps()[ind])
                    surface_val = surface_kernel(
                        cell_sol_flat,
                        physical_surface_quad_points,
                        face_shape_vals,
                        face_shape_grads,
                        face_nanson_scale,
                        *cell_internal_vars_surface,
                    )
                else:
                    surface_val = 0.0

                if hasattr(self, "get_universal_kernels_surface"):
                    universal_kernel = self.get_universal_kernels_surface()[ind]
                    universal_val = universal_kernel(
                        cell_sol_flat,
                        physical_surface_quad_points,
                        face_shape_vals,
                        face_shape_grads,
                        face_nanson_scale,
                        *cell_internal_vars_surface,
                    )
                else:
                    universal_val = 0.0

                return surface_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                def kernel_partial(cell_sol_flat):
                    return kernel(cell_sol_flat, *args)

                return value_and_jacfwd(kernel_partial, cell_sol_flat)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        self.kernel = jax.jit(jax.vmap(kernel))
        self.kernel_jac = jax.jit(jax.vmap(kernel_jac))

        num_surfaces = len(self.boundary_inds_list)
        if hasattr(self, "get_surface_maps"):
            assert num_surfaces == len(self.get_surface_maps())
        elif hasattr(self, "get_universal_kernels_surface"):
            assert num_surfaces == len(self.get_universal_kernels_surface())
        else:
            assert num_surfaces == 0, "Missing definitions for surface integral"

        self.kernel_face = []
        self.kernel_jac_face = []
        for i in range(len(self.boundary_inds_list)):
            kernel_face, kernel_jac_face = get_kernel_fn_face(i)
            self.kernel_face.append(jax.jit(jax.vmap(kernel_face)))
            self.kernel_jac_face.append(jax.jit(jax.vmap(kernel_jac_face)))

    def compute_cell(self, cells_sol_flat, jac_flag, internal_vars):
        """Evaluate the volume weak form (and its Jacobian if ``jac_flag``) on every cell."""
        vmap_fn = self.kernel_jac if jac_flag else self.kernel
        return vmap_fn(
            cells_sol_flat,
            self.physical_quad_points,
            self.shape_grads,
            self.jxw,
            self.v_grads_jxw,
            *internal_vars,
        )

    def compute_face(self, cells_sol_flat, np_version, jac_flag, internal_vars_surfaces):
        """Evaluate the surface weak forms (and Jacobians if ``jac_flag``) on every boundary set."""
        values = []
        jacs = []
        for i, boundary_inds in enumerate(self.boundary_inds_list):
            vmap_fn = self.kernel_jac_face[i] if jac_flag else self.kernel_face[i]
            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]
            input_collection = [
                selected_cell_sols_flat,
                self.physical_surface_quad_points[i],
                self.selected_face_shape_vals[i],
                self.selected_face_shape_grads[i],
                self.nanson_scale[i],
                *internal_vars_surfaces[i],
            ]
            if jac_flag:
                val, jac = vmap_fn(*input_collection)
                values.append(val)
                jacs.append(jac)
            else:
                values.append(vmap_fn(*input_collection))
        return (values, jacs) if jac_flag else values

    def compute_residual_vars_helper(self, weak_form_flat, weak_form_face_flat):
        """Scatter per-cell and per-face weak forms into global residual arrays."""
        res_list = [
            np.zeros((fe.num_total_nodes, fe.vec), dtype=weak_form_flat.dtype) for fe in self.fes
        ]
        weak_form_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_flat)
        res_list = [
            res_list[i]
            .at[self.cells_list[i].reshape(-1)]
            .add(weak_form_list[i].reshape(-1, self.fes[i].vec))
            for i in range(self.num_vars)
        ]

        for ind, cells_list_face in enumerate(self.cells_list_face_list):
            weak_form_face_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(
                weak_form_face_flat[ind]
            )
            res_list = [
                res_list[i]
                .at[cells_list_face[i].reshape(-1)]
                .add(weak_form_face_list[i].reshape(-1, self.fes[i].vec))
                for i in range(self.num_vars)
            ]

        return res_list

    def compute_residual_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        """Return the global residual list for ``sol_list``."""
        logger.debug("Computing cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
        weak_form_flat = self.compute_cell(cells_sol_flat, False, internal_vars)
        weak_form_face_flat = self.compute_face(cells_sol_flat, np, False, internal_vars_surfaces)
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_newton_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        """Return (residual list, flat Jacobian entries) for ``sol_list``.

        The entries line up with ``row_inds`` and ``col_inds``; they are also
        stored on ``self.jac_values``.
        """
        logger.debug("Computing cell Jacobian and cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
        weak_form_flat, cells_jac_flat = self.compute_cell(cells_sol_flat, True, internal_vars)
        jac_values = cells_jac_flat.reshape(-1)

        weak_form_face_flat, cells_jac_face_flat = self.compute_face(
            cells_sol_flat, np, True, internal_vars_surfaces
        )
        for cells_jac_f_flat in cells_jac_face_flat:
            jac_values = np.hstack((jac_values, cells_jac_f_flat.reshape(-1)))

        self.jac_values = np.asarray(jac_values)

        return (
            self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat),
            self.jac_values,
        )

    def compute_residual(self, sol_list):
        """Return the residual list using the stored internal variables."""
        return self.compute_residual_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def newton_update(self, sol_list):
        """Return (residual list, flat Jacobian entries) using the stored internal variables."""
        return self.compute_newton_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def set_params(self, params):
        raise NotImplementedError("Child class must implement this function!")

    def print_bc_info(self):
        """Print the surface-integral boundary sets and the Dirichlet data."""
        boundary_inds_list = self.boundary_inds_list
        if len(boundary_inds_list) != 0:
            print("\n\n### Surface integral boundary sets are specified")
            for i in range(len(boundary_inds_list)):
                print(f"\nSurface boundary set {i + 1} information:")
                print(boundary_inds_list[i])
                print(f"Array.shape = (num_selected_faces, 2) = {boundary_inds_list[i].shape}")
                print("Interpretation:")
                print("    Array[i, 0] returns the global cell index of the ith selected face")
                print("    Array[i, 1] returns the local face index of the ith selected face")
        else:
            print("\n\n### No surface integral boundary sets found.")

        fe = self.fes[0]
        if len(fe.node_inds_list) != 0:
            print("\n\n### Dirichlet B.C. is specified")
            for i in range(len(fe.node_inds_list)):
                print(f"\nDirichlet Boundary part {i + 1} information:")
                if len(fe.node_inds_list[i]) == 0:
                    bc_array = onp.zeros((0, 3))
                else:
                    bc_array = onp.stack(
                        [
                            fe.node_inds_list[i],
                            fe.vec_inds_list[i],
                            fe.vals_list[i],
                        ]
                    ).T
                print(bc_array)
                print(f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}")
                print("Interpretation:")
                print("    Array[i, 0] returns the node index of the ith selected dof")
                print("    Array[i, 1] returns the vec index of the ith selected dof")
                print("    Array[i, 2] returns the value assigned to ith selected dof")
        else:
            print("\n\n### No Dirichlet B.C. found.")


# ---- AXISYMMETRIC MAGNETOSTATICS ----


class MagnetostaticProblem(Problem):
    """Axisymmetric magnetostatics in the flux function psi = r A_phi.

    Weak form: int nu (2 pi / r) grad psi . grad v dr dz = int 2 pi J v dr dz.
    The material enters through ``set_params`` as a per-quadrature fill
    field; the coils are rectangles carrying one turn of one ampere in total.
    """

    linear_problem = True

    def custom_init(self, core_spec, mesh_size):
        self.mu0 = problem_settings.mu0
        self.mur = core_spec["params"]["mur"]
        self.mesh_size = mesh_size

        params = dict(core_spec["params"])
        params.setdefault("gap_number", 0)
        params.setdefault("gap_size", 0.0)

        (self.core_rects_padded, self.rect_mask, self.primary_rect, self.secondary_rect) = (
            define_core_rectangles(params)
        )

        # One measure for the coil fill and its area, so the source is exactly
        # turns * current on any mesh.
        cell_pts = self.fes[0].points[self.fes[0].cells]
        self.cell_box = (
            cell_pts[..., 0].min(axis=1),
            cell_pts[..., 0].max(axis=1),
            cell_pts[..., 1].min(axis=1),
            cell_pts[..., 1].max(axis=1),
        )
        self.cell_area = (self.cell_box[1] - self.cell_box[0]) * (
            self.cell_box[3] - self.cell_box[2]
        )
        self.cell_jxw = np.sum(self.fes[0].jxw, axis=1)

        # The two coil rectangles are half-turns of one winding.
        self.turns = 0.5
        self.current = 1.0

        self.area_prim = np.sum(self._coil_fill(self.primary_rect) * self.cell_jxw)
        self.area_sec = np.sum(self._coil_fill(self.secondary_rect) * self.cell_jxw)

    def _coil_fill(self, rect):
        """Return the exact area fraction of ``rect`` inside each cell, (num_cells,)."""
        overlap = intersect_area(
            self.cell_box[0], self.cell_box[1], self.cell_box[2], self.cell_box[3], *rect
        )
        return np.clip(overlap / self.cell_area, 0.0, 1.0)

    def coil_current_density(self, prim=None, sec=None):
        """Return J_phi at every quadrature point, (num_cells, num_quads).

        ``prim``/``sec`` let a jitted caller pass traced coil rectangles;
        omitted, the stored coil geometry is used.
        """
        if prim is None:
            prim, sec = self.primary_rect, self.secondary_rect
        f_prim, f_sec = self._coil_fill(prim), self._coil_fill(sec)
        j = compute_current_density(
            f_prim,
            f_sec,
            np.sum(f_prim * self.cell_jxw),
            np.sum(f_sec * self.cell_jxw),
            self.current,
            self.turns,
        )
        return np.broadcast_to(j[:, None], (self.num_cells, self.fes[0].num_quads))

    def get_tensor_map(self):
        def tensor_fn(u_grad, nu_2pi_over_r):
            return nu_2pi_over_r * u_grad

        return tensor_fn

    def get_mass_map(self):
        # residual = laplace + mass = 0, so the load enters with a minus sign.
        def mass_fn(u, x, nu_2pi_over_r, j):
            return -2.0 * np.pi * j[..., None]

        return mass_fn

    def set_params(self, params):
        """Set the material fill and, optionally, traced coil rectangles.

        ``params`` is the fill field (num_cells, num_quads, 3) -- or a bare
        area fraction -- or a tuple ``(fill, primary_rect, secondary_rect,
        ...)``.  Internal variable 0 feeds the stiffness, 1 the load.
        """
        if isinstance(params, tuple):
            fill, prim, sec = params[0], params[1], params[2]
        else:
            fill, prim, sec = params, self.primary_rect, self.secondary_rect
        nu = self._reluctivity(fill)
        r = self.physical_quad_points[..., 0]
        r_safe = np.maximum(r, 1e-6)
        self.internal_vars = [
            nu * (2 * np.pi / r_safe)[..., None],
            self.coil_current_density(prim, sec),
        ]

    def _reluctivity(self, fill):
        """Return the directional reluctivity per quadrature point, (nc, nq, dim).

        A fill of (area, mu_r, mu_z) carries the laminate permeabilities of
        geometry.compute_fill_laminate; a bare area fraction gives the
        isotropic harmonic mean.
        """
        fill = np.asarray(fill)

        def harmonic(x):
            return 1.0 / (x / self.mur + (1.0 - x))

        if fill.ndim >= 3 and fill.shape[-1] == 3:
            mu_r, mu_z = fill[..., 1], fill[..., 2]
        else:
            mu_r = mu_z = harmonic(fill)

        # d(psi)/dr carries B_z and d(psi)/dz carries B_r, so each gradient
        # component takes the other direction's permeability.
        return np.stack(
            [1.0 / (problem_settings.mu0 * mu_z), 1.0 / (problem_settings.mu0 * mu_r)], axis=-1
        )

    def _fast_assembly_ok(self):
        return (
            self.num_vars == 1
            and len(self.fes) == 1
            and self.fes[0].vec == 1
            and len(self.boundary_inds_list) == 0
            and len(self.internal_vars) >= 2
        )

    def compute_newton_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        """Assemble the linear problem directly with einsum when the layout allows."""
        if not self._fast_assembly_ok():
            return super().compute_newton_vars(sol_list, internal_vars, internal_vars_surfaces)

        nu2 = internal_vars[0]
        sg = self.shape_grads[:, :, : self.fes[0].num_nodes, :]
        jxw = self.jxw[:, 0, :]
        sv = self.fes[0].shape_vals

        # nu is diagonal in the gradient index, so it contracts with d.
        k_cell = np.einsum("cqd,cqid,cqjd,cq->cij", nu2, sg, sg, jxw)
        jac_values = k_cell.reshape(-1)

        j = internal_vars[1]
        mass = np.einsum("cq,qn,cq->cn", -2.0 * np.pi * j, sv, jxw)

        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)]
        u_cell = cells_sol_list[0][..., 0]
        weak_form_flat = np.einsum("cij,cj->ci", k_cell, u_cell) + mass

        res_list = self.compute_residual_vars_helper(weak_form_flat, [])
        self.jac_values = np.asarray(jac_values)
        return res_list, jac_values

    def compute_inductance(self, sol_list, prim=None, sec=None):
        """Return L = 2 pi / I^2 int J psi dA from the stored energy.

        ``prim``/``sec`` let a jitted caller pass its own traced coil
        rectangles rather than reading them from ``self``.
        """
        psi_quads = self.fes[0].convert_from_dof_to_quad(sol_list[0])
        jxw = self.fes[0].jxw[:, :, None]
        j_quads = self.coil_current_density(prim, sec)[:, :, None]
        energy = 2 * np.pi * np.sum(psi_quads * j_quads * jxw)
        return energy / self.current**2
