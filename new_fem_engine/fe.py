"""Finite-element space on a mesh: shape functions, quadrature and boundary
conditions.  Derived from jax-fem."""

import time
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as np

from . import logger
from .basis import get_face_shape_vals_and_grads, get_shape_vals_and_grads
from .generate_mesh import Mesh


@dataclass
class FiniteElement:
    """One finite-element variable on a mesh.

    Attributes:
        mesh: Node coordinates and cell connectivity.
        vec: Components per node.
        dim: Spatial dimension.
        ele_type: Element family name, e.g. ``QUAD4``.
        dirichlet_bc_info: ``[location_fns, vecs, value_fns]`` or None.
        quadrature_rule, quadrature_order: Passed to the basis module.
    """

    mesh: Mesh
    vec: int
    dim: int
    ele_type: str
    dirichlet_bc_info: Optional[list] = None
    quadrature_rule: Any = None
    quadrature_order: Optional[int] = None

    def __post_init__(self):
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes * self.vec

        start = time.time()
        logger.info("Computing shape function values, gradients, etc.")

        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(
            self.ele_type,
            quadrature_rule=self.quadrature_rule,
            quadrature_order=self.quadrature_order,
        )
        (
            self.face_shape_vals,
            self.face_shape_grads_ref,
            self.face_quad_weights,
            self.face_normals,
            self.face_inds,
        ) = get_face_shape_vals_and_grads(
            self.ele_type,
            quadrature_rule=self.quadrature_rule,
            quadrature_order=self.quadrature_order,
        )
        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.num_face_quads = self.face_quad_weights.shape[1]

        self.shape_grads, self.jxw = self.get_shape_grads()

        self.node_inds_list, self.vec_inds_list, self.vals_list = (
            self.dirichlet_boundary_conditions(self.dirichlet_bc_info)
        )

        end = time.time()

        logger.info(
            f"Solving a problem with {len(self.cells)} cells, "
            f"{self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs."
        )
        logger.info(
            f"Element type is {self.ele_type}, using {self.num_quads} quad points per element."
        )
        logger.info(f"Pre-computations took {end - start:.3f} [s]")

    def get_shape_grads(self, points=None):
        """Return physical shape gradients and quadrature weights times |J|.

        Returns:
            (shape_grads, jxw) of shapes (num_cells, num_quads, num_nodes, dim)
            and (num_cells, num_quads).
        """
        points = self.points if points is None else points
        assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
        physical_coos = np.take(points, self.cells, axis=0)
        jacobian_dx_deta = np.sum(
            physical_coos[:, None, :, :, None] * self.shape_grads_ref[None, :, :, None, :],
            axis=2,
            keepdims=True,
        )
        jacobian_det = np.linalg.det(jacobian_dx_deta)[:, :, 0]
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
        shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :] @ jacobian_deta_dx)[
            :, :, :, 0, :
        ]
        jxw = jacobian_det * self.quad_weights[None, :]
        return shape_grads_physical, jxw

    def get_face_shape_grads(self, boundary_inds, points=None):
        """Return physical face shape gradients and the Nanson surface measure.

        Returns:
            (face_shape_grads, nanson_scale) of shapes
            (num_selected_faces, num_face_quads, num_nodes, dim) and
            (num_selected_faces, num_face_quads).
        """
        points = self.points if points is None else points
        physical_coos = np.take(points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]]
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]]
        selected_f_normals = self.face_normals[boundary_inds[:, 1]]

        jacobian_dx_deta = np.sum(
            selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :],
            axis=2,
        )
        jacobian_det = np.linalg.det(jacobian_dx_deta)
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)

        face_shape_grads_physical = (
            selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :]
        )[:, :, :, 0, :]

        nanson_scale = np.linalg.norm(
            (selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1
        )
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]]
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self, points=None):
        """Return the quadrature points in physical coordinates, (num_cells, num_quads, dim)."""
        points = self.points if points is None else points
        physical_coos = np.take(points, self.cells, axis=0)
        return np.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)

    def get_physical_surface_quad_points(self, boundary_inds, points=None):
        """Return the face quadrature points of the selected faces in physical coordinates."""
        points = self.points if points is None else points
        physical_coos = np.take(points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]]
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]
        return np.sum(
            selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2
        )

    def dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Resolve ``[location_fns, vecs, value_fns]`` to node indices, components and values.

        Location functions take either ``(point)`` or ``(point, index)``.

        Returns:
            (node_inds_list, vec_inds_list, vals_list), one entry per location.
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        if dirichlet_bc_info is not None:
            location_fns, vecs, value_fns = dirichlet_bc_info
            assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:

                    def location_fn(point, ind, i=i):
                        return location_fns[i](point)

                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(
                        f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}"
                    )

                node_inds = np.argwhere(
                    jax.vmap(location_fn)(self.points, np.arange(self.num_total_nodes))
                ).reshape(-1)
                vec_inds = np.ones_like(node_inds, dtype=np.int32) * vecs[i]
                values = jax.vmap(value_fns[i])(
                    self.points[node_inds].reshape(-1, self.dim)
                ).reshape(-1)
                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list

    def update_dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Replace the stored Dirichlet data with a new ``dirichlet_bc_info``."""
        self.node_inds_list, self.vec_inds_list, self.vals_list = (
            self.dirichlet_boundary_conditions(dirichlet_bc_info)
        )

    def get_boundary_conditions_inds(self, location_fns):
        """Return, per location function, the (cell, local face) pairs lying on it."""
        cell_points = np.take(self.points, self.cells, axis=0)
        cell_face_points = np.take(cell_points, self.face_inds, axis=1)
        cell_face_inds = np.take(self.cells, self.face_inds, axis=1)
        boundary_inds_list = []
        if location_fns is not None:
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:

                    def location_fn(point, ind, i=i):
                        return location_fns[i](point)

                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(
                        f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}"
                    )

                vmap_location_fn = jax.vmap(location_fn)

                def on_boundary(cell_points, cell_inds):
                    boundary_flag = vmap_location_fn(cell_points, cell_inds)
                    return np.all(boundary_flag)

                vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
                boundary_flags = vvmap_on_boundary(cell_face_points, cell_face_inds)
                boundary_inds = np.argwhere(boundary_flags)
                boundary_inds_list.append(boundary_inds)
        return boundary_inds_list

    def convert_from_dof_to_quad(self, sol):
        """Interpolate a nodal solution to the quadrature points, (num_cells, num_quads, vec)."""
        cells_sol = sol[self.cells]
        return np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)

    def convert_from_dof_to_face_quad(self, sol, boundary_inds):
        """Interpolate a nodal solution to the face quadrature points of the selected faces."""
        cells_old_sol = sol[self.cells]
        selected_cell_sols = cells_old_sol[boundary_inds[:, 0]]
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]
        return np.sum(
            selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2
        )

    def sol_to_grad(self, sol):
        """Return the solution gradient at the quadrature points, (num_cells, num_quads, vec, dim)."""
        u_grads = (
            np.take(sol, self.cells, axis=0)[:, None, :, :, None]
            * self.shape_grads[:, :, :, None, :]
        )
        return np.sum(u_grads, axis=2)
