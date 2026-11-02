"""Structured quadrilateral meshes over the (r, z) domain.  Derived from jax-fem."""

import jax
import jax.numpy as np
import meshio
import numpy as onp

from .basis import get_elements, get_face_shape_vals_and_grads


class Mesh:
    """Node coordinates and cell connectivity in basix node order."""

    def __init__(self, points, cells, ele_type=None):
        self.points = np.asarray(points)
        self.cells = np.asarray(cells)
        self.ele_type = ele_type
        assert self._check_cell_orientation(), "Cells must be in CCW order"

    def _check_cell_orientation(self):
        v0 = self.points[self.cells[:, 0]]
        v1 = self.points[self.cells[:, 1]]
        v3 = self.points[self.cells[:, 3]]
        cross = (v1 - v0)[:, 0] * (v3 - v0)[:, 1] - (v1 - v0)[:, 1] * (v3 - v0)[:, 0]
        return bool(onp.all(cross > 0))

    def count_selected_faces(self, location_fn):
        """Return how many cell faces have every vertex satisfying ``location_fn``."""
        _, _, _, _, face_inds = get_face_shape_vals_and_grads(self.ele_type)
        cell_points = self.points[self.cells]
        cell_face_points = cell_points[:, face_inds]

        vmap_location_fn = jax.vmap(location_fn)

        def on_boundary(cell_points):
            boundary_flag = vmap_location_fn(cell_points)
            return np.all(boundary_flag)

        vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
        boundary_flags = vvmap_on_boundary(cell_face_points)
        boundary_inds = onp.argwhere(np.asarray(boundary_flags))
        return boundary_inds.shape[0]


def get_meshio_cell_type(ele_type):
    """Return the meshio cell name for an element type."""
    if ele_type == "QUAD4":
        return "quad"
    raise NotImplementedError(f"Unsupported element type: {ele_type}")


def _structured_quads(r_nodes, z_nodes):
    """Return a meshio quad mesh on the tensor grid of ``r_nodes`` x ``z_nodes``, r-major."""
    n_r, n_z = len(r_nodes) - 1, len(z_nodes) - 1
    rv, zv = onp.meshgrid(r_nodes, z_nodes, indexing="ij")
    points = onp.stack((rv, zv), axis=2).reshape(-1, 2)

    inds = onp.arange(len(points)).reshape(n_r + 1, n_z + 1)
    i1, i2 = inds[:-1, :-1], inds[1:, :-1]
    i3, i4 = inds[1:, 1:], inds[:-1, 1:]
    cells = onp.stack((i1, i2, i3, i4), axis=2).reshape(-1, 4)

    return meshio.Mesh(points=points, cells={"quad": cells})


def rectangle_mesh(n_r, n_z, r_min, r_max, z_min, z_max):
    """Return a uniform n_r x n_z quad mesh of the rectangle."""
    return _structured_quads(
        onp.linspace(r_min, r_max, n_r + 1), onp.linspace(z_min, z_max, n_z + 1)
    )


def conforming_quad_mesh(r_edges, z_edges):
    """Return the quad mesh whose node lines are the given edge coordinates."""
    return _structured_quads(onp.asarray(r_edges), onp.asarray(z_edges))


def create_mesh(meshio_mesh, cell_type, data_type=onp.float64):
    """Convert a meshio mesh to a Mesh in basix node order."""
    if cell_type == "quad":
        _, _, _, _, _, re_order = get_elements("QUAD4")
    else:
        raise NotImplementedError(f"Unsupported cell type: {cell_type}")

    points = meshio_mesh.points.astype(data_type)
    cells = meshio_mesh.cells_dict[cell_type].astype(onp.int32)

    cells = cells[:, re_order]

    return Mesh(points, cells, ele_type="QUAD4")


def get_face_elems(elems, num_face_quads):
    """Return the (num_faces, num_cells, 2) node pairs of every cell edge."""
    num_cells, num_nodes = elems.shape
    face_elems = onp.zeros((num_face_quads, num_cells, 2), dtype=onp.int32)
    for i in range(num_face_quads):
        face_elems[i] = elems[:, [i, (i + 1) % num_nodes]]
    return face_elems


def get_face_nodes(points, faces):
    """Return the coordinates of the nodes of each face."""
    return points[faces]


def get_face_edges(faces):
    """Return the consecutive node pairs of each face as an array."""
    face_edges = []
    for face in faces:
        num_nodes = len(face)
        edges = [[face[i], face[(i + 1) % num_nodes]] for i in range(num_nodes)]
        face_edges.append(edges)
    return onp.array(face_edges)


def get_node_indices(edge_nodes, all_nodes, method="all", interval=None):
    """Return indices of ``all_nodes`` that coincide with any of ``edge_nodes``.

    With ``method="partial"`` only nodes whose last coordinate lies in
    ``interval`` are kept.
    """
    match = (all_nodes[None, :, :] == edge_nodes[:, None, :]).all(axis=-1)
    inds = onp.argwhere(match.any(axis=0))[:, 0]

    if method == "partial":
        if interval is None:
            raise ValueError("interval is required for method='partial'")
        low, high = interval
        mask = (all_nodes[inds, -1] >= low) & (all_nodes[inds, -1] <= high)
        inds = inds[mask]
    return inds
