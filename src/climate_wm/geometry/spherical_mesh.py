"""Icosahedral multi-mesh on S^2 and grid <-> mesh connectivity.

The icosahedron construction and the 4-way face split follow
`weathernext/utils/icosahedral_mesh.py` (GraphCast, Apache 2.0) so that our
mesh hierarchy has the same node ordering: the vertices of a coarse level are a
prefix of the vertices of every finer level. We rely on that nesting for the
fine -> coarse latent pooling and for the patch decomposition.

Connectivity mirrors GraphCast:
  * grid2mesh: every grid node sends to mesh nodes within a radius
    (`radius_fraction` x the longest edge of the finest mesh),
  * mesh2grid: every grid node receives from the 3 vertices of the mesh
    triangle that contains it.
"""

from __future__ import annotations

import dataclasses
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class TriangularMesh:
    vertices: np.ndarray  # [num_vertices, 3], unit norm, float64
    faces: np.ndarray  # [num_faces, 3], counter-clockwise seen from outside

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]


def icosahedron(pole_parallel_faces: bool = True) -> TriangularMesh:
    """Regular icosahedron with circumscribed unit sphere (GraphCast layout)."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1., -1.]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.))
            vertices.append((0., c1, c2))
            vertices.append((c2, 0., c1))
    vertices = np.array(vertices, dtype=np.float64)
    vertices /= np.linalg.norm([1., phi])
    faces = [
        (0, 1, 2), (0, 6, 1), (8, 0, 2), (8, 4, 0), (3, 8, 2), (3, 2, 7),
        (7, 2, 1), (0, 4, 6), (4, 11, 6), (6, 11, 5), (1, 5, 7), (4, 10, 11),
        (4, 8, 10), (10, 8, 3), (10, 3, 9), (11, 10, 9), (11, 9, 5), (5, 9, 7),
        (9, 3, 7), (1, 6, 5),
    ]
    if pole_parallel_faces:
        # Rotate so that the top/bottom faces are parallel to the x-y plane,
        # which avoids having a mesh node exactly at the poles.
        angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
        rotation_angle = (np.pi - angle_between_faces) / 2
        rotation = Rotation.from_euler(seq="y", angles=rotation_angle)
        vertices = vertices @ rotation.as_matrix()
    return TriangularMesh(vertices=vertices, faces=np.array(faces, np.int64))


def subdivide(mesh: TriangularMesh) -> TriangularMesh:
    """Splits every face in 4; new vertices are appended (nesting is kept)."""
    vertices = list(mesh.vertices)
    child_index = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in child_index:
            p = mesh.vertices[i] + mesh.vertices[j]
            vertices.append(p / np.linalg.norm(p))
            child_index[key] = len(vertices) - 1
        return child_index[key]

    faces = []
    for i1, i2, i3 in mesh.faces:
        i12, i23, i31 = midpoint(i1, i2), midpoint(i2, i3), midpoint(i3, i1)
        faces.extend([[i1, i12, i31], [i12, i2, i23], [i31, i23, i3],
                      [i12, i23, i31]])
    return TriangularMesh(np.stack(vertices), np.array(faces, np.int64))


def icosphere_hierarchy(splits: int) -> List[TriangularMesh]:
    """Meshes from the icosahedron (level 0) to `splits` refinements."""
    meshes = [icosahedron()]
    for _ in range(splits):
        meshes.append(subdivide(meshes[-1]))
    return meshes


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Directed edges (both directions) of a triangular mesh."""
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def multimesh_edges(
        meshes: Sequence[TriangularMesh]) -> Tuple[np.ndarray, np.ndarray]:
    """Union of the edges of all levels, indexed into the finest vertex set."""
    pairs = np.concatenate(
        [np.stack(faces_to_edges(m.faces), axis=1) for m in meshes], axis=0)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]


def max_edge_length(mesh: TriangularMesh) -> float:
    s, r = faces_to_edges(mesh.faces)
    lengths = np.linalg.norm(mesh.vertices[s] - mesh.vertices[r], axis=-1)
    return float(lengths.max())


def lat_lon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=-1)


def xyz_to_lat_lon(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1., 1.)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0])) % 360.
    return lat, lon


def grid_nodes(lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Flattened lat-major grid (index = i_lat * num_lon + i_lon), as GraphCast.

    Returns (lat [H*W], lon [H*W], xyz [H*W, 3]).
    """
    lon2d, lat2d = np.meshgrid(lon_deg, lat_deg)
    lat_flat, lon_flat = lat2d.reshape(-1), lon2d.reshape(-1)
    return lat_flat, lon_flat, lat_lon_to_xyz(lat_flat, lon_flat)


def radius_edges(src_xyz: np.ndarray, dst_xyz: np.ndarray, radius: float,
                 min_per_dst: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Edges src -> dst for all pairs closer than `radius` (chordal distance).

    Every destination additionally receives from its `min_per_dst` nearest
    sources so that no node is left without incoming messages when the source
    grid is coarse (smoke-test resolutions).
    """
    tree = cKDTree(src_xyz)
    neighbors = tree.query_ball_point(dst_xyz, r=radius)
    src = [np.asarray(n, np.int64) for n in neighbors]
    dst = [np.full(len(n), i, np.int64) for i, n in enumerate(neighbors)]
    if min_per_dst > 0:
        _, nearest = tree.query(dst_xyz, k=min_per_dst)
        nearest = nearest.reshape(len(dst_xyz), -1)
        src.append(nearest.reshape(-1))
        dst.append(np.repeat(np.arange(len(dst_xyz)), nearest.shape[1]))
    pairs = np.stack([np.concatenate(src), np.concatenate(dst)], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]


def face_inverse_matrices(mesh: TriangularMesh) -> np.ndarray:
    """[F, 3, 3] inverses of [a b c] (columns = face vertices).

    For a point q, lam = Minv @ q normalised to sum 1 are the gnomonic
    barycentric coordinates of q in the spherical triangle; q lies inside the
    triangle iff all lam >= 0 (and the sum is positive).
    """
    m = np.transpose(mesh.vertices[mesh.faces], (0, 2, 1))  # columns a, b, c
    return np.linalg.inv(m)


def barycentric(points: np.ndarray, minv: np.ndarray) -> np.ndarray:
    lam = np.einsum("...ij,...j->...i", minv, points)
    total = lam.sum(-1, keepdims=True)
    safe = np.where(total == 0, 1., total)
    return np.where(total > 0, lam / safe, -np.inf)


def containing_triangles(points_xyz: np.ndarray, mesh: TriangularMesh,
                         num_candidates: int = 8):
    """Containing face index [P] and barycentric weights [P, 3] for points."""
    centroids = mesh.vertices[mesh.faces].mean(1)
    centroids /= np.linalg.norm(centroids, axis=-1, keepdims=True)
    k = min(num_candidates, len(mesh.faces))
    _, cand = cKDTree(centroids).query(points_xyz, k=k)
    cand = cand.reshape(len(points_xyz), k)
    minv = face_inverse_matrices(mesh)
    lam = barycentric(points_xyz[:, None, :], minv[cand])  # [P, k, 3]
    rows = np.arange(len(points_xyz))
    best = np.argmax(lam.min(-1), axis=1)
    face = cand[rows, best]
    weights = np.clip(lam[rows, best], 0., None)
    weights /= weights.sum(-1, keepdims=True)
    return face, weights


def mesh2grid_edges(grid_xyz: np.ndarray, mesh: TriangularMesh):
    """Edges mesh -> grid from the 3 vertices of each grid node's triangle.

    Returns (senders = mesh indices, receivers = grid indices, barycentric
    weight of every edge).
    """
    face, weights = containing_triangles(grid_xyz, mesh)
    senders = mesh.faces[face].reshape(-1)
    receivers = np.repeat(np.arange(len(grid_xyz)), 3)
    return senders, receivers, weights.reshape(-1)


def vertex_neighborhood_faces(mesh: TriangularMesh,
                              rings: int = 2) -> np.ndarray:
    """For every vertex, the faces touching its (rings - 1)-ring neighbourhood.

    Returned as a dense [N, K] array padded by repeating the first face. Used
    by the semi-Lagrangian group action to look up the triangle that contains
    a departure point near each node.
    """
    n = mesh.num_vertices
    vertex_faces = [[] for _ in range(n)]
    for f, tri in enumerate(mesh.faces):
        for v in tri:
            vertex_faces[v].append(f)
    neighbors = [{i} for i in range(n)]
    for a, b in zip(*faces_to_edges(mesh.faces)):
        neighbors[b].add(a)
    region = [{i} for i in range(n)]
    for _ in range(rings - 1):
        region = [set().union(*(neighbors[j] for j in reg)) for reg in region]
    cand = [sorted(set().union(*(vertex_faces[j] for j in reg)))
            for reg in region]
    k = max(len(c) for c in cand)
    out = np.zeros((n, k), np.int64)
    for i, c in enumerate(cand):
        out[i, :len(c)] = c
        out[i, len(c):] = c[0]
    return out
