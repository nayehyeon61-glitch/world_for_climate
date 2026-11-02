"""Tangent bundle of S^2: local frames, wind vectors and parallel transport.

Wind (u, v) is a tangent vector field on the sphere, not a pair of scalars.
We store vector features in ambient R^3 coordinates (tangent to the sphere at
their base point). This removes the gauge (choice of local east/north frame)
from the network: a global rotation Q acts as p -> Qp, V -> QV and every
operation below commutes with it.

  * `east_north_frames`: ERA5's local (east, north) basis, used only at the
    model boundary to convert (u, v) <-> R^3.
  * `transport_matrices`: Levi-Civita parallel transport along the geodesic
    between two points, i.e. the rotation about p_j x p_i taking p_j to p_i.
    This is the connection used for "connection-aware feature transport".
  * `log_map`: Riemannian log map log_{p_i}(p_j) in T_{p_i} S^2.
"""

from __future__ import annotations

import numpy as np
import torch


def east_north_frames(lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Unit east and north vectors [N, 3] at the given points."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                      np.cos(lat)], axis=-1)
    return east, north


def uv_to_xyz(u: torch.Tensor, v: torch.Tensor, east: torch.Tensor,
              north: torch.Tensor) -> torch.Tensor:
    """(u, v) [..., N, C] -> ambient tangent vectors [..., N, C, 3].

    `east`/`north` are [N, 3].
    """
    return u[..., None] * east[:, None, :] + v[..., None] * north[:, None, :]


def xyz_to_uv(vec: torch.Tensor, east: torch.Tensor, north: torch.Tensor):
    """Ambient tangent vectors [..., N, C, 3] -> (u, v), each [..., N, C]."""
    u = (vec * east[:, None, :]).sum(-1)
    v = (vec * north[:, None, :]).sum(-1)
    return u, v


def project_tangent(vec: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Removes the normal component. vec [..., N, C, 3], normal [..., N, 3]."""
    n = normal.unsqueeze(-2)
    return vec - (vec * n).sum(-1, keepdim=True) * n


def rodrigues(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] from unit axes and angles (numpy)."""
    k = np.zeros(axis.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -axis[..., 2], axis[..., 1]
    k[..., 1, 0], k[..., 1, 2] = axis[..., 2], -axis[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -axis[..., 1], axis[..., 0]
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + s * k + (1 - c) * (k @ k)


def transport_matrices(p_src: np.ndarray, p_dst: np.ndarray) -> np.ndarray:
    """Parallel transport T_{src->dst} along the minimal geodesic, [E, 3, 3].

    T maps T_{p_src} S^2 isometrically onto T_{p_dst} S^2 and T p_src = p_dst.
    Coincident points give the identity.
    """
    cross = np.cross(p_src, p_dst)
    sin = np.linalg.norm(cross, axis=-1)
    cos = np.clip((p_src * p_dst).sum(-1), -1., 1.)
    angle = np.arctan2(sin, cos)
    axis = np.where(sin[:, None] > 1e-12, cross / np.maximum(sin, 1e-12)[:, None],
                    np.array([0., 0., 1.]))
    return rodrigues(axis, angle)


def log_map(p_base: np.ndarray, p_target: np.ndarray) -> np.ndarray:
    """log_{p_base}(p_target) in T_{p_base} S^2 (length = geodesic distance)."""
    cos = np.clip((p_base * p_target).sum(-1, keepdims=True), -1., 1.)
    tangent = p_target - cos * p_base
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    angle = np.arccos(cos)
    return np.where(norm > 1e-12, tangent / np.maximum(norm, 1e-12) * angle, 0.)
