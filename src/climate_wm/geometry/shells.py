"""Shell / patch decomposition of the atmosphere (raw field -> shell units).

The atmosphere is treated as S^2 x [pressure]. We split it into

  * shells: groups of adjacent pressure levels (plus one surface shell that
    carries the single-level variables). Each shell is a copy of S^2 that is
    processed by shared GNN weights plus a learned shell embedding, and shells
    exchange information through vertical (inter-shell) message passing.
  * patches: pieces of the sphere given by the Voronoi cells of a coarse
    icosphere level. A (shell, patch) pair is the elementary unit of the
    Lie-structured transition: each one carries its own so(3) generator.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn

from . import spherical_mesh

# Default grouping of the 13 WeatherBench / GraphCast levels (hPa).
# A shell (name, lo, hi) holds the levels lo <= p < hi.
DEFAULT_SHELL_BOUNDS = (
    ("boundary_layer", 850, 1100),
    ("lower_troposphere", 500, 850),
    ("upper_troposphere", 200, 500),
    ("stratosphere", 0, 200),
)


@dataclasses.dataclass(frozen=True)
class Shell:
    name: str
    levels: Tuple[int, ...] = ()
    surface: bool = False


@dataclasses.dataclass(frozen=True)
class ShellSpec:
    shells: Tuple[Shell, ...]

    def __post_init__(self):
        if sum(s.surface for s in self.shells) != 1:
            raise ValueError("Exactly one shell must carry the surface variables.")
        levels = [lvl for s in self.shells for lvl in s.levels]
        if len(levels) != len(set(levels)):
            raise ValueError(f"A pressure level is assigned to two shells: {levels}")

    def __len__(self) -> int:
        return len(self.shells)

    @property
    def surface_index(self) -> int:
        return next(i for i, s in enumerate(self.shells) if s.surface)

    def shell_of_level(self, level: int) -> int:
        for i, s in enumerate(self.shells):
            if level in s.levels:
                return i
        raise KeyError(f"Level {level} hPa is not assigned to any shell.")

    @classmethod
    def from_levels(cls, levels: Sequence[int],
                    bounds=DEFAULT_SHELL_BOUNDS) -> "ShellSpec":
        """Surface shell + pressure shells from (name, lo, hi) bounds."""
        shells = [Shell("surface", (), surface=True)]
        for name, lo, hi in bounds:
            group = tuple(sorted((l for l in levels if lo <= l < hi), reverse=True))
            if group:
                shells.append(Shell(name, group))
        spec = cls(tuple(shells))
        for level in levels:
            spec.shell_of_level(level)
        return spec

    @classmethod
    def from_config(cls, cfg: Optional[Sequence[dict]],
                    levels: Sequence[int]) -> "ShellSpec":
        if not cfg:
            return cls.from_levels(levels)
        return cls(tuple(Shell(c["name"], tuple(c.get("levels", ())),
                               bool(c.get("surface", False))) for c in cfg))


class SpherePatches(nn.Module):
    """Soft partition of mesh nodes into patches + projection onto so(3).

    Patch centres are the vertices of icosphere level `patch_splits` (12, 42,
    162, ... patches). Node i belongs to its `k` nearest patches with weights
    m_iP = softmax(-d_iP^2 / tau^2); `tau` is the patch spacing times
    `softness`. The same weights are used for

      * pooling: the so(3) element of patch P is the least-squares Killing
        field fitted to a node velocity field u_i inside the patch,
            omega_P = argmin_w sum_i m_iP |w x p_i - u_i|^2
                    = (sum_i m_iP (I - p_i p_i^T) + eps I)^-1 sum_i m_iP p_i x u_i,
      * blending: omega_i = sum_P m_iP omega_P (done in the Lie algebra).
    """

    def __init__(self, node_xyz: np.ndarray, patch_splits: int = 1, k: int = 3,
                 softness: float = 0.5, eps: float = 1e-2):
        super().__init__()
        centres = spherical_mesh.icosphere_hierarchy(patch_splits)[-1].vertices
        spacing = spherical_mesh.max_edge_length(
            spherical_mesh.icosphere_hierarchy(patch_splits)[-1])
        k = min(k, len(centres))
        dist, idx = cKDTree(centres).query(node_xyz, k=k)
        dist, idx = dist.reshape(len(node_xyz), k), idx.reshape(len(node_xyz), k)
        tau = softness * spacing
        logits = -(dist / tau) ** 2
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)

        p = node_xyz
        proj = np.eye(3)[None] - p[:, :, None] * p[:, None, :]  # [N, 3, 3]
        a = np.zeros((len(centres), 3, 3))
        mass = np.zeros(len(centres))
        for j in range(k):
            np.add.at(a, idx[:, j], weights[:, j, None, None] * proj)
            np.add.at(mass, idx[:, j], weights[:, j])
        a += eps * np.maximum(mass, 1e-6)[:, None, None] * np.eye(3)[None]
        self.num_patches = len(centres)
        self.register_buffer("centres", torch.tensor(centres, dtype=torch.float32))
        self.register_buffer("node_pos", torch.tensor(node_xyz, dtype=torch.float32))
        self.register_buffer("index", torch.tensor(idx, dtype=torch.long))
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("mass", torch.tensor(mass, dtype=torch.float32))
        self.register_buffer("a_inv", torch.tensor(np.linalg.inv(a), dtype=torch.float32))
        self.register_buffer("hard", torch.tensor(idx[:, 0], dtype=torch.long))

    def pool(self, x: Tensor) -> Tensor:
        """Weighted patch mean of node features. x [B, N, C] -> [B, P, C]."""
        b, n, c = x.shape
        out = x.new_zeros(b, self.num_patches, c)
        for j in range(self.index.shape[1]):
            out.index_add_(1, self.index[:, j], x * self.weight[:, j, None])
        return out / self.mass.clamp_min(1e-6)[None, :, None]

    def killing_projection(self, u: Tensor) -> Tensor:
        """Least-squares so(3) element per patch from tangent velocities.

        u [B, N, 3] -> omega [B, P, 3]; equivariant under global rotations.
        """
        torque = torch.cross(self.node_pos.expand_as(u), u, dim=-1)
        b = u.shape[0]
        rhs = u.new_zeros(b, self.num_patches, 3)
        for j in range(self.index.shape[1]):
            rhs.index_add_(1, self.index[:, j], torque * self.weight[:, j, None])
        return torch.einsum("pij,bpj->bpi", self.a_inv, rhs)

    def blend(self, omega_patch: Tensor) -> Tensor:
        """Patch algebra elements [B, P, 3] -> node elements [B, N, 3]."""
        gathered = omega_patch[:, self.index]  # [B, N, k, 3]
        return (gathered * self.weight[None, :, :, None]).sum(2)
