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
        self.register_buffer("reg", torch.tensor(eps * np.maximum(mass, 1e-6), dtype=torch.float32),
                             persistent=False)  # derived; keeps old checkpoints loadable
        self.register_buffer("hard", torch.tensor(idx[:, 0], dtype=torch.long))

    def pool(self, x: Tensor) -> Tensor:
        """Weighted patch mean of node features. x [B, N, C] -> [B, P, C]."""
        b, n, c = x.shape
        out = x.new_zeros(b, self.num_patches, c)
        for j in range(self.index.shape[1]):
            out.index_add_(1, self.index[:, j], x * self.weight[:, j, None])
        return out / self.mass.clamp_min(1e-6)[None, :, None]

    def killing_projection(self, u: Tensor, prior: Optional[Tensor] = None) -> Tensor:
        """Least-squares so(3) element per patch from tangent velocities.

        u [B, N, 3] -> omega [B, P, 3]; equivariant under global rotations.
        The Tikhonov term `eps * mass |omega - prior|^2` pulls the component a
        small patch cannot observe (rotation about its own axis) towards
        `prior` [B, P, 3] (zero when omitted); a pyramid passes the parent's
        element so that a rigid field projects exactly at every level.
        """
        torque = torch.cross(self.node_pos.expand_as(u), u, dim=-1)
        b = u.shape[0]
        rhs = u.new_zeros(b, self.num_patches, 3)
        for j in range(self.index.shape[1]):
            rhs.index_add_(1, self.index[:, j], torque * self.weight[:, j, None])
        if prior is not None:
            rhs = rhs + self.reg[None, :, None] * prior
        return torch.einsum("pij,bpj->bpi", self.a_inv, rhs)

    def blend(self, omega_patch: Tensor) -> Tensor:
        """Patch algebra elements [B, P, 3] -> node elements [B, N, 3]."""
        gathered = omega_patch[:, self.index]  # [B, N, k, 3]
        return (gathered * self.weight[None, :, :, None]).sum(2)


def _compose(a: Tensor, b: Tensor) -> Tensor:
    """log(exp(a) exp(b)) on so(3) elements [..., 3]."""
    from .group_actions import so3_exp, so3_log
    return so3_log(so3_exp(a) @ so3_exp(b))


class PatchPyramid(nn.Module):
    """Patch levels over one node set and the so(3) pyramid on them (stage 4).

    Level l uses the icosphere-level-`levels[l]` centres (12, 42, 162, ...).
    The Killing projection at a coarse level is the least-squares so(3) fit
    over the coarse patch, i.e. the inertia-weighted pooling of the finer
    elements, so `project` gives the whole pyramid omega^(0..L) from one node
    velocity field. Details are relative rotations to the parent patch,

        delta_P = log(exp(-omega_parent(P)) exp(omega_P)),

    and the scale chain g = exp(omega^(0)) exp(delta^(1)) ... exp(delta^(L))
    is blended onto the nodes level by level (`node_algebra`). This is the
    shell chain's forward kinematics applied along the scale axis.
    """

    def __init__(self, node_xyz: np.ndarray, levels: Sequence[int] = (0, 1, 2),
                 k: int = 3, softness: float = 0.5):
        super().__init__()
        if list(levels) != sorted(set(levels)):
            raise ValueError("pyramid levels must be strictly increasing")
        self.levels = tuple(int(l) for l in levels)
        self.patches = nn.ModuleList([SpherePatches(node_xyz, l, k, softness) for l in self.levels])
        for i in range(1, len(self.levels)):
            coarse = self.patches[i - 1].centres.numpy()
            fine = self.patches[i].centres.numpy()
            parent = cKDTree(coarse).query(fine)[1]
            self.register_buffer(f"parent_{i}", torch.tensor(parent, dtype=torch.long))

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def parent(self, level: int) -> Tensor:
        """Index of the parent patch (level - 1) of every patch at `level` >= 1."""
        return getattr(self, f"parent_{level}")

    def root(self, u: Tensor) -> Tensor:
        """Whole-sphere least-squares rotation [B, 3]: the pyramid's level -1.

        For a near-uniform node set sum_i (I - p_i p_i^T) ~ (2N/3) I, but the
        exact normal matrix is used.
        """
        p = self.patches[0].node_pos
        a = torch.eye(3, device=p.device, dtype=p.dtype) * len(p) - p.T @ p
        rhs = torch.cross(p.expand_as(u), u, dim=-1).sum(1)
        return torch.linalg.solve(a, rhs.T).T

    def project(self, u: Tensor) -> list:
        """Node velocities [B, N, 3] -> [omega^(l) [B, P_l, 3] for every level].

        Hierarchical: level l is regularised towards its parent's element, so
        the axial component a small patch cannot see is inherited from the
        coarser scale instead of being shrunk to zero.
        """
        root = self.root(u)
        omegas = [self.patches[0].killing_projection(
            u, root[:, None, :].expand(-1, self.patches[0].num_patches, -1))]
        for i in range(1, self.num_levels):
            prior = omegas[i - 1][:, self.parent(i)]
            omegas.append(self.patches[i].killing_projection(u, prior))
        return omegas

    def details(self, omegas: Sequence[Tensor]) -> list:
        """Absolute elements per level -> [omega^(0), delta^(1), ..., delta^(L)]."""
        parts = [omegas[0]]
        for i in range(1, self.num_levels):
            parent = omegas[i - 1][:, self.parent(i)]
            parts.append(_compose(-parent, omegas[i]))
        return parts

    def reconstruct(self, parts: Sequence[Tensor]) -> list:
        """[omega^(0), delta^(1), ...] -> absolute elements per level."""
        omegas = [parts[0]]
        for i in range(1, self.num_levels):
            omegas.append(_compose(omegas[i - 1][:, self.parent(i)], parts[i]))
        return omegas

    def node_algebra(self, parts: Sequence[Tensor]) -> Tensor:
        """Scale chain blended onto the nodes: [B, N, 3] algebra element per node."""
        out = self.patches[0].blend(parts[0])
        for i in range(1, self.num_levels):
            out = _compose(out, self.patches[i].blend(parts[i]))
        return out

    @staticmethod
    def energy(parts: Sequence[Tensor]) -> Tensor:
        """Mean squared rotation per level [L] (a rotational energy spectrum)."""
        return torch.stack([p.square().sum(-1).mean() for p in parts])
