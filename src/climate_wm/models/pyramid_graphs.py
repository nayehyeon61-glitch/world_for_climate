"""Graphs of the pyramid (graph V-Net) latent: several icosphere levels at once.

Stage 4 keeps the latent state on a *stack* of icosphere levels instead of a
single one. With state levels (1, 2, 3) the stacked node set is

    [ level 1 (42 nodes) | level 2 (162 nodes) | level 3 (642 nodes) ]

and a `Latent` on it has 846 nodes per shell. Everything the transition
networks need (`latent_mesh`, `latent_pos`, `latent_struct`) is defined on
the stacked set, so the Lie generator, the residual and the DIAMOND denoiser
run unchanged on all levels at once; the stacked mesh carries the multimesh
edges of every level plus parent-child edges between consecutive levels.

Per level the module also holds what the V-Net encoder / decoder and the
so(3) pyramid need (GraphCast conventions, see docs/results stage 4 design):

  * `mesh[j]`        multimesh GeoGraph of path level j (edges of levels <= j)
  * `grid2level[j]`  grid -> level j ball query, radius = fraction x level edge
                     (GraphCast's grid2mesh rule applied at every level)
  * `down[j]`        level j+1 -> level j ball query (pooling)
  * `up[j]`          level j -> level j+1 containing-triangle edges (unpooling)
  * `rotation[i]`    semi-Lagrangian SO(3) action on state level i
  * `pyramid[i]`     PatchPyramid over the nodes of state level i

"Path" levels run from the coarsest state level up to the mesh level (the
encoder has to walk down through all of them); "state" levels are the ones
that carry latent state (a subset; P-skip keeps only the coarsest).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn

from ..geometry import spherical_mesh as sm
from ..geometry.group_actions import SphereRotationAction
from ..geometry.shells import PatchPyramid
from .layers import GeoGraph, structural_features


class PyramidGraphs(nn.Module):

    def __init__(self, meshes: Sequence[sm.TriangularMesh], grid_xyz: np.ndarray,
                 grid_lat: np.ndarray, levels: Sequence[int], detail_level: int,
                 radius_fraction: float, break_symmetry: bool, action_rings: int,
                 patch_neighbors: int):
        super().__init__()
        levels = tuple(int(l) for l in levels)
        top = len(meshes) - 1
        if list(levels) != sorted(set(levels)) or levels[-1] > top or levels[0] < 0:
            raise ValueError("pyramid_levels must be increasing icosphere levels <= mesh_splits")
        if list(levels) != list(range(levels[0], levels[-1] + 1)):
            raise ValueError("pyramid_levels must be consecutive icosphere levels")
        self.levels = levels
        self.path_levels = tuple(range(levels[0], top + 1))
        self.detail_level = int(min(detail_level, levels[-1]))
        bs = break_symmetry
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

        # ---- per path level: mesh graph, structure, grid -> level ball query
        self.mesh = nn.ModuleList()
        self.grid2level = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        num_path = len(self.path_levels)
        for j, lv in enumerate(self.path_levels):
            m = meshes[lv]
            s, r = sm.multimesh_edges(meshes[:lv + 1])
            self.mesh.append(GeoGraph(m.vertices, m.vertices, s, r, bs))
            radius = radius_fraction * sm.max_edge_length(m)
            s, r = sm.radius_edges(grid_xyz, m.vertices, radius, min_per_dst=1)
            self.grid2level.append(GeoGraph(grid_xyz, m.vertices, s, r, bs))
            struct = structural_features(m.vertices, bs)
            onehot = np.zeros((len(m.vertices), num_path), np.float32)
            onehot[:, j] = 1.0
            self.register_buffer(f"struct_{j}", f32(np.concatenate([struct, onehot], 1)))
            self.register_buffer(f"pos_{j}", f32(m.vertices))
            if j + 1 < num_path:
                fine = meshes[lv + 1]
                s, r = sm.radius_edges(fine.vertices, m.vertices, radius, 1)
                self.down.append(GeoGraph(fine.vertices, m.vertices, s, r, bs))
                s, r, w = sm.mesh2grid_edges(fine.vertices, m)
                self.up.append(GeoGraph(m.vertices, fine.vertices, s, r, bs, w))

        # ---- per state level: rotation action, patch pyramid, node offsets
        self.rotation = nn.ModuleList()
        self.pyramid = nn.ModuleList()
        counts, offsets, total = [], [], 0
        for lv in self.levels:
            m = meshes[lv]
            self.rotation.append(SphereRotationAction(m, rings=action_rings))
            patch_levels = tuple(range(0, min(lv, self.detail_level) + 1))
            self.pyramid.append(PatchPyramid(m.vertices, patch_levels, k=patch_neighbors))
            counts.append(m.num_vertices)
            offsets.append(total)
            total += m.num_vertices
        self.counts, self.offsets, self.num_nodes = tuple(counts), tuple(offsets), total
        self.max_edge = tuple(float(sm.max_edge_length(meshes[lv])) for lv in self.levels)

        # ---- so(3) pyramid bookkeeping: patch level k is projected from the
        # coarsest state level whose nodes resolve it (>= k).
        self.rot_levels = tuple(range(0, self.detail_level + 1))
        self.rot_source = tuple(self.levels.index(max(k, self.levels[0])) for k in self.rot_levels)
        self.patch_counts = tuple(self.pyramid[-1].patches[k].num_patches for k in self.rot_levels)

        # ---- stacked node set for the transition networks
        pos = np.concatenate([meshes[lv].vertices for lv in self.levels], 0)
        struct = np.concatenate([getattr(self, f"struct_{self.path_levels.index(lv)}").numpy()
                                 for lv in self.levels], 0)
        senders, receivers, kinds = [], [], []
        for i, lv in enumerate(self.levels):
            j = self.path_levels.index(lv)
            s, r = self.mesh[j].edge_index.numpy()
            senders.append(s + offsets[i]); receivers.append(r + offsets[i]); kinds.append(np.zeros(len(s)))
            if i + 1 < len(self.levels):  # parent-child edges, both directions
                s, r = self.down[j].edge_index.numpy()  # fine -> coarse
                senders.append(s + offsets[i + 1]); receivers.append(r + offsets[i]); kinds.append(np.ones(len(s)))
                s, r = self.up[j].edge_index.numpy()  # coarse -> fine
                senders.append(s + offsets[i]); receivers.append(r + offsets[i + 1]); kinds.append(-np.ones(len(s)))
        self.stacked_mesh = GeoGraph(pos, pos, np.concatenate(senders), np.concatenate(receivers),
                                     bs, edge_weight=np.concatenate(kinds))
        self.register_buffer("stacked_pos", f32(pos))
        self.register_buffer("stacked_struct", f32(struct))

        # ---- grid -> stacked nodes (forcings / static fields at every level)
        area = np.cos(np.deg2rad(grid_lat)) + 1e-3
        grid_tree = cKDTree(grid_xyz)
        idx, cells, wts = [], [], []
        for i, lv in enumerate(self.levels):
            nearest = cKDTree(meshes[lv].vertices).query(grid_xyz)[1]
            norm = np.bincount(nearest, weights=area, minlength=counts[i])
            idx.append(nearest + offsets[i])
            cells.append(np.arange(len(grid_xyz)))
            wts.append(area / np.where(norm > 0, norm, 1.0)[nearest])
            empty = np.flatnonzero(norm == 0)  # nodes finer than the grid: nearest cell
            if len(empty):
                idx.append(empty + offsets[i])
                cells.append(grid_tree.query(meshes[lv].vertices[empty])[1])
                wts.append(np.ones(len(empty)))
        self.register_buffer("grid_to_nodes", torch.tensor(np.concatenate(idx), dtype=torch.long))
        self.register_buffer("grid_to_cells", torch.tensor(np.concatenate(cells), dtype=torch.long))
        self.register_buffer("grid_to_nodes_w", f32(np.concatenate(wts)))

    # ------------------------------------------------------------ helpers
    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def struct(self, j: int) -> Tensor:
        return getattr(self, f"struct_{j}")

    def pos(self, j: int) -> Tensor:
        return getattr(self, f"pos_{j}")

    def path_index(self, level: int) -> int:
        return self.path_levels.index(level)

    def slice(self, i: int) -> slice:
        """Node slice of state level i inside the stacked latent."""
        return slice(self.offsets[i], self.offsets[i] + self.counts[i])

    def split(self, x: Tensor, dim: int = 2) -> List[Tensor]:
        """Splits a stacked node axis into the per-level pieces."""
        return list(torch.split(x, list(self.counts), dim=dim))

    def level_weights(self, weights: Sequence[float]) -> Tuple[float, ...]:
        w = tuple(float(x) for x in weights)
        if len(w) < self.num_levels:
            w = w + (w[-1],) * (self.num_levels - len(w))
        return w[:self.num_levels]

    def summary(self) -> str:
        return (f"pyramid levels {self.levels} nodes {self.counts} (stacked {self.num_nodes}, "
                f"{self.stacked_mesh.num_edges} edges), rotation patch levels "
                f"{self.rot_levels} patches {self.patch_counts}")
