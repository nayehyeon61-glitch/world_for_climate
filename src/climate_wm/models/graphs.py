"""All static graphs of the model, built once from the grid and mesh sizes.

    grid (H x W)  --grid2mesh-->  mesh (splits=mesh_splits, multimesh edges)
                  --pool------->  latent mesh (splits=latent_splits)   [Z_t]
                  <--unpool-----
    grid          <--mesh2grid--  mesh

Down-scaling (information compression) happens in two places: grid -> mesh
(GraphCast-style) and mesh -> coarser latent mesh, plus the channel bottleneck
of the latent state. If latent_splits == mesh_splits there is no pooling.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from ..geometry import spherical_mesh as sm
from ..geometry import tangent_frames
from ..geometry.group_actions import SphereRotationAction
from ..geometry.shells import SpherePatches
from .layers import GeoGraph, structural_features
from .pyramid_graphs import PyramidGraphs


@dataclasses.dataclass
class MeshConfig:
    mesh_splits: int = 3
    latent_splits: int = 2
    patch_splits: int = 1
    patch_neighbors: int = 3
    radius_fraction: float = 0.6  # GraphCast radius_query_fraction_edge_length
    break_symmetry: bool = True
    action_rings: int = 3
    # Stage 4 pyramid: icosphere levels that carry latent state (coarse -> fine,
    # consecutive, ending at mesh_splits), e.g. (1, 2, 3); () = single scale.
    pyramid_levels: Tuple[int, ...] = ()
    pyramid_detail_level: int = 2  # finest patch level with its own detail rotation


class WorldGraphs(nn.Module):

    def __init__(self, grid_lat: np.ndarray, grid_lon: np.ndarray,
                 cfg: MeshConfig):
        super().__init__()
        if cfg.latent_splits > cfg.mesh_splits:
            raise ValueError("latent_splits must be <= mesh_splits")
        self.cfg = cfg
        self.grid_shape = (len(grid_lat), len(grid_lon))
        meshes = sm.icosphere_hierarchy(cfg.mesh_splits)
        fine, latent = meshes[-1], meshes[cfg.latent_splits]
        lat_f, lon_f, grid_xyz = sm.grid_nodes(np.asarray(grid_lat, np.float64),
                                               np.asarray(grid_lon, np.float64))
        bs = cfg.break_symmetry
        self.num_grid, self.num_mesh = len(grid_xyz), fine.num_vertices
        self.num_latent = latent.num_vertices
        self.has_pooling = cfg.latent_splits < cfg.mesh_splits

        radius = cfg.radius_fraction * sm.max_edge_length(fine)
        s, r = sm.radius_edges(grid_xyz, fine.vertices, radius, min_per_dst=1)
        self.grid2mesh = GeoGraph(grid_xyz, fine.vertices, s, r, bs)
        s, r = sm.multimesh_edges(meshes)
        self.mesh = GeoGraph(fine.vertices, fine.vertices, s, r, bs)
        if self.has_pooling:
            radius = cfg.radius_fraction * sm.max_edge_length(latent)
            s, r = sm.radius_edges(fine.vertices, latent.vertices, radius, 1)
            self.pool = GeoGraph(fine.vertices, latent.vertices, s, r, bs)
            s, r = sm.multimesh_edges(meshes[:cfg.latent_splits + 1])
            self.latent_mesh = GeoGraph(latent.vertices, latent.vertices, s, r, bs)
            s, r, w = sm.mesh2grid_edges(fine.vertices, latent)
            self.unpool = GeoGraph(latent.vertices, fine.vertices, s, r, bs, w)
        else:
            self.latent_mesh = self.mesh
        s, r, w = sm.mesh2grid_edges(grid_xyz, fine)
        self.mesh2grid = GeoGraph(fine.vertices, grid_xyz, s, r, bs, w)

        self.rotation_action = SphereRotationAction(latent, rings=cfg.action_rings)
        self.patches = SpherePatches(latent.vertices, cfg.patch_splits,
                                     k=cfg.patch_neighbors)

        east, north = tangent_frames.east_north_frames(lat_f, lon_f)
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        self.register_buffer("grid_east", f32(east))
        self.register_buffer("grid_north", f32(north))
        self.register_buffer("grid_pos", f32(grid_xyz))
        self.register_buffer("mesh_pos", f32(fine.vertices))
        self.register_buffer("grid_struct", f32(structural_features(grid_xyz, bs)))
        self.register_buffer("mesh_struct", f32(structural_features(fine.vertices, bs)))

        # Stage 4: the latent lives on a stack of icosphere levels. The
        # transition networks only see the stacked node set.
        self.pyramid = None
        if cfg.pyramid_levels:
            if self.has_pooling:
                raise ValueError("pyramid_levels needs latent_splits == mesh_splits")
            self.pyramid = PyramidGraphs(meshes, grid_xyz, lat_f, cfg.pyramid_levels,
                                         cfg.pyramid_detail_level, cfg.radius_fraction,
                                         bs, cfg.action_rings, cfg.patch_neighbors)
            self.num_latent = self.pyramid.num_nodes
            self.latent_mesh = self.pyramid.stacked_mesh
            self.register_buffer("latent_pos", self.pyramid.stacked_pos.clone())
            self.register_buffer("latent_struct", self.pyramid.stacked_struct.clone())
            self.register_buffer("grid_to_latent", self.pyramid.grid_to_nodes.clone())
            self.register_buffer("grid_to_latent_w", self.pyramid.grid_to_nodes_w.clone())
            self.register_buffer("grid_to_latent_cells", self.pyramid.grid_to_cells.clone())
        else:
            self.register_buffer("latent_pos", f32(latent.vertices))
            self.register_buffer("latent_struct", f32(structural_features(latent.vertices, bs)))
            # Grid -> latent node area-weighted assignment (for forcings at nodes).
            nearest = cKDTree(latent.vertices).query(grid_xyz)[1]
            area = np.cos(np.deg2rad(lat_f)) + 1e-3
            norm = np.bincount(nearest, weights=area, minlength=self.num_latent)
            self.register_buffer("grid_to_latent", torch.tensor(nearest, dtype=torch.long))
            self.register_buffer("grid_to_latent_w", f32(area / norm[nearest]))

    @property
    def struct_dim(self) -> int:
        """Structural feature width of grid and mesh nodes."""
        return self.grid_struct.shape[1]

    @property
    def latent_struct_dim(self) -> int:
        """Structural feature width of latent nodes (+ level one-hot for a pyramid)."""
        return self.latent_struct.shape[1]

    def grid_to_latent_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Area-weighted average of grid fields onto latent nodes.

        x [B, C, H, W] -> [B, N_latent, C]. For a pyramid every level gets its
        own average (the grid is assigned once per level).
        """
        b, c = x.shape[:2]
        flat = x.reshape(b, c, -1).transpose(1, 2)
        if self.pyramid is not None:  # (cell, node) pairs, one set per level
            flat = flat[:, self.grid_to_latent_cells]
        flat = flat * self.grid_to_latent_w[None, :, None]
        out = flat.new_zeros(b, self.num_latent, c)
        return out.index_add_(1, self.grid_to_latent, flat)

    def summary(self) -> str:
        parts = [f"grid {self.grid_shape} = {self.num_grid} nodes",
                 f"mesh {self.num_mesh} nodes / {self.mesh.num_edges} edges",
                 f"latent {self.num_latent} nodes / {self.latent_mesh.num_edges} edges",
                 f"patches {self.patches.num_patches}",
                 f"grid2mesh {self.grid2mesh.num_edges} edges",
                 f"mesh2grid {self.mesh2grid.num_edges} edges"]
        if self.pyramid is not None:
            parts.append(self.pyramid.summary())
        return ", ".join(parts)
