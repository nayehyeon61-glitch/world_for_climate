"""Deterministic residual transition (Stage 2, ablations B2-B5).

    r = g_theta(Zbar_{t+1}, Z_{t-k+1:t}, c_{t+1}) in T_{Zbar} M_Z
    Z_{t+1} = Exp_{Zbar}(r)

With the Lie generator disabled, Zbar_{t+1} = Z_t and this is the plain
Euclidean latent dynamics baseline. The output layer is zero-initialised so
training starts from the structural (or persistence) prediction.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

from .graphs import WorldGraphs
from .latent import Latent, LatentEmbedding
from .layers import ShellProcessor, VectorLinear


class ResidualDynamics(nn.Module):

    def __init__(self, graphs: WorldGraphs, num_shells: int, history: int,
                 forcing_dim: int, cs: int, cv_latent: int, d: int, cv: int,
                 hidden_layers: int, layers: int):
        super().__init__()
        self.embed = LatentEmbedding(history + 1, cs, cv_latent, forcing_dim,
                                     graphs.struct_dim, num_shells, d, cv,
                                     hidden_layers)
        self.processor = ShellProcessor(d, cv, graphs.latent_mesh.edge_dim,
                                        layers, num_shells, hidden_layers)
        self.out_h = nn.Linear(d, cs)
        nn.init.zeros_(self.out_h.weight)
        nn.init.zeros_(self.out_h.bias)
        self.out_v = VectorLinear(cv, cv_latent, zero_init=True)

    def forward(self, graphs: WorldGraphs, zbar: Latent,
                history: Sequence[Latent], node_forcing: Tensor) -> Latent:
        h, v = self.embed([zbar, *history], node_forcing, graphs.latent_struct)
        h, v = self.processor(graphs.latent_mesh, h, v)
        return Latent(self.out_h(h), self.out_v(v)).project(graphs.latent_pos)
