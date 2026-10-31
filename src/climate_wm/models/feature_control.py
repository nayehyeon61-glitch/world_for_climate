"""Stage 7: a controller that steers a frozen world model through its GNN features.

World Models (Ha & Schmidhuber 2018) train the world model (V, M) first,
freeze it, and train only a small controller C for reward. Here the frozen
model is the Shell-GNN world model and the controller's action modulates the
latent GNN features before every transition step:

    h  <-  h + scale * (h * (U gamma) + U beta)       (scalars of each shell and node)

with (gamma, beta) in R^r per (shell, patch), blended onto the latent nodes
with the soft patch weights, and U a fixed d x r orthonormal basis. Only
scalars change, so the SO(3) equivariance of the latent state is kept.

What the controller sees, per (shell, patch):
    * the pooled latent scalars and vector norms of the current state,
    * physical diagnostics of the decoded current state (stage 5 module:
      hydrostatic and geostrophic residuals, N^2, saturation deficit,
      divergence, vorticity, surface coupling), averaged per family and shell,
    * the fraction of the shell's input that the prediction game hid (`mask`),
    * the shell and patch identity (all an open-loop controller sees).

The last layer starts at zero, so an untrained controller leaves the model
unchanged. `explore` > 0 samples Gaussian actions around the mean and, when
`record` is a list, appends (features, action) of every call for the policy
gradient (train/prediction_game.py).
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..eval.physics_diag import SCALES, DiagnosticsConfig, PhysicsDiagnostics
from ..geometry.shells import SpherePatches
from .latent import Latent

FAMILIES = tuple(SCALES)  # hydrostatic, geostrophic, stability, saturation, divergence, vorticity, surface


@dataclasses.dataclass
class ControlConfig:
    rank: int = 4                # r: FiLM directions per (shell, patch)
    patch_splits: int = 0        # icosphere level of the patches (0: 12 patches)
    hidden: int = 64
    action_scale: float = 0.1
    use_state: bool = True       # False: open loop (shell and patch identity only)
    use_diagnostics: bool = True
    seed: int = 0


class FeatureController(nn.Module):

    def __init__(self, model, lat: np.ndarray, lon: np.ndarray, cfg: ControlConfig, stats=None):
        super().__init__()
        self.cfg = cfg
        layout = model.layout
        g = model.graphs
        self.num_shells = len(layout.shells)
        self.cs = model.cfg.gnn.latent_scalars
        self.cv = model.cfg.gnn.latent_vectors
        self.patches = SpherePatches(g.latent_pos.cpu().numpy(), cfg.patch_splits)
        self.num_patches = self.patches.num_patches
        self.diag = PhysicsDiagnostics(layout, lat, lon, DiagnosticsConfig(enabled=True))
        if stats is not None:
            self.diag.set_stats(stats)
        # family x shell averaging of the diagnostic channels: [C_diag, S * F]
        s, f = self.num_shells, len(FAMILIES)
        avg = torch.zeros(len(self.diag.channels), s * f)
        for c, ch in enumerate(self.diag.channels):
            avg[c, layout.shell_of(ch) * f + FAMILIES.index(ch.name[len("diag_"):])] = 1.0
        self.register_buffer("diag_avg", avg / avg.sum(0).clamp_min(1.0), persistent=False)
        self.dynamic_shells = [sorted(p.scalar + p.vector_u + p.vector_v) for p in layout.dynamic_shells]
        self.target_to_dynamic = list(layout.target_to_dynamic)
        gen = torch.Generator().manual_seed(cfg.seed)
        basis, _ = torch.linalg.qr(torch.randn(self.cs, cfg.rank, generator=gen))
        self.register_buffer("basis", basis)  # U [d, r]
        self.in_dim = self.cs + self.cv + f + 1 + s + self.num_patches
        self.net = nn.Sequential(nn.Linear(self.in_dim, cfg.hidden), nn.SiLU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
                                 nn.Linear(cfg.hidden, 2 * cfg.rank))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Game state, set per episode (not parameters):
        self.mask: Optional[Tensor] = None      # [B, C_dyn, H, W], 1 = hidden by the game
        self.explore = 0.0                      # std of the Gaussian policy (0: mean action)
        self.record: Optional[list] = None      # receives (features, action) per call
        self.decoded: Optional[Tensor] = None   # D(z) of the next call's state, when the caller has it
        self.enabled = True
        self.zero_diagnostics = False           # eval-time knock-out of the physics input
        self.random_actions = False             # eval-time control: actions at random places

    # ------------------------------------------------------------ features
    def _pool(self, x: Tensor) -> Tensor:
        """Node features [B, S, N, C] -> patch means [B, S, P, C]."""
        b, s, n, c = x.shape
        pooled = self.patches.pool(x.permute(0, 2, 1, 3).reshape(b, n, s * c))
        return pooled.reshape(b, -1, s, c).permute(0, 2, 1, 3)

    def _grid_to_patches(self, model, x: Tensor) -> Tensor:
        """Grid fields [B, C, H, W] -> patch means [B, P, C]."""
        return self.patches.pool(model.graphs.grid_to_latent_mean(x))

    def features(self, model, z: Latent) -> Tensor:
        """Controller input [B, S, P, in_dim] for latent state z."""
        b, s = z.h.shape[:2]
        p, f = self.num_patches, len(FAMILIES)
        state = torch.cat([self._pool(z.h), self._pool(z.v.norm(dim=-1))], -1)  # [B, S, P, cs + cv]
        diag = state.new_zeros(b, s, p, f)
        if self.cfg.use_diagnostics and not self.zero_diagnostics:
            decoded = self.decoded if self.decoded is not None else model.decode(z)
            dyn = decoded[:, self.target_to_dynamic]
            d = self._grid_to_patches(model, self.diag(dyn)) @ self.diag_avg  # [B, P, S * F]
            diag = d.reshape(b, p, s, f).permute(0, 2, 1, 3)
        hidden = state.new_zeros(b, s, p, 1)
        if self.mask is not None:
            per_shell = torch.stack([self.mask[:, idx].float().mean(1) for idx in self.dynamic_shells], 1)
            hidden = self._grid_to_patches(model, per_shell).permute(0, 2, 1)[..., None]
        if not self.cfg.use_state:
            state, diag = torch.zeros_like(state), torch.zeros_like(diag)
        ident = torch.cat([torch.eye(s, device=z.h.device)[:, None].expand(s, p, s),
                           torch.eye(p, device=z.h.device)[None].expand(s, p, p)], -1)
        return torch.cat([state, diag, hidden, ident.expand(b, s, p, s + p)], -1)

    def mean(self, feats: Tensor) -> Tensor:
        """Mean action [B, S, P, 2r]."""
        return self.net(feats)

    # -------------------------------------------------------------- action
    def apply(self, z: Latent, action: Tensor) -> Latent:
        """FiLM of the latent scalars by the (shell, patch) action [B, S, P, 2r]."""
        idx, w = self.patches.index, self.patches.weight  # [N, k]
        node = (action[:, :, idx] * w[None, None, :, :, None]).sum(3)  # [B, S, N, 2r]
        r = self.cfg.rank
        gamma, beta = node[..., :r] @ self.basis.T, node[..., r:] @ self.basis.T
        return Latent(z.h + self.cfg.action_scale * (z.h * gamma + beta), z.v)

    def forward(self, model, z: Latent) -> Latent:
        if not self.enabled:
            return z
        with torch.no_grad():  # the controller observes the state; gradients reach it through its action
            feats = self.features(model, z)
        self.decoded = None
        mu = self.mean(feats)
        if self.random_actions:  # the same actions at shuffled (shell, patch) positions
            flat = mu.detach().flatten(1, 2)
            perm = torch.argsort(torch.rand(flat.shape[:2], device=flat.device), 1)
            action = flat.gather(1, perm[..., None].expand_as(flat)).view_as(mu)
        elif self.explore > 0:
            action = mu.detach() + self.explore * torch.randn_like(mu)
        else:
            action = mu
        if self.record is not None:
            self.record.append((feats.detach(), action.detach()))
        return self.apply(z, action)

    def log_prob(self, feats: Tensor, action: Tensor) -> Tensor:
        """Gaussian log-density per sample [B] (up to a constant), mean over action dims."""
        err = (action - self.mean(feats)).square().flatten(1)
        return -err.mean(1) / (2 * self.explore ** 2)


@contextlib.contextmanager
def controlled(model, controller: Optional[FeatureController]) -> Iterator[None]:
    """Attaches `controller` to `model.step` (not as a submodule, so the world
    model's state dict and checkpoints stay unchanged)."""
    old = getattr(model, "feature_control", None)
    object.__setattr__(model, "feature_control", controller)
    try:
        yield
    finally:
        object.__setattr__(model, "feature_control", old)


def controller_state(controller: FeatureController) -> dict:
    return {"cfg": dataclasses.asdict(controller.cfg), "state": controller.state_dict()}


def load_controller(model, lat, lon, saved: dict, stats=None) -> FeatureController:
    ctrl = FeatureController(model, lat, lon, ControlConfig(**saved["cfg"]), stats)
    ctrl.load_state_dict(saved["state"])
    return ctrl.to(model.residual_scale.device)


def frozen(model) -> List[nn.Parameter]:
    """Freezes every world-model parameter (stage 7: only the controller learns)."""
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(False)
    return params
