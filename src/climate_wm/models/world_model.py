"""Geometric stochastic climate world model (research design section 6).

    A. Z_t = E_phi(X_t)                                   Shell-GNN encoder
    B. xi_t = f_theta(Z_{t-k+1:t}, c_{t+1}) in so(3)^{SxP}  Lie generator
    C. Zbar_{t+1} = Phi_{exp(dt xi_t)}(Z_t)                structural transition
    D. Z_{t+1} = Exp_Zbar(r),  r ~ p_theta(r | Zbar, Z_hist, c)   residual
         deterministic (ResidualDynamics) or DIAMOND-style diffusion
    E. Xhat_{t+1} = D_psi(Z_{t+1})                         Shell-GNN decoder

With k = 2 latent frames the model consumes exactly GraphCast's inputs
(X[t-6h], X[t], forcings at t+6h, static) and predicts GraphCast's targets;
rollouts are autoregressive in latent space ("imagination", as in DIAMOND).

Transition modes (research design ablations, section 10):
  euclidean            Zbar = Z_t, deterministic residual      (B2 / B3)
  lie                  Lie structural transition only          (B4)
  lie_residual         Lie + deterministic residual            (B5)
  lie_diffusion        Lie + diffusion residual                (B6)
  euclidean_diffusion  Zbar = Z_t + diffusion residual         (Euclidean latent
                                                               diffusion baseline, H3)
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..data.graphcast_mapping import ChannelLayout
from ..train.losses import GraphCastLoss
from .graphs import MeshConfig, WorldGraphs
from .latent import Latent
from .lie_generator import LieConfig, LieGenerator
from .residual_dynamics import ResidualDynamics
from .shell_gnn_decoder import ShellGNNDecoder
from .shell_gnn_encoder import ShellGNNConfig, ShellGNNEncoder
from .stochastic_transition import (DiffusionConfig, LatentDenoiser,
                                    LatentDiffusionSampler)

TRANSITIONS = ("euclidean", "lie", "lie_residual", "lie_diffusion",
               "euclidean_diffusion")


@dataclasses.dataclass
class WorldModelConfig:
    mesh: MeshConfig = dataclasses.field(default_factory=MeshConfig)
    gnn: ShellGNNConfig = dataclasses.field(default_factory=ShellGNNConfig)
    lie: LieConfig = dataclasses.field(default_factory=LieConfig)
    diffusion: DiffusionConfig = dataclasses.field(default_factory=DiffusionConfig)
    transition: str = "lie_diffusion"
    history: int = 2  # latent frames conditioning the transition (= GraphCast inputs)
    residual_layers: int = 2

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WorldModelConfig":
        d = dict(d or {})
        sub = {"mesh": MeshConfig, "gnn": ShellGNNConfig, "lie": LieConfig,
               "diffusion": DiffusionConfig}
        kwargs = {k: sub[k](**d.pop(k)) for k in list(d) if k in sub}
        return cls(**kwargs, **d)


@dataclasses.dataclass
class StepOutput:
    z: Latent  # Z_{t+1}
    zbar: Latent  # structural prediction Zbar_{t+1}
    omega: Optional[Tensor] = None  # [B, S, P, 3]
    joints: Optional[Tensor] = None  # [B, S, P, 3] joint rotations (chain)


class ClimateWorldModel(nn.Module):

    def __init__(self, cfg: WorldModelConfig, layout: ChannelLayout,
                 grid_lat: np.ndarray, grid_lon: np.ndarray):
        super().__init__()
        if cfg.transition not in TRANSITIONS:
            raise ValueError(f"transition must be one of {TRANSITIONS}")
        if cfg.history > layout.task.input_steps:
            raise ValueError("history cannot exceed the task's input steps")
        self.cfg = cfg
        self.layout = layout
        self.graphs = WorldGraphs(grid_lat, grid_lon, cfg.mesh)
        self.encoder = ShellGNNEncoder(layout, self.graphs, cfg.gnn)
        self.decoder = ShellGNNDecoder(layout, self.graphs, cfg.gnn)
        g = cfg.gnn
        common = dict(graphs=self.graphs, num_shells=len(layout.shells),
                      history=cfg.history, forcing_dim=len(layout.forcings),
                      cs=g.latent_scalars, cv_latent=g.latent_vectors,
                      d=g.hidden_dim, cv=g.vector_dim, hidden_layers=g.hidden_layers)
        t = cfg.transition
        self.lie = LieGenerator(**common, cfg=cfg.lie) if t.startswith("lie") else None
        self.residual = (ResidualDynamics(**common, layers=cfg.residual_layers)
                         if t in ("euclidean", "lie_residual") else None)
        self.denoiser = (LatentDenoiser(**common, cfg=cfg.diffusion)
                         if t.endswith("diffusion") else None)
        self.loss_fn = GraphCastLoss(layout, grid_lat)
        # RMS of the latent residual (scalar part, vector part); calibrated once
        # at the start of the stochastic stage so that sigma_data = 1 holds.
        self.register_buffer("residual_scale", torch.ones(2))
        self.register_buffer("residual_calibrated", torch.tensor(False))

    # ------------------------------------------------------------------ I/O
    def encoder_input(self, dynamic: Tensor, forcing: Tensor, static: Tensor) -> Tensor:
        """[B, C_dyn, H, W], [B, C_f, H, W], [B, C_s, H, W] -> [B, C_enc, H, W]."""
        return torch.cat([dynamic, forcing, static], 1)

    def encode(self, dynamic: Tensor, forcing: Tensor, static: Tensor) -> Latent:
        return self.encoder(self.graphs, self.encoder_input(dynamic, forcing, static))

    def encode_target(self, batch: Dict[str, Tensor], k: int) -> Latent:
        """Z_{t+k+1} = E(X_{t+k+1}) from the k-th target frame."""
        dyn = batch["targets"][:, k][:, self.layout.target_to_dynamic]
        return self.encode(dyn, batch["forcings_tgt"][:, k], batch["static"])

    def decode(self, z: Latent) -> Tensor:
        return self.decoder(self.graphs, z)

    def node_forcing(self, forcing: Tensor) -> Tensor:
        return self.graphs.grid_to_latent_mean(forcing)

    def encode_history(self, batch: Dict[str, Tensor]) -> List[Latent]:
        n = batch["inputs"].shape[1]
        return [self.encode(batch["inputs"][:, j], batch["forcings_in"][:, j],
                            batch["static"]) for j in range(n - self.cfg.history, n)]

    # ----------------------------------------------------------- transition
    def structural(self, history: Sequence[Latent], nf: Tensor):
        """Zbar_{t+1} = Phi_{exp(dt xi)}(Z_t) (or Z_t without Lie generator)."""
        if self.lie is None:
            return history[-1], None, None
        omega, omega_node, joints = self.lie(self.graphs, history, nf)
        return self.lie.act(self.graphs, history[-1], omega_node), omega, joints

    def _residual_scale(self, z: Latent, inverse: bool = False) -> Latent:
        s_h, s_v = self.residual_scale
        if inverse:
            return Latent(z.h / s_h, z.v / s_v)
        return Latent(z.h * s_h, z.v * s_v)

    def step(self, history: Sequence[Latent], forcing_next: Tensor,
             sample: bool = True, num_denoising_steps: Optional[int] = None
             ) -> StepOutput:
        nf = self.node_forcing(forcing_next)
        zbar, omega, joints = self.structural(history, nf)
        pos = self.graphs.latent_pos
        if self.residual is not None:
            z = zbar.exp(self.residual(self.graphs, zbar, history, nf), pos)
        elif self.denoiser is not None and sample:
            r = LatentDiffusionSampler(self.denoiser).sample(
                self.graphs, zbar, history, nf, num_denoising_steps)
            z = zbar.exp(self._residual_scale(r), pos)
        else:
            z = zbar
        return StepOutput(z=z, zbar=zbar, omega=omega, joints=joints)

    # ------------------------------------------------------------- training
    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        """Recompute every message-passing layer in backward (large grids)."""
        from .layers import GeometricInteraction
        for m in self.modules():
            if isinstance(m, GeometricInteraction):
                m.checkpoint = enabled

    @torch.no_grad()
    def calibrate_residual_scale(self, batch: Dict[str, Tensor]) -> None:
        hist = self.encode_history(batch)
        nf = self.node_forcing(batch["forcings_tgt"][:, 0])
        zbar, _, _ = self.structural(hist, nf)
        r = self.encode_target(batch, 0) - zbar
        self.residual_scale.copy_(torch.stack([
            r.h.square().mean().sqrt(), r.v.square().mean().sqrt()]).clamp_min(1e-4))
        self.residual_calibrated.fill_(True)

    def losses(self, batch: Dict[str, Tensor], stage: str,
               weights: Dict[str, float], rollout_steps: int = 1
               ) -> Tuple[Tensor, Dict[str, float]]:
        """Stage losses (research design section 7).

        repr:       L_rec
        dynamics:   L_forecast (multi-step, latent rollout) + L_latent + L_rec
                    + L_geo (|omega|^2) + L_joint (torsion springs)
        stochastic: L_gen (EDM denoising of the latent residual, DIAMOND)
                    + optional L_forecast on the denoised estimate
        """
        w = lambda k: float(weights.get(k, 0.0))
        logs: Dict[str, Tensor] = {}
        steps = min(rollout_steps, batch["targets"].shape[1])
        if stage == "repr":
            rec = sum(self.loss_fn(self.decode(self.encode_target(batch, k)),
                                   batch["targets"][:, k]) for k in range(steps)) / steps
            logs["rec"] = rec
            total = w("rec") * rec
        elif stage in ("dynamics", "stochastic"):
            history = self.encode_history(batch)
            total = torch.zeros((), device=batch["targets"].device)
            for k in range(steps):
                target_x = batch["targets"][:, k]
                forcing = batch["forcings_tgt"][:, k]
                if stage == "dynamics":
                    out = self.step(history, forcing, sample=False)
                    z_next = out.z
                    z_tgt = self.encode_target(batch, k)
                    terms = {"forecast": self.loss_fn(self.decode(z_next), target_x),
                             "latent": z_next.squared_error(z_tgt.detach())}
                    if k == 0:
                        terms["rec"] = self.loss_fn(self.decode(z_tgt), target_x)
                    if out.omega is not None:
                        terms["geo"] = out.omega.square().sum(-1).mean()
                    if out.joints is not None and self.cfg.lie.shell_coupling == "chain":
                        terms["joint"] = out.joints[:, 1:].square().sum(-1).mean()
                else:
                    nf = self.node_forcing(forcing)
                    with torch.no_grad():
                        zbar, _, _ = self.structural(history, nf)
                        z_tgt = self.encode_target(batch, k)
                    r_star = self._residual_scale(z_tgt - zbar, inverse=True)
                    gen, denoised = self.denoiser.loss(self.graphs, r_star, zbar,
                                                       history, nf)
                    z_next = zbar.exp(self._residual_scale(denoised),
                                      self.graphs.latent_pos)
                    terms = {"gen": gen}
                    if w("forecast"):
                        terms["forecast"] = self.loss_fn(self.decode(z_next), target_x)
                    z_next = z_next.detach()  # DIAMOND feeds back the denoised frame
                for name, value in terms.items():
                    logs[name] = logs.get(name, 0.0) + value / steps
                    total = total + w(name) * value / steps
                history = list(history[1:]) + [z_next]
        else:
            raise ValueError(f"unknown stage {stage}")
        return total, {k: float(v.detach() if torch.is_tensor(v) else v)
                       for k, v in logs.items()}

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def rollout(self, batch: Dict[str, Tensor], steps: int, members: int = 1,
                num_denoising_steps: Optional[int] = None,
                return_omega: bool = False):
        """Autoregressive latent rollout.

        Returns normalised predictions [M, B, L, C_tgt, H, W] (and the patch
        Lie algebra elements per step if requested).
        """
        b = batch["inputs"].shape[0]
        rep = {k: (v.repeat_interleave(members, 0) if torch.is_tensor(v) and v.ndim > 1
                   else v) for k, v in batch.items()}
        history = self.encode_history(rep)
        preds, omegas = [], []
        for k in range(steps):
            out = self.step(history, rep["forcings_tgt"][:, k], sample=True,
                            num_denoising_steps=num_denoising_steps)
            preds.append(self.decode(out.z))
            omegas.append(out.omega)
            history = list(history[1:]) + [out.z]
        pred = torch.stack(preds, 1)  # [B*M, L, C, H, W]
        pred = pred.unflatten(0, (b, members)).transpose(0, 1)
        if return_omega:
            return pred, omegas
        return pred
