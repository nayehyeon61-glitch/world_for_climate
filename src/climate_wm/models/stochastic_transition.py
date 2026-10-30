"""DIAMOND-style conditional diffusion of the latent residual (Stage 3).

Port of `diamond/src/models/diffusion/{denoiser,diffusion_sampler}.py` from
pixel frames to the latent graph:

  DIAMOND                              here
  -------                              ----
  next frame x_{t+1}                   residual r = (Z_{t+1} - Zbar_{t+1}) / scale
  past frames obs_{t-k+1:t}            past latents Z_{t-k+1:t} and Zbar_{t+1}
  actions a_{t-k+1:t} (embedding)      known forcings c_{t+1} at latent nodes
  UNet + AdaGroupNorm(noise emb)       Shell-GNN + AdaLayerNorm(noise emb)
  EDM preconditioning, offset noise    identical (c_in, c_out, c_skip, c_noise)
  Euler / Heun sampler, Karras sigmas  identical (default 3 steps)

The residual lives in the tangent space T_{Zbar} M_Z (design 6.D): noise for
vector channels is projected onto the tangent planes, and the sample is
mapped back with Exp_{Zbar}. Unlike DIAMOND we do not quantise the output
(no pixels) and start sampling from sigma_max * N(0, I) as in EDM.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .graphs import WorldGraphs
from .latent import Latent, LatentEmbedding, scale_per_sample
from .layers import FourierFeatures, ShellProcessor, VectorLinear


@dataclasses.dataclass
class DiffusionConfig:
    layers: int = 4
    cond_dim: int = 128
    sigma_data: float = 1.0  # residuals are rescaled to unit RMS
    sigma_offset_noise: float = 0.1
    # Training noise distribution (DIAMOND trainer.yaml values).
    sigma_loc: float = -0.4
    sigma_scale: float = 1.2
    sigma_min_train: float = 2e-3
    sigma_max_train: float = 20.0
    # Sampler (DIAMOND world_model_env defaults).
    num_steps_denoising: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: float = 7.0
    order: int = 1  # 1: Euler, 2: Heun
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0


@dataclasses.dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


class LatentDenoiser(nn.Module):

    def __init__(self, graphs: WorldGraphs, num_shells: int, history: int,
                 forcing_dim: int, cs: int, cv_latent: int, d: int, cv: int,
                 hidden_layers: int, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.cs = cs
        # Inputs: noisy residual, Zbar_{t+1}, Z_{t-k+1:t}.
        self.embed = LatentEmbedding(history + 2, cs, cv_latent, forcing_dim,
                                     graphs.struct_dim, num_shells, d, cv,
                                     hidden_layers)
        self.noise_emb = FourierFeatures(cfg.cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim), nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.processor = ShellProcessor(d, cv, graphs.latent_mesh.edge_dim,
                                        cfg.layers, num_shells, hidden_layers,
                                        cond_dim=cfg.cond_dim)
        self.out_h = nn.Linear(d, cs)
        nn.init.zeros_(self.out_h.weight)
        nn.init.zeros_(self.out_h.bias)
        self.out_v = VectorLinear(cv, cv_latent, zero_init=True)

    def sample_sigma_training(self, n: int, device) -> Tensor:
        c = self.cfg
        s = torch.randn(n, device=device) * c.sigma_scale + c.sigma_loc
        return s.exp().clip(c.sigma_min_train, c.sigma_max_train)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        c = self.cfg
        sigma = (sigma ** 2 + c.sigma_offset_noise ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + c.sigma_data ** 2).sqrt()
        c_skip = c.sigma_data ** 2 / (sigma ** 2 + c.sigma_data ** 2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(c_in, c_out, c_skip, c_noise)

    def apply_noise(self, x: Latent, sigma: Tensor, pos: Tensor) -> Latent:
        b, s = x.h.shape[:2]
        off = self.cfg.sigma_offset_noise
        offset = Latent(off * torch.randn(b, s, 1, x.h.shape[-1], device=x.h.device),
                        off * torch.randn(b, s, 1, x.v.shape[-2], 3, device=x.h.device))
        offset = Latent(offset.h.expand_as(x.h), offset.v.expand_as(x.v))
        noise = Latent(torch.randn_like(x.h), torch.randn_like(x.v))
        return (x + offset + scale_per_sample(noise, sigma)).project(pos)

    def compute_model_output(self, graphs: WorldGraphs, noisy: Latent,
                             cs: Conditioners, zbar: Latent,
                             history: Sequence[Latent],
                             node_forcing: Tensor) -> Latent:
        rescaled = scale_per_sample(noisy, cs.c_in)
        h, v = self.embed([rescaled, zbar, *history], node_forcing,
                          graphs.latent_struct)
        cond = self.cond_proj(self.noise_emb(cs.c_noise))
        h, v = self.processor(graphs.latent_mesh, h, v, cond)
        return Latent(self.out_h(h), self.out_v(v)).project(graphs.latent_pos)

    def wrap_model_output(self, noisy: Latent, out: Latent,
                          cs: Conditioners) -> Latent:
        return scale_per_sample(noisy, cs.c_skip) + scale_per_sample(out, cs.c_out)

    def denoise(self, graphs: WorldGraphs, noisy: Latent, sigma: Tensor,
                zbar: Latent, history: Sequence[Latent],
                node_forcing: Tensor) -> Latent:
        cs = self.compute_conditioners(sigma)
        out = self.compute_model_output(graphs, noisy, cs, zbar, history,
                                        node_forcing)
        return self.wrap_model_output(noisy, out, cs)

    def loss_per_sample(self, graphs: WorldGraphs, target: Latent, zbar: Latent,
                        history: Sequence[Latent], node_forcing: Tensor
                        ) -> Tuple[Tensor, Latent]:
        """EDM denoising loss [B] (as DIAMOND) and the denoised estimate."""
        b = target.batch_size
        sigma = self.sample_sigma_training(b, target.h.device)
        noisy = self.apply_noise(target, sigma, graphs.latent_pos)
        cs = self.compute_conditioners(sigma)
        out = self.compute_model_output(graphs, noisy, cs, zbar, history,
                                        node_forcing)
        inv_c_out = 1 / cs.c_out
        tgt = scale_per_sample(target - scale_per_sample(noisy, cs.c_skip), inv_c_out)
        err = torch.cat([(out.h.float() - tgt.h.float()).flatten(1),
                         (out.v.float() - tgt.v.float()).flatten(1)], 1).square().mean(1)
        denoised = self.wrap_model_output(noisy, out, cs)
        return err, denoised

    def loss(self, graphs: WorldGraphs, target: Latent, zbar: Latent,
             history: Sequence[Latent], node_forcing: Tensor
             ) -> Tuple[Tensor, Latent]:
        """EDM denoising loss (as DIAMOND) and the denoised estimate."""
        err, denoised = self.loss_per_sample(graphs, target, zbar, history, node_forcing)
        return err.mean(), denoised


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float,
                 rho: float, device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


class LatentDiffusionSampler:
    """DIAMOND's DiffusionSampler on Latent objects."""

    def __init__(self, denoiser: LatentDenoiser, cfg: Optional[DiffusionConfig] = None):
        self.denoiser = denoiser
        self.cfg = cfg or denoiser.cfg

    @torch.no_grad()
    def sample(self, graphs: WorldGraphs, zbar: Latent,
               history: Sequence[Latent], node_forcing: Tensor,
               num_steps: Optional[int] = None) -> Latent:
        c = self.cfg
        device = zbar.h.device
        sigmas = build_sigmas(num_steps or c.num_steps_denoising, c.sigma_min,
                              c.sigma_max, c.rho, device)
        b = zbar.batch_size
        ones = torch.ones(b, device=device)
        pos = graphs.latent_pos
        x = Latent(torch.randn_like(zbar.h), torch.randn_like(zbar.v)).project(pos)
        x = x.scale(sigmas[0])
        gamma_ = min(c.s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        den: Callable = lambda x_, s_: self.denoiser.denoise(
            graphs, x_, s_ * ones, zbar, history, node_forcing)
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            gamma = gamma_ if c.s_tmin <= sigma <= c.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = Latent(torch.randn_like(x.h), torch.randn_like(x.v)).project(pos)
                x = x + eps.scale(c.s_noise * (sigma_hat ** 2 - sigma ** 2) ** 0.5)
            denoised = den(x, sigma_hat)
            d = (x - denoised).scale(1 / sigma_hat)
            dt = next_sigma - sigma_hat
            if c.order == 1 or next_sigma == 0:
                x = x + d.scale(dt)
            else:
                x_2 = x + d.scale(dt)
                d_2 = (x_2 - den(x_2, next_sigma)).scale(1 / next_sigma)
                x = x + (d + d_2).scale(0.5 * dt)
        return x.project(pos)
