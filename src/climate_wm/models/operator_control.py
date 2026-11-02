"""Stage 11: a bank of neural operators with a selection policy, as the controller.

감자's hypothesis (10/31 21:42-21:43 KST): what separates reinforcement learning
from plain deep learning is control by choosing among several networks. So the
controller bundles several neural operators, and RL chooses between them.

Operators: K spherical Fourier neural operators (SFNO, Bonev et al. 2023, from
the neuraloperator library) on the physics core's Gaussian grid. Each maps the
controller's view of the state to one control field per field and layer of
the stage 9 tendency controller (T, zeta, q in four layers, Phi_b). The view
is the layer-mean T anomaly, zeta, D and q, the Phi_b anomaly, the game mask
per variable group, sin(lat) and the step. The control field is truncated at
`out_degree` and scaled by the stage 9 action units. Each operator's last
layer starts at zero, so an untrained bank does nothing.

Selection: in each latitude band (`band_edges`, smooth partition of unity with
logistic tapers) the policy chooses one of K + 1 options: operator 1..K, or no
control. It reads the stage 9 controller's features (low harmonics of the state
and the mask, and the step). The control is
    u(x) = sum_r w_r(lat(x)) O_{k_r}(state)(x).
Modes (train/prediction_game.GameConfig.mode):
  hybrid: the choices are sampled. The policy learns by REINFORCE on the
          group-normalised reward-to-go, since a choice has no gradient. The
          operators learn by the exact gradient of the episode return through
          the frozen model, given those choices. This is the actor loss
          lambda L_RL + (1 - lambda) L_model of Petrovskyi and Dolgikh (2026),
          here with the operators as the actor and the physics core as the
          differentiable world model.
  grad:   `selection: soft` mixes the options with softmax weights (a
          differentiable mixture of experts, no choice), K = 1 is a single
          operator; plain deep learning.
Evaluation knock-outs: `random_actions` (uniform random choices), `fixed_choice`
(always option j) and `zero_diagnostics` (the policy and the operators see no
state).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .tendency_control import FEATURE_SCALE, MASK_GROUPS, TendencyControlConfig, TendencyController


def _sfno(in_channels: int, out_channels: int, hidden: int, layers: int, modes: int) -> nn.Module:
    try:
        from neuralop.layers.spherical_convolution import SphericalConv
        from neuralop.models import SFNO
    except ImportError as e:  # optional dependency (requirements: neuraloperator, torch-harmonics)
        raise ImportError("stage 11 needs the neuraloperator and torch-harmonics packages") from e
    conv = functools.partial(SphericalConv, sht_grids="legendre-gauss")  # the core's Gaussian grid
    net = SFNO(n_modes=(modes, modes), in_channels=in_channels, out_channels=out_channels,
               hidden_channels=hidden, n_layers=layers, positional_embedding=None, conv_module=conv)
    last = net.projection.fcs[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


@dataclasses.dataclass
class OperatorControlConfig(TendencyControlConfig):
    kind: str = "operator"
    n_operators: int = 4
    selection: str = "choice"          # choice (sampled / argmax) | soft (softmax mixture)
    band_edges: Tuple[float, ...] = (-30.0, 30.0)  # latitude (deg) between the selection bands
    band_taper: float = 5.0            # logistic width of the band edges (deg)
    operator_hidden: int = 16
    operator_layers: int = 2
    operator_modes: int = 16           # SFNO modes (degree and order)
    out_degree: int = 10               # truncation of the control field

    def __post_init__(self):
        super().__post_init__()
        self.band_edges = tuple(self.band_edges)
        if self.selection not in ("choice", "soft"):
            raise ValueError("selection must be choice or soft")
        if self.target != "state":
            raise ValueError("the operator bank controls the physical state")


class OperatorBankController(TendencyController):
    """Drop-in for TendencyController in the stage 9 game (train/physics_game.play_physics)."""

    def __init__(self, model, cfg: OperatorControlConfig, diff_std: Optional[Dict[str, float]] = None):
        super().__init__(model, cfg, diff_std)
        sh = model.core.sh
        n_g = len(self.groups)
        self.n_options = cfg.n_operators + 1  # the last option: no control
        self.n_bands = len(cfg.band_edges) + 1
        self.n_out = len(self.layer_fields) * n_g + ("phib" in cfg.fields)
        self.n_in = len(self.state_fields) * n_g + 1 + len(MASK_GROUPS) + 2
        self.operators = nn.ModuleList([_sfno(self.n_in, self.n_out, cfg.operator_hidden, cfg.operator_layers,
                                              min(cfg.operator_modes, sh.trunc + 1))
                                        for _ in range(cfg.n_operators)])
        hidden = cfg.hidden
        self.net = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, self.n_bands * self.n_options))
        nn.init.zeros_(self.net[-1].weight)  # uniform choice at the start
        nn.init.zeros_(self.net[-1].bias)
        lat = torch.rad2deg(torch.asin(model.sinlat_g.float()))
        edges = torch.tensor((-1e9,) + cfg.band_edges + (1e9,))
        sig = lambda x: torch.sigmoid(x / cfg.band_taper)
        w = torch.stack([sig(lat - edges[r]) - sig(lat - edges[r + 1]) for r in range(self.n_bands)])
        self.register_buffer("band_w", w / w.sum(0, keepdim=True), persistent=False)  # [R, H_gauss]
        n = torch.arange(sh.trunc + 1)
        self.register_buffer("out_keep", (n <= cfg.out_degree).float(), persistent=False)  # per degree n
        self.fixed_choice: Optional[int] = None  # eval knock-out
        self.last_choice: Optional[Tensor] = None
        self.usage: Optional[list] = None        # eval: the choices [B, R] of every step
        self.mask_grid: Optional[Tensor] = None

    # ------------------------------------------------------------ inputs
    def set_mask(self, mask: Optional[Tensor]) -> None:
        super().set_mask(mask)
        self.mask_grid = None
        if mask is not None:
            sh = self.model_ref[0].core.sh
            per = torch.stack([mask[:, idx].float().mean(1) if idx else torch.zeros_like(mask[:, 0])
                               for idx in self.mask_index], 1)
            self.mask_grid = sh.to_grid(sh.from_data(per))  # [B, groups, H_g, W_g]

    def fields(self, state: Dict[str, Tensor], k: int) -> Tensor:
        """The operators' input [B, n_in, H_g, W_g] on the Gaussian grid."""
        sh = self.model_ref[0].core.sh
        lev = lambda x: torch.view_as_complex(torch.einsum("gl,bl...c->bg...c", self.avg,
                                                           torch.view_as_real(x)).contiguous())
        parts = []
        for f in self.state_fields:
            g = sh.to_grid(lev(state[f]))
            if f == "temp":
                g = g - self.t_ref_layer[None, :, None, None]
            parts.append(g / FEATURE_SCALE[f])
        parts.append(((sh.to_grid(state["phib"]) - self.phib_mean) / FEATURE_SCALE["phib"])[:, None])
        x = torch.cat(parts, 1)
        if not self.cfg.use_state or self.zero_diagnostics:
            x = torch.zeros_like(x)
        b, _, h, w = x.shape
        m = self.mask_grid if self.mask_grid is not None else x.new_zeros(b, len(MASK_GROUPS), h, w)
        sinlat = self.model_ref[0].sinlat_g.to(x)[None, None, :, None].expand(b, 1, h, w)
        step = torch.full_like(sinlat, min(k, self.cfg.max_horizon - 1) / self.cfg.max_horizon)
        return torch.cat([x, m, sinlat, step], 1)

    # ------------------------------------------------------------ choice
    def _weights(self, logits: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Option weights [B, R, K + 1] and the hard choice [B, R] (None for soft)."""
        b = logits.shape[0]
        if self.fixed_choice is not None:
            choice = torch.full((b, self.n_bands), self.fixed_choice, dtype=torch.long, device=logits.device)
        elif self.random_actions:
            choice = torch.randint(self.n_options, (b, self.n_bands), device=logits.device)
        elif self.cfg.selection == "soft":
            return torch.softmax(logits, -1), None
        elif self.explore > 0:
            choice = torch.distributions.Categorical(logits=logits).sample()
        else:
            choice = logits.argmax(-1)
        return nn.functional.one_hot(choice, self.n_options).to(logits.dtype), choice

    def control_field(self, x: Tensor, weights: Tensor) -> Tensor:
        """u [B, n_out, H_g, W_g] in action units: band-weighted mixture of the chosen operators."""
        u = x.new_zeros(x.shape[0], self.n_out, *x.shape[-2:])
        for j, op in enumerate(self.operators):
            wj = torch.einsum("br,rh->bh", weights[:, :, j], self.band_w)  # [B, H_g]
            if bool((wj.abs() > 0).any()):
                u = u + wj[:, None, :, None] * op(x)
        return u

    def apply_field(self, state: Dict[str, Tensor], u: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        """Adds the truncated control fields to the spectral state; returns the state and 1/2 |a|^2."""
        sh = self.model_ref[0].core.sh
        spec = sh.to_spec(u) * self.out_keep[None, None, None, :]  # [B, n_out, M, N]; N is the degree
        out = dict(state)
        n_g = len(self.groups)
        for i, f in enumerate(self.layer_fields):
            inc = spec[:, i * n_g:(i + 1) * n_g] * self.unit[i, :, None, None]
            out[f] = state[f] + torch.view_as_complex(torch.einsum(
                "gl,bg...c->bl...c", self.member, torch.view_as_real(inc)).contiguous())
        if "phib" in self.cfg.fields:
            out["phib"] = state["phib"] + spec[:, -1] * self.unit_phib
        cost = 0.5 * torch.view_as_real(spec).square().flatten(1).sum(1)
        return out, cost

    def forward(self, state: Dict[str, Tensor], k: int) -> Dict[str, Tensor]:
        if not self.enabled:
            self.last_cost = None
            return state
        with torch.no_grad():  # C observes the state; gradients reach it through its action
            feats = self.features(state, k)
            x = self.fields(state, k)
        logits = self.net(feats).view(-1, self.n_bands, self.n_options)
        weights, choice = self._weights(logits)
        if choice is not None:
            weights = weights.detach()  # a choice carries no gradient (it is learned by RL)
        self.last_choice = choice
        if self.usage is not None:
            self.usage.append((choice if choice is not None else weights.argmax(-1))[0].detach())
        if self.record is not None and choice is not None:
            self.record.append((feats.detach(), choice.detach()))
        out, self.last_cost = self.apply_field(state, self.control_field(x, weights))
        return out

    def log_prob(self, feats: Tensor, choice: Tensor) -> Tensor:
        """Log-probability of the band choices [B] (summed over bands)."""
        logp = torch.log_softmax(self.net(feats).view(-1, self.n_bands, self.n_options), -1)
        return logp.gather(-1, choice[..., None])[..., 0].sum(1)

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        # neuralop models add a "_metadata" entry to every state dict; it holds no tensors
        state = {k: v for k, v in state.items() if not k.endswith("_metadata")}
        return super().load_state_dict(state, strict=strict, assign=assign)

    def choice_probs(self, state: Dict[str, Tensor], k: int) -> Tensor:
        """The policy's option probabilities [B, R, K + 1] (for usage statistics)."""
        with torch.no_grad():
            return torch.softmax(self.net(self.features(state, k)).view(-1, self.n_bands, self.n_options), -1)


def operator_state(ctrl: OperatorBankController) -> dict:
    return {"cfg": dataclasses.asdict(ctrl.cfg), "state": ctrl.state_dict()}


def load_operator_controller(model, saved: dict) -> OperatorBankController:
    ctrl = OperatorBankController(model, OperatorControlConfig(**saved["cfg"]))
    ctrl.load_state_dict(saved["state"])
    return ctrl.to(model.mean.device)
