"""Stage 9 arm N3: a World-Models controller on the frozen physics model.

As in Ha and Schmidhuber (2018), the world model (core + column physics, M)
does not change; a small controller C reads M's state and the game's mask and
acts on the state before every 6 h step. The action is a nudge of the state in
the lowest spherical harmonics (n <= max_degree) of temperature, vorticity and
humidity in a few layers, and of the lowest-level geopotential. One unit of
action is `action_scale` times the std of the field's 6 h change in that layer
(for vorticity: of the wind's, at degree 2), so the cost is dimensionless and
stratospheric humidity is not nudged by tropospheric amounts:

    x_{k+1} = M(x_k + B a_k),   cost = 1/2 |a_k|^2,

the controlled form of stage 8's transition cost (dX/dt = F + B a). Where and
how strongly C acts is a diagnosis of what the physics is missing.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# physical size of one unit of feature
FEATURE_SCALE = {"temp": 10.0, "zeta": 1e-4, "div": 1e-5, "q": 5e-3, "phib": 1000.0}
MASK_GROUPS = ("temperature", "geopotential", "u_component_of_wind", "v_component_of_wind",
               "vertical_velocity", "specific_humidity", "surface")


@dataclasses.dataclass
class TendencyControlConfig:
    max_degree: int = 4
    fields: Tuple[str, ...] = ("temp", "zeta", "q", "phib")
    # layers (hPa, inclusive) that share one nudge
    layers: Tuple[Tuple[int, int], ...] = ((850, 1000), (500, 700), (200, 400), (50, 150))
    hidden: int = 64
    action_scale: float = 0.2      # one action unit, in stds of the 6 h change
    use_state: bool = True        # False: open loop (mask and step only)
    control_cost: float = 1e-4    # weight of 1/2 |a|^2 in the training reward
    max_horizon: int = 8
    seed: int = 0
    # stage 10 (train/dreamer_game.MemoryController): "state" nudges the physical state as above;
    # "memory" acts only on the adaptive memory: per channel, low harmonics (n <= mem_degree) of a
    # write into m and of a shift of log tau for the next 6 h (the physics sees it through m)
    target: str = "state"
    mem_degree: int = 2
    mem_write_unit: float = 0.2   # change of m per action unit (m is O(1))
    mem_tau_unit: float = 0.5     # shift of ln tau per action unit

    def __post_init__(self):
        self.fields = tuple(self.fields)
        self.layers = tuple(tuple(l) for l in self.layers)
        if self.target not in ("state", "memory"):
            raise ValueError("control target must be state or memory")


class TendencyController(nn.Module):

    def __init__(self, model, cfg: TendencyControlConfig, diff_std: Optional[Dict[str, float]] = None):
        super().__init__()
        self.cfg = cfg
        sh = model.core.sh
        n = cfg.max_degree
        if n > sh.trunc:
            raise ValueError("max_degree exceeds the core truncation")
        levels = model.levels
        groups = []
        for lo, hi in cfg.layers:
            idx = [i for i, p in enumerate(levels) if lo <= p <= hi]
            if idx:
                groups.append(idx)
        self.groups = groups
        member = torch.zeros(len(groups), len(levels))
        for g, idx in enumerate(groups):
            member[g, idx] = 1.0
        self.register_buffer("member", member, persistent=False)  # layer -> level
        self.register_buffer("avg", member / member.sum(1, keepdim=True), persistent=False)
        # real coordinates of the complex coefficients with n <= max_degree
        coords = [(m, l, part) for m in range(n + 1) for l in range(m, n + 1)
                  for part in ((0,) if m == 0 else (0, 1))]
        self.coords = coords
        self.n_coef = len(coords)  # (n + 1)^2
        self.layer_fields = [f for f in cfg.fields if f != "phib"]
        self.n_action = self.n_coef * (len(self.layer_fields) * len(groups) + ("phib" in cfg.fields))
        self.state_fields = ("temp", "zeta", "div", "q")
        mask_dim = self.n_coef * len(MASK_GROUPS)
        state_dim = self.n_coef * (len(self.state_fields) * len(groups) + 1)
        self.in_dim = state_dim + mask_dim + cfg.max_horizon
        self.net = nn.Sequential(nn.Linear(self.in_dim, cfg.hidden), nn.SiLU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
                                 nn.Linear(cfg.hidden, self.n_action))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        t_ref = model.core.t_ref
        self.register_buffer("t_ref_layer", self.avg @ t_ref, persistent=False)
        k = model.tgt["z"][-1]
        self.register_buffer("phib_mean", model.mean[k].clone(), persistent=False)
        # physical size of one action unit per field and layer (saved with the
        # controller); without diff_std, a tenth of the model's normalisation std
        keys = [c.key for c in model.layout.targets]
        name = {"temp": "temperature", "zeta": "u_component_of_wind", "q": "specific_humidity"}

        def change_std(var, level=None):
            key = var if level is None else f"{var}@{level}"
            if diff_std is not None:
                return diff_std[key]
            return 0.1 * float(model.std[keys.index(key)])

        unit = torch.tensor([[np.mean([change_std(name[f], levels[i]) for i in idx]) for idx in groups]
                             for f in self.layer_fields], dtype=torch.float32)
        if "zeta" in self.layer_fields:  # wind change -> vorticity at degree 2
            unit[self.layer_fields.index("zeta")] *= np.sqrt(6.0) / model.core.sh.radius
        self.register_buffer("unit", cfg.action_scale * unit)
        self.register_buffer("unit_phib", torch.tensor(
            cfg.action_scale * change_std("geopotential", levels[-1]), dtype=torch.float32))
        dyn = [c for c in model.layout.dynamic]
        self.mask_index = [[i for i, c in enumerate(dyn)
                            if (c.name if c.level is not None else "surface") == g] for g in MASK_GROUPS]
        self.model_ref = [model]  # not a submodule: the world model's state dict stays unchanged
        # game state, set per episode (not parameters)
        self.mask: Optional[Tensor] = None
        self.mask_feats: Optional[Tensor] = None
        self.explore = 0.0
        self.record: Optional[list] = None
        self.enabled = True
        self.zero_diagnostics = False   # eval knock-out: hide the state from C
        self.random_actions = False     # eval control: the same actions on shuffled modes
        self.cost_weight = 0.0          # set by the game while training
        self.last_cost: Optional[Tensor] = None

    # ------------------------------------------------------------ features
    def _low(self, c: Tensor) -> Tensor:
        """Complex coefficients [..., M, N] -> real low-order vector [..., n_coef]."""
        r = torch.view_as_real(c)
        return torch.stack([r[..., m, l, part] for m, l, part in self.coords], -1)

    def _from_low(self, x: Tensor, like: Tensor, coords=None) -> Tensor:
        """Real vector [..., n_coef] -> complex coefficients shaped like `like` (zero elsewhere)."""
        r = torch.zeros(like.shape + (2,), dtype=x.dtype, device=x.device)
        for j, (m, l, part) in enumerate(coords or self.coords):
            r[..., m, l, part] = x[..., j]
        return torch.view_as_complex(r)

    def set_mask(self, mask: Optional[Tensor]) -> None:
        """Projects the game mask [B, C_dyn, H, W] on the low harmonics (once per episode)."""
        self.mask = mask
        if mask is None:
            self.mask_feats = None
            return
        sh = self.model_ref[0].core.sh
        per = torch.stack([mask[:, idx].float().mean(1) if idx else torch.zeros_like(mask[:, 0])
                           for idx in self.mask_index], 1)
        self.mask_feats = self._low(sh.from_data(per)).flatten(1)

    def features(self, state: Dict[str, Tensor], k: int) -> Tensor:
        b = state["temp"].shape[0]
        lev = lambda x: torch.view_as_complex(torch.einsum("gl,bl...c->bg...c", self.avg,
                                                           torch.view_as_real(x)).contiguous())
        parts = []
        for f in self.state_fields:
            low = self._low(lev(state[f])) / FEATURE_SCALE[f]  # [B, G, n_coef]
            if f == "temp":
                low[..., 0] = low[..., 0] - np.sqrt(2) * self.t_ref_layer / FEATURE_SCALE[f]
            parts.append(low.flatten(1))
        pb = self._low(state["phib"]) / FEATURE_SCALE["phib"]
        pb[..., 0] = pb[..., 0] - np.sqrt(2) * self.phib_mean / FEATURE_SCALE["phib"]
        parts.append(pb)
        x = torch.cat(parts, 1)
        if not self.cfg.use_state or self.zero_diagnostics:
            x = torch.zeros_like(x)
        m = self.mask_feats if self.mask_feats is not None else x.new_zeros(b, self.n_coef * len(MASK_GROUPS))
        step = x.new_zeros(b, self.cfg.max_horizon)
        step[:, min(k, self.cfg.max_horizon - 1)] = 1.0
        return torch.cat([x, m, step], 1)

    # -------------------------------------------------------------- action
    def apply(self, state: Dict[str, Tensor], action: Tensor) -> Dict[str, Tensor]:
        out = dict(state)
        n_g, n_c = len(self.groups), self.n_coef
        j = 0
        for i, f in enumerate(self.layer_fields):
            a = action[:, j:j + n_g * n_c].view(-1, n_g, n_c) * self.unit[i, :, None]
            j += n_g * n_c
            inc = self._from_low(a, state[f][:, :n_g])  # [B, G, M, N]
            per_level = torch.view_as_complex(torch.einsum("gl,bg...c->bl...c", self.member,
                                                           torch.view_as_real(inc)).contiguous())
            out[f] = state[f] + per_level
        if "phib" in self.cfg.fields:
            a = action[:, j:j + n_c] * self.unit_phib
            out["phib"] = state["phib"] + self._from_low(a, state["phib"])
        return out

    def forward(self, state: Dict[str, Tensor], k: int) -> Dict[str, Tensor]:
        if not self.enabled:
            self.last_cost = None
            return state
        with torch.no_grad():  # C observes the state; gradients reach it through its action
            feats = self.features(state, k)
        mu = self.net(feats)
        if self.random_actions:
            perm = torch.argsort(torch.rand(mu.shape, device=mu.device), 1)
            action = mu.detach().gather(1, perm)
        elif self.explore > 0:
            action = mu.detach() + self.explore * torch.randn_like(mu)
        else:
            action = mu
        if self.record is not None:
            self.record.append((feats.detach(), action.detach()))
        self.last_cost = 0.5 * action.square().sum(1)
        return self.apply(state, action)

    def log_prob(self, feats: Tensor, action: Tensor) -> Tensor:
        """Gaussian log-density per sample [B] (up to a constant), mean over action dims."""
        err = (action - self.net(feats)).square()
        return -err.mean(1) / (2 * self.explore ** 2)


def controller_state(controller: TendencyController) -> dict:
    return {"cfg": dataclasses.asdict(controller.cfg), "state": controller.state_dict()}


def load_controller(model, saved: dict) -> TendencyController:
    ctrl = TendencyController(model, TendencyControlConfig(**saved["cfg"]))
    ctrl.load_state_dict(saved["state"])
    return ctrl.to(model.mean.device)
