"""Composite Lie group G = (SO(3) x Vert) ⋉ Fib on a column latent (stage 5).

The stage 1-4 transition moves the latent with one generator per shell and
patch: an element of so(3), i.e. a horizontal rigid rotation. Everything the
atmosphere does besides horizontal transport (ascent, heating, mixing,
surface exchange, condensation) had to be absorbed by the residual. Stage 5
widens the group so that those processes are generators too:

  * SO(3)   horizontal transport, the existing Killing projection
            (`shells.SpherePatches`, `group_actions.SphereRotationAction`);
  * Vert    the translation group of the shell axis: a column is shifted by
            w shells (ascent / subsidence), `vert_shift`;
  * Fib     affine transformations of one column's latent Z in R^{S x C},
            dZ/dt = A Z + b, built from physics families (`FibreGenerators`):
              radiation     b = alpha_rad[s] u_rad[s]       (per-shell source)
              mixing        A = -sum_i alpha_i (q_i q_i^T ⊗ M)  (shell Laplacian
                            modes q_i tensor a low-rank PSD channel matrix M)
              surface       b = alpha_sfc (e_1 - e_0) ⊗ u_sfc (conserving exchange)
              condensation  b = alpha_cond[s] u_cond[s]     (gated source)
            The mixing generators share one eigenbasis, so the family is a
            commuting set and, together with the translations, a closed affine
            Lie subalgebra (the book's axiom 1: a subalgebra <-> a subgroup).
            Laplacian modes are orthogonal to the constant vector, so mixing
            conserves the column sum exactly; so does the surface exchange.

Transport acts on the base (sphere x shell axis) and Fib on the fibre, which is
the semidirect product: physics is applied where the parcel is. One step is an
operator splitting of the three exponentials (Strang by default), the same
physics-dynamics splitting numerical weather prediction uses:

    g_dt = exp(dt/2 F) exp(dt w d/d eta) exp(dt omega) exp(dt/2 F).

The affine flow of Fib is computed exactly in the eigenbasis,

    Z(dt) = e^{dt A} Z + phi_1(dt A) dt b,   phi_1(x) = (e^x - 1) / x,

so repeated flows compose exactly (a one-parameter subgroup). Also here:
the Cartan split of a patch rotation into spin about the patch normal (local
vorticity, k) and tangential transport (local advection, p), and the
intrinsic (Karcher) mean / log-covariance of rotation samples used for the
group-Gaussian statistics of the year-invariance test.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .group_actions import so3_exp, so3_log

FIB_FAMILIES = ("radiation", "mixing", "surface", "condensation")


@dataclasses.dataclass
class CompositeConfig:
    """Stage 5 options (model.composite in the config)."""
    enabled: bool = False
    vert: bool = True  # vertical shift generator
    families: Tuple[str, ...] = FIB_FAMILIES  # fibre families that are switched on
    mixing_rank: int = 4  # rank of the channel matrix M of the mixing generator
    max_shift: float = 0.5  # |dt w| cap in shells (CFL-like limiter)
    splitting: str = "strang"  # "strang" | "trotter"
    diag_gates: bool = True  # gate mixing / condensation by the diagnostics (needs model.diagnostics)
    coef_init: float = 0.1  # scale of the physics direction vectors at init

    def __post_init__(self):
        bad = set(self.families) - set(FIB_FAMILIES)
        if bad:
            raise ValueError(f"unknown fibre families {sorted(bad)}")
        if self.splitting not in ("strang", "trotter"):
            raise ValueError("splitting must be strang or trotter")
        self.families = tuple(self.families)


# ----------------------------------------------------------------- Vert
def clamp_shift(w: Tensor, max_shift: float) -> Tensor:
    """Smoothly limits a shift to |w| <= max_shift (shell units)."""
    return max_shift * torch.tanh(w / max_shift)


def vert_shift(h: Tensor, v: Tensor, w: Tensor, dt: float = 1.0,
               max_shift: float = 0.5) -> Tuple[Tensor, Tensor]:
    """Semi-Lagrangian shift of a column along the shell axis.

    h [B, S, N, C], v [B, S, N, Cv, 3], w [B, N] (shells per unit time, positive
    = upward = towards higher shell index). The departure coordinate of shell
    s is s - dt w; values are linearly interpolated and clamped at the ends
    (no inflow through the surface or the top). Exactly the identity at w = 0
    and differentiable in w.
    """
    b, s, n = h.shape[:3]
    if s == 1:
        return h, v
    d = clamp_shift(dt * w, max_shift)  # [B, N]
    src = torch.arange(s, device=h.device, dtype=h.dtype)[None, :, None] - d[:, None, :]
    src = src.clamp(0.0, float(s - 1))  # [B, S, N]
    lo = src.floor().long()
    hi = (lo + 1).clamp(max=s - 1)
    frac = (src - lo.to(src.dtype))[..., None]

    def gather(x: Tensor, idx: Tensor) -> Tensor:
        shape = idx.shape + (1,) * (x.ndim - 3)
        return torch.gather(x, 1, idx.reshape(shape).expand(-1, -1, -1, *x.shape[3:]))

    h_new = (1 - frac) * gather(h, lo) + frac * gather(h, hi)
    fv = frac[..., None]
    v_new = (1 - fv) * gather(v, lo) + fv * gather(v, hi)
    return h_new, v_new


# ------------------------------------------------------------------ Fib
@dataclasses.dataclass
class FibreCoefficients:
    """Generator coefficients of one step (already constrained: mixing and
    condensation non-negative). Any field may be None (= family off)."""
    radiation: Optional[Tensor] = None  # [B, S, N]
    mixing: Optional[Tensor] = None  # [B, N, I]   I = S - 1 Laplacian modes
    surface: Optional[Tensor] = None  # [B, N]
    condensation: Optional[Tensor] = None  # [B, S, N]

    def knockout(self, families: Sequence[str]) -> "FibreCoefficients":
        """Zeroes the given families (the knock-out intervention)."""
        kw = dataclasses.asdict(self)
        for f in families:
            if kw.get(f) is not None:
                kw[f] = torch.zeros_like(kw[f])
        return FibreCoefficients(**kw)

    def active(self) -> Dict[str, Tensor]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    def norm2(self) -> Dict[str, Tensor]:
        """Mean squared coefficient per family (for the generator penalty)."""
        return {k: v.square().mean() for k, v in self.active().items()}


def shell_laplacian_modes(num_shells: int) -> Tuple[Tensor, Tensor]:
    """Non-zero eigenpairs (q [S, S-1], lambda [S-1]) of the path-graph
    Laplacian of the shell axis; q is orthogonal to the constant vector."""
    s = num_shells
    lap = torch.zeros(s, s, dtype=torch.float64)
    for k in range(s - 1):
        lap[k, k] += 1
        lap[k + 1, k + 1] += 1
        lap[k, k + 1] -= 1
        lap[k + 1, k] -= 1
    lam, q = torch.linalg.eigh(lap)
    return q[:, 1:].float().contiguous(), lam[1:].float().clamp_min(0.0)


def _phi1m1(c: Tensor) -> Tensor:
    """phi_1(-c) - 1 = (1 - e^{-c}) / c - 1, stable near 0 (c >= 0)."""
    small = c < 1e-4
    safe = torch.where(small, torch.ones_like(c), c)
    return torch.where(small, -c / 2 + c * c / 6, -torch.expm1(-safe) / safe - 1)


class FibreGenerators(nn.Module):
    """Physics families as affine generators on a column latent."""

    def __init__(self, num_shells: int, cs: int, cv: int, cfg: CompositeConfig):
        super().__init__()
        self.cfg = cfg
        self.num_shells = num_shells
        q, lam = shell_laplacian_modes(num_shells)
        self.register_buffer("modes", q)  # [S, I]
        self.register_buffer("mode_eig", lam)  # [I]
        r = max(1, min(cfg.mixing_rank, cs))
        rv = max(1, min(cfg.mixing_rank, cv))
        g = torch.Generator().manual_seed(0)
        self.mix_h = nn.Parameter(torch.randn(cs, r, generator=g) / cs ** 0.5)
        self.mix_v = nn.Parameter(torch.randn(cv, rv, generator=g) / cv ** 0.5)
        init = cfg.coef_init
        self.u_rad = nn.Parameter(init * torch.randn(num_shells, cs, generator=g))
        self.u_cond = nn.Parameter(init * torch.randn(num_shells, cs, generator=g))
        self.u_sfc = nn.Parameter(init * torch.randn(cs, generator=g))

    # ------------------------------------------------------------ helpers
    @property
    def num_modes(self) -> int:
        return int(self.modes.shape[1])

    def has(self, family: str) -> bool:
        if family == "surface" and self.num_shells < 2:
            return False
        if family == "mixing" and self.num_shells < 2:
            return False
        return family in self.cfg.families

    @staticmethod
    def _psd_eig(w: Tensor) -> Tuple[Tensor, Tensor]:
        """M = W W^T: orthonormal eigenvectors E [C, r] and eigenvalues mu [r]."""
        gram = w.T @ w  # [r, r]
        mu, vecs = torch.linalg.eigh(gram)
        mu = mu.clamp_min(1e-8)
        e = (w @ vecs) / mu.sqrt()[None]
        return e, mu

    def bias(self, coef: FibreCoefficients) -> Optional[Tensor]:
        """Translation part b [B, S, N, Cs] of the generator (None if no source)."""
        parts = []
        if coef.radiation is not None and self.has("radiation"):
            parts.append(coef.radiation[..., None] * self.u_rad[None, :, None, :])
        if coef.condensation is not None and self.has("condensation"):
            parts.append(coef.condensation[..., None] * self.u_cond[None, :, None, :])
        if coef.surface is not None and self.has("surface"):
            f = coef.surface[:, None, :, None] * self.u_sfc[None, None, None, :]  # [B, 1, N, Cs]
            pad = [-f, f] + [torch.zeros_like(f)] * (self.num_shells - 2)
            parts.append(torch.cat(pad, 1))
        if not parts:
            return None
        return sum(parts)

    def _mix(self, x: Tensor, alpha: Tensor, dt: float, w: Tensor,
             bias: Optional[Tensor]) -> Tensor:
        """Exact affine flow of dZ/dt = -sum_i alpha_i (q_i q_i^T ⊗ M) Z + b.

        x [B, S, N, C, ...] (extra trailing dims are carried along), alpha
        [B, N, I] >= 0, w the factor of M = W W^T, bias [B, S, N, C] or None.
        """
        e, mu = self._psd_eig(w)  # [C, r], [r]
        q, lam = self.modes, self.mode_eig
        extra = x.shape[4:]
        xf = x.flatten(4) if extra else x[..., None]  # [B, S, N, C, K]
        c = dt * alpha[..., :, None] * lam[None, None, :, None] * mu[None, None, None, :]  # [B, N, I, r]
        proj = torch.einsum("si,bsnck,cj->bnijk", q, xf, e)
        decay = torch.expm1(-c)[..., None]  # e^{-c} - 1
        out = xf + torch.einsum("si,bnijk,cj->bsnck", q, decay * proj, e)
        if bias is not None:
            bf = bias[..., None]
            bproj = torch.einsum("si,bsnck,cj->bnijk", q, bf, e)
            out = out + dt * (bf + torch.einsum("si,bnijk,cj->bsnck", q,
                                                _phi1m1(c)[..., None] * bproj, e))
        return out.reshape(x.shape) if extra else out[..., 0]

    # -------------------------------------------------------------- flow
    def flow(self, h: Tensor, v: Tensor, coef: FibreCoefficients, dt: float
             ) -> Tuple[Tensor, Tensor]:
        """exp(dt F) applied to the column latent (h [B, S, N, Cs], v [B, S, N, Cv, 3])."""
        h = h.float()
        v = v.float()
        bias = self.bias(coef)
        mixing = coef.mixing if (coef.mixing is not None and self.has("mixing")) else None
        if mixing is None:
            if bias is not None:
                h = h + dt * bias
            return h, v
        h = self._mix(h, mixing, dt, self.mix_h, bias)
        v = self._mix(v, mixing, dt, self.mix_v, None)
        return h, v

    # ----------------------------------------------------- raw -> coefficients
    @property
    def num_raw(self) -> int:
        """Raw channels the generator head must output per (shell, node)."""
        return 1 + self.num_modes + 1 + 1  # radiation, mixing modes, surface, condensation

    def coefficients(self, raw: Tensor, gate_mix: Optional[Tensor] = None,
                     gate_cond: Optional[Tensor] = None) -> FibreCoefficients:
        """raw [B, S, N, num_raw] -> constrained coefficients.

        Per-column quantities (mixing, surface) are read from the shell mean of
        the raw head output; mixing and condensation are non-negative
        (softplus) and may be gated by diagnostics (gate_mix [B, N], gate_cond
        [B, S, N] in [0, 1]).
        """
        i = self.num_modes
        rad = raw[..., 0]
        col = raw.mean(1)  # [B, N, num_raw]
        mix = F.softplus(col[..., 1:1 + i])
        if gate_mix is not None:
            mix = mix * gate_mix[..., None]
        sfc = col[..., 1 + i]
        cond = F.softplus(raw[..., 2 + i])
        if gate_cond is not None:
            cond = cond * gate_cond
        return FibreCoefficients(
            radiation=rad if self.has("radiation") else None,
            mixing=mix if self.has("mixing") else None,
            surface=sfc if self.has("surface") else None,
            condensation=cond if self.has("condensation") else None)


# ----------------------------------------------------------- splitting
def split_step(h: Tensor, v: Tensor,
               rotate: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
               shift: Optional[Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]],
               fibre: Optional[Callable[[Tensor, Tensor, float], Tuple[Tensor, Tensor]]],
               dt: float, splitting: str = "strang", reverse: bool = False
               ) -> Tuple[Tensor, Tensor]:
    """One step of the composite action by operator splitting.

    `rotate(h, v)` applies exp(dt omega) (the SO(3) action already includes
    dt), `shift(h, v, dt)` the vertical shift and `fibre(h, v, dt)` the fibre
    flow. Strang: fibre half step, shift, rotation, fibre half step. Trotter:
    fibre, shift, rotation (or the reverse order with `reverse=True`, used to
    measure the splitting error).
    """
    fib = fibre or (lambda a, b, t: (a, b))
    sh = shift or (lambda a, b, t: (a, b))
    if splitting == "strang":
        h, v = fib(h, v, dt / 2)
        h, v = sh(h, v, dt)
        h, v = rotate(h, v)
        return fib(h, v, dt / 2)
    if splitting != "trotter":
        raise ValueError("splitting must be strang or trotter")
    if reverse:
        h, v = rotate(h, v)
        h, v = sh(h, v, dt)
        return fib(h, v, dt)
    h, v = fib(h, v, dt)
    h, v = sh(h, v, dt)
    return rotate(h, v)


# ---------------------------------------------------------------- Cartan
def cartan_split(omega: Tensor, centres: Tensor) -> Tuple[Tensor, Tensor]:
    """so(3) = k ⊕ p at a patch: spin about the patch normal (local vorticity)
    and the tangential part (local advection). omega [..., P, 3], centres [P, 3]."""
    n = centres.expand_as(omega)
    k = (omega * n).sum(-1, keepdim=True) * n
    return k, omega - k


# ------------------------------------------------ group-Gaussian statistics
def karcher_mean(w: Tensor, iters: int = 20, step: float = 1.0, tol: float = 1e-7) -> Tensor:
    """Intrinsic mean on SO(3) of algebra samples w [M, ..., 3] (x_i = exp(w_i)):
    mu <- mu exp(step * mean_i log(mu^-1 x_i)) (the book's Algorithm 9.3).
    Returns the mean as an algebra element [..., 3]."""
    x = so3_exp(w)  # [M, ..., 3, 3]
    mu = so3_exp(w.mean(0))
    for _ in range(iters):
        delta = so3_log(mu.transpose(-1, -2)[None] @ x).mean(0)  # [..., 3]
        mu = mu @ so3_exp(step * delta)
        if float(delta.norm(dim=-1).max()) < tol:
            break
    return so3_log(mu)


def log_cov(w: Tensor, mu: Tensor) -> Tensor:
    """Covariance on the algebra of samples w [M, ..., 3] around mu [..., 3]:
    the wrapped-Gaussian covariance of the book's eq. 9.15 -> [..., 3, 3]."""
    r_mu = so3_exp(mu)
    d = so3_log(r_mu.transpose(-1, -2)[None] @ so3_exp(w))  # [M, ..., 3]
    return torch.einsum("m...i,m...j->...ij", d, d) / max(w.shape[0], 1)


def geodesic_distance(a: Tensor, b: Tensor) -> Tensor:
    """||log(exp(a)^-1 exp(b))|| for algebra elements a, b [..., 3]."""
    return so3_log(so3_exp(a).transpose(-1, -2) @ so3_exp(b)).norm(dim=-1)


def group_gaussian_shift(w_a: Tensor, w_b: Tensor) -> Dict[str, Tensor]:
    """Distribution shift between two sample sets of rotations [M, ..., 3]:
    geodesic distance of the intrinsic means and the relative Frobenius
    change of the algebra covariance (both averaged over the trailing dims)."""
    mu_a, mu_b = karcher_mean(w_a), karcher_mean(w_b)
    cov_a, cov_b = log_cov(w_a, mu_a), log_cov(w_b, mu_b)
    scale = cov_a.flatten(-2).norm(dim=-1).clamp_min(1e-12)
    return {"mean_distance": geodesic_distance(mu_a, mu_b).mean(),
            "cov_change": ((cov_b - cov_a).flatten(-2).norm(dim=-1) / scale).mean()}
