"""Stage 9: differentiable spectral primitive-equation core on pressure levels.

Prognostic spectral fields on the data's pressure levels (top to bottom):
vorticity zeta, divergence D, temperature T, specific humidity q, plus the
geopotential of the lowest level Phi_b. Diagnostic: winds from (zeta, D),
vertical velocity omega from continuity, geopotential from the hydrostatic
relation (plus the fixed hydrostatic residual of the initial state).

    d zeta/dt = curl A,     d D/dt = div A - lap(K + Phi),
    A = ((zeta + f) v - omega du/dp + F_u,  -(zeta + f) u - omega dv/dp + F_v),
    dT/dt = -V.grad T + omega (kappa T / p - dT/dp) + Q,
    dq/dt = -V.grad q - omega dq/dp + S,
    dPhi_b/dt = -V_b.grad Phi_b + (R T_v,b / p_b) omega_b + P_b,
    omega(p) = -int_0^p D dp',   Phi(p) = Phi_b + int_p^{p_b} R T_v dln p'.

Gravity waves (the D - Phi - T coupling linearised about a reference
temperature profile) are treated implicitly with the IMEX Runge-Kutta scheme
SIL3 of Whitaker and Kar (2013), so a 20-minute step is stable at T31. F, Q, S
and P_b are the tendencies of the learned column physics (zero in arm N0).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..geometry.spectral import SphericalHarmonics

RADIUS = 6.371e6
OMEGA = 7.292e-5
R_DRY = 287.05
CP = 1004.6
KAPPA = R_DRY / CP
EPS_V = 0.608  # T_v = T (1 + EPS_V q)

FIELDS = ("zeta", "div", "temp", "q", "phib")
TERMS = ("advection", "coriolis", "pgf", "adiabatic", "vertical", "moisture")

# IMEX SIL3 (Whitaker and Kar 2013): explicit and implicit tableaus (4 stages).
SIL3_EX = ((0, 0, 0, 0), (1 / 3, 0, 0, 0), (1 / 6, 1 / 2, 0, 0), (1 / 2, -1 / 2, 1, 0))
SIL3_IM = ((0, 0, 0, 0), (1 / 6, 1 / 6, 0, 0), (1 / 3, 0, 1 / 3, 0), (3 / 8, 0, 3 / 8, 1 / 4))
SIL3_B_EX = (1 / 2, -1 / 2, 1, 0)
SIL3_B_IM = (3 / 8, 0, 3 / 8, 1 / 4)

State = Dict[str, Tensor]


@dataclasses.dataclass
class CoreConfig:
    trunc: int = 31
    nlat: int = 48                   # quadratic Gaussian grid for T31
    nlon: int = 96
    dt_minutes: float = 20.0
    diffusion_order: int = 4         # del^(2 order) hyperdiffusion
    diffusion_hours: float = 3.0     # e-folding time at n = trunc
    divergence_damping_hours: float = 0.0  # extra del^2 damping of D at n = trunc (0: off)
    surface_mass: bool = True        # column mass convergence moves Phi_b
    # Rayleigh friction of the boundary layer (Held and Suarez 1994): rate
    # 1/surface_drag_days at the lowest level, linear in p/p_bottom down to 0 at 0.7
    surface_drag_days: float = 0.0
    terms: Tuple[str, ...] = TERMS   # knock-outs for the cause analysis

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CoreConfig":
        d = dict(d or {})
        if "terms" in d:
            d["terms"] = tuple(d["terms"])
        return cls(**d)


def vertical_operators(levels_hpa: Sequence[float]):
    """W (omega = -W D), Hy (Phi = Phi_b + Hy T_v) and Dp (d/dp), each [L, L] (float64)."""
    p = np.asarray(levels_hpa, np.float64) * 100.0
    n = len(p)
    if np.any(np.diff(p) <= 0):
        raise ValueError("levels must be ordered top (low pressure) to bottom")
    half = np.concatenate([[0.0], 0.5 * (p[1:] + p[:-1])])  # interface above each level
    w = np.zeros((n, n))
    for k in range(n):
        w[k, :k] = np.diff(np.append(half[:k], half[k])) if k else 0.0
        w[k, k] = p[k] - half[k]
    hy = np.zeros((n, n))
    for k in range(n - 1):
        for i in range(k, n - 1):
            dl = 0.5 * R_DRY * math.log(p[i + 1] / p[i])
            hy[k, i] += dl
            hy[k, i + 1] += dl
    dp = np.zeros((n, n))
    for k in range(n):
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        dp[k, b] += 1 / (p[b] - p[a])
        dp[k, a] -= 1 / (p[b] - p[a])
    return p, w, hy, dp


class PrimitiveCore(nn.Module):
    """Spectral primitive equations on pressure levels with semi-implicit gravity waves."""

    def __init__(self, cfg: CoreConfig, levels_hpa: Sequence[float],
                 t_ref: Sequence[float], data_lat: Optional[Sequence[float]] = None,
                 data_nlon: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.sh = SphericalHarmonics(cfg.trunc, cfg.nlat, cfg.nlon, RADIUS, data_lat, data_nlon)
        p, w, hy, dp = vertical_operators(levels_hpa)
        t_ref = np.asarray(t_ref, np.float64)
        stab = KAPPA * t_ref / p - dp @ t_ref  # static stability S = kappa T/p - dT/dp
        if np.any(stab <= 0):
            raise ValueError("reference profile must be statically stable")
        if "adiabatic" not in cfg.terms:
            stab = 0 * stab  # the knocked-out term leaves the linear part too
        n_lev = len(p)
        cb = R_DRY * t_ref[-1] / p[-1] if cfg.surface_mass else 0.0
        # Phi_lin = Phi_b + Hy T; d(Phi_lin)/dt|_lin = -G D
        g = cb * np.outer(np.ones(n_lev), w[-1]) + hy @ (stab[:, None] * w)
        self.dt = cfg.dt_minutes * 60.0
        eig = self.sh.eig.double().numpy()
        etas = sorted({SIL3_IM[i][i] for i in range(1, 4)})
        inv = np.stack([np.linalg.inv(np.eye(n_lev)[None] + (self.dt * a) ** 2 * eig[:, None, None] * g[None])
                        for a in etas])  # [E, N, L, L]
        self._eta_index = {a: i for i, a in enumerate(etas)}
        # derived from the config, not learned: rebuilt on construction (knock-outs)
        buf = lambda name, a: self.register_buffer(name, torch.as_tensor(a, dtype=torch.float32),
                                                   persistent=False)
        buf("p", p)
        buf("w", w)
        buf("hy", hy)
        buf("dp", dp)
        buf("stab", stab)
        buf("t_ref", t_ref)
        buf("inv", inv)
        self.cb = cb
        buf("f", 2 * OMEGA * self.sh.mu.double().numpy())
        n = np.arange(cfg.trunc + 1)
        lam = (n * (n + 1)) / (cfg.trunc * (cfg.trunc + 1))
        diff = self.dt / (cfg.diffusion_hours * 3600.0) * lam ** cfg.diffusion_order
        buf("diffusion", np.exp(-diff))
        ddamp = (self.dt / (cfg.divergence_damping_hours * 3600.0) * lam
                 if cfg.divergence_damping_hours > 0 else 0 * lam)
        buf("div_damping", np.exp(-ddamp))
        sigma = p / p[-1]
        drag = (np.clip((sigma - 0.7) / 0.3, 0, None) / (cfg.surface_drag_days * 86400.0)
                if cfg.surface_drag_days > 0 else 0 * sigma)
        buf("drag", drag)

    # ----------------------------------------------------------- helpers
    def _lev(self, mat: Tensor, x: Tensor) -> Tensor:
        """Applies a level matrix to [..., L, *spatial] (real or complex)."""
        if torch.is_complex(x):
            return torch.view_as_complex(torch.einsum("kj,bj...c->bk...c", mat, torch.view_as_real(x)).contiguous())
        return torch.einsum("kj,bj...->bk...", mat, x)

    def geopotential(self, state: State, tq: Optional[Tensor] = None) -> Tensor:
        """Spectral Phi [B, L, M, N] (incl. the virtual-temperature part when tq = to_spec(T q))."""
        tv = state["temp"] if tq is None else state["temp"] + EPS_V * tq
        phi = state["phib"][:, None] + self._lev(self.hy, tv)
        return phi + state["resid"] if "resid" in state else phi

    def omega_spec(self, div: Tensor) -> Tensor:
        return -self._lev(self.w, div)

    # --------------------------------------------------------- tendencies
    def tendencies(self, state: State, forcing: Optional[Dict[str, Tensor]] = None) -> State:
        """Full tendencies (spectral). `forcing` holds grid tendencies F_u, F_v, Q, S, P_b."""
        sh, on = self.sh, set(self.cfg.terms)
        zeta, div, temp, q, phib = (state[k] for k in FIELDS)
        u, v = sh.uv(zeta, div)
        om = sh.to_grid(self.omega_spec(div))
        tg, qg = sh.to_grid(temp), sh.to_grid(q)
        zg = sh.to_grid(zeta)
        fz = (self.f[:, None] if "coriolis" in on else 0.0) + (zg if "advection" in on else 0.0)
        vert = "vertical" in on
        au = fz * v
        av = -fz * u
        if vert:
            au = au - om * self._lev(self.dp, u)
            av = av - om * self._lev(self.dp, v)
        if self.cfg.surface_drag_days > 0:
            au = au - self.drag[:, None, None] * u
            av = av - self.drag[:, None, None] * v
        if forcing is not None:
            au = au + forcing.get("F_u", 0.0)
            av = av + forcing.get("F_v", 0.0)
        dd, cc = sh.div_curl(au, av)
        tq = sh.to_spec(tg * qg)
        grad_terms = 0.0
        if "pgf" in on:
            grad_terms = self.geopotential(state, tq)
        if "advection" in on:
            grad_terms = grad_terms + sh.to_spec(0.5 * (u * u + v * v))
        d_div = dd - sh.laplacian(grad_terms) if torch.is_tensor(grad_terms) else dd
        # temperature
        tx, ty = sh.grad(temp)
        dtg = 0.0 * tg
        if "advection" in on:
            dtg = dtg - (u * tx + v * ty)
        if vert:
            dtg = dtg - om * self._lev(self.dp, tg)
        if "adiabatic" in on:
            dtg = dtg + KAPPA * tg * om / self.p[:, None, None]
        if forcing is not None and "Q" in forcing:
            dtg = dtg + forcing["Q"]
        # moisture
        dqg = 0.0 * qg
        if "moisture" in on:
            qx, qy = sh.grad(q)
            dqg = dqg - (u * qx + v * qy)
            if vert:
                dqg = dqg - om * self._lev(self.dp, qg)
        if forcing is not None and "S" in forcing:
            dqg = dqg + forcing["S"]
        # lowest-level geopotential
        bx, by = sh.grad(phib)
        dbg = 0.0 * bx
        if "advection" in on:
            dbg = dbg - (u[:, -1] * bx + v[:, -1] * by)
        if "pgf" in on and self.cfg.surface_mass:  # column mass convergence raises Phi_b
            dbg = dbg + R_DRY * tg[:, -1] * (1 + EPS_V * qg[:, -1]) / self.p[-1] * om[:, -1]
        if forcing is not None and "P_b" in forcing:
            dbg = dbg + forcing["P_b"]
        return {"zeta": cc, "div": d_div, "temp": sh.to_spec(dtg), "q": sh.to_spec(dqg),
                "phib": sh.to_spec(dbg)}

    def linear(self, state: State) -> State:
        """Linear gravity-wave part (about the reference profile)."""
        on = set(self.cfg.terms)
        zero = torch.zeros_like
        out = {k: zero(state[k]) for k in FIELDS}
        if "pgf" in on:
            phi = state["phib"][:, None] + self._lev(self.hy, state["temp"])
            out["div"] = -self.sh.laplacian(phi)
            wd = self._lev(self.w, state["div"])
            out["temp"] = -self.stab[None, :, None, None] * wd
            out["phib"] = -self.cb * wd[:, -1]
        return out

    def implicit_solve(self, rhs: State, a: float) -> State:
        """Solves (I - dt a L) Y = rhs."""
        if a == 0 or "pgf" not in self.cfg.terms:
            return rhs
        eta = self.dt * a
        eig = self.sh.eig
        phi_r = rhs["phib"][:, None] + self._lev(self.hy, rhs["temp"])
        b = rhs["div"] + eta * eig * phi_r
        inv = self.inv[self._eta_index[a]]  # [N, L, L]
        br = torch.view_as_real(b)  # [B, L, M, N, 2]
        d_new = torch.view_as_complex(torch.einsum("nkj,bjmnc->bkmnc", inv, br).contiguous())
        wd = self._lev(self.w, d_new)
        out = dict(rhs)
        out["div"] = d_new
        out["temp"] = rhs["temp"] - eta * self.stab[None, :, None, None] * wd
        out["phib"] = rhs["phib"] - eta * self.cb * wd[:, -1]
        return out

    # ----------------------------------------------------------- stepping
    def step(self, state: State, forcing: Optional[Dict[str, Tensor]] = None) -> State:
        """One IMEX SIL3 step of length dt, then hyperdiffusion."""
        dt = self.dt
        fixed = {k: v for k, v in state.items() if k not in FIELDS}
        y0 = {k: state[k] for k in FIELDS}
        ys, nl, ln = [], [], []
        for i in range(4):
            rhs = dict(y0)
            for j in range(i):
                for k in FIELDS:
                    if SIL3_EX[i][j]:
                        rhs[k] = rhs[k] + dt * SIL3_EX[i][j] * nl[j][k]
                    if SIL3_IM[i][j]:
                        rhs[k] = rhs[k] + dt * SIL3_IM[i][j] * ln[j][k]
            y = self.implicit_solve(rhs, SIL3_IM[i][i])
            full = self.tendencies({**y, **fixed}, forcing)
            lin = self.linear(y)
            ys.append(y)
            ln.append(lin)
            nl.append({k: full[k] - lin[k] for k in FIELDS})
        out = dict(y0)
        for j in range(4):
            for k in FIELDS:
                if SIL3_B_EX[j]:
                    out[k] = out[k] + dt * SIL3_B_EX[j] * nl[j][k]
                if SIL3_B_IM[j]:
                    out[k] = out[k] + dt * SIL3_B_IM[j] * ln[j][k]
        for k in FIELDS:
            out[k] = self.sh.truncate(out[k] * self.diffusion)
        out["div"] = out["div"] * self.div_damping
        out.update(fixed)
        return out

    # ------------------------------------------------------- data <-> state
    def state_from_grid(self, u: Tensor, v: Tensor, temp: Tensor, q: Tensor, z: Tensor) -> State:
        """Spectral state from data-grid fields [B, L, H, W] (physical units)."""
        sh = self.sh
        div, zeta = sh.div_curl_from_data(u, v)
        t_s, q_s, z_s = sh.from_data(temp), sh.from_data(q), sh.from_data(z)
        state = {"zeta": zeta, "div": div, "temp": t_s, "q": q_s, "phib": z_s[:, -1]}
        tq = sh.from_data(temp * q)
        state["resid"] = z_s - self.geopotential(state, tq)
        return state

    def grid_fields(self, state: State) -> Dict[str, Tensor]:
        """Data-grid diagnostics [B, L, H, W]: u, v, temp, q, z, omega."""
        sh = self.sh
        u, v = sh.uv(state["zeta"], state["div"], data=True)
        temp = sh.to_grid(state["temp"], data=True)
        q = sh.to_grid(state["q"], data=True)
        tq = sh.from_data(temp * q)
        z = sh.to_grid(self.geopotential(state, tq), data=True)
        om = sh.to_grid(self.omega_spec(state["div"]), data=True)
        return {"u": u, "v": v, "temp": temp, "q": q, "z": z, "omega": om}
