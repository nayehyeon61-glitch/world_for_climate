"""Spherical-harmonic transforms for the stage 9 spectral dynamical core.

Triangular truncation T (n <= T) with orthonormal associated Legendre
functions (no Condon-Shortley phase):

    f(lambda, mu) = sum_m sum_n f_nm Pbar_nm(mu) e^{i m lambda},   int_{-1}^{1} Pbar_nm^2 dmu = 1,

where the m > 0 terms also carry their complex conjugates (f is real).
Coefficients are complex tensors [..., M, N] with M = N = T + 1 (zero for
n < m). Fields on the Gaussian grid (Gauss-Legendre latitudes) are analysed
exactly; a regular latitude grid such as the WeatherBench2 one is analysed by
least squares per zonal wavenumber (cos-lat weighted) and synthesised by
evaluation. Vector fields are handled through U = u cos(lat), V = v cos(lat)
and H_nm = (1 - mu^2) dPbar_nm/dmu, as in Hoskins and Simmons (1975).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def legendre_tables(trunc: int, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pbar_nm(mu) and H_nm(mu) = (1 - mu^2) dPbar_nm/dmu, each [J, M, N] (float64)."""
    mu = np.asarray(mu, np.float64)
    nmax = trunc + 2  # H needs Pbar_{n+1}
    p = np.zeros((len(mu), nmax + 1, nmax + 1))
    s = np.sqrt(1 - mu ** 2)
    p[:, 0, 0] = 1 / np.sqrt(2)
    for m in range(1, nmax + 1):
        p[:, m, m] = np.sqrt((2 * m + 1) / (2 * m)) * s * p[:, m - 1, m - 1]
    eps = lambda n, m: np.sqrt((n * n - m * m) / (4 * n * n - 1))
    for m in range(nmax + 1):
        if m + 1 <= nmax:
            p[:, m, m + 1] = np.sqrt(2 * m + 3) * mu * p[:, m, m]
        for n in range(m + 2, nmax + 1):
            p[:, m, n] = (mu * p[:, m, n - 1] - eps(n - 1, m) * p[:, m, n - 2]) / eps(n, m)
    h = np.zeros_like(p)
    for m in range(nmax):
        for n in range(m, nmax):
            h[:, m, n] = -n * eps(n + 1, m) * p[:, m, n + 1]
            if n > m:
                h[:, m, n] += (n + 1) * eps(n, m) * p[:, m, n - 1]
    k = trunc + 1
    return p[:, :k, :k], h[:, :k, :k]


def _matmul_spec(table: Tensor, c: Tensor) -> Tensor:
    """sum_n table[j, m, n] c[..., m, n] -> [..., J, M] (complex)."""
    r = torch.view_as_real(c)  # [..., M, N, 2]
    out = torch.einsum("jmn,...mnc->...jmc", table, r)
    return torch.view_as_complex(out.contiguous())


def _project(table: Tensor, f: Tensor) -> Tensor:
    """sum_j table[j, m, n] f[..., j, m] -> [..., M, N] (complex)."""
    r = torch.view_as_real(f)  # [..., J, M, 2]
    out = torch.einsum("jmn,...jmc->...mnc", table, r)
    return torch.view_as_complex(out.contiguous())


class SphericalHarmonics(nn.Module):
    """Transforms between spectral coefficients, the Gaussian grid and a data grid.

    Grids are [..., lat, lon] with latitude ascending (south to north) and
    longitude starting at 0, matching the WeatherBench2 layout.
    """

    def __init__(self, trunc: int = 21, nlat: int = 32, nlon: int = 64,
                 radius: float = 6.371e6, data_lat: Optional[Sequence[float]] = None,
                 data_nlon: Optional[int] = None):
        super().__init__()
        self.trunc, self.nlat, self.nlon, self.radius = trunc, nlat, nlon, radius
        k = trunc + 1
        mu, w = np.polynomial.legendre.leggauss(nlat)  # ascending mu = south to north
        p, h = legendre_tables(trunc, mu)
        n = np.arange(k)
        m = np.arange(k)
        mask = (n[None, :] >= m[:, None]).astype(np.float64)
        f64 = lambda a: torch.as_tensor(a, dtype=torch.float32)
        self.register_buffer("mu", f64(mu), persistent=False)
        self.register_buffer("lat", f64(np.rad2deg(np.arcsin(mu))), persistent=False)
        self.register_buffer("coslat", f64(np.sqrt(1 - mu ** 2)), persistent=False)
        self.register_buffer("p", f64(p * mask), persistent=False)
        self.register_buffer("h", f64(h * mask), persistent=False)
        self.register_buffer("p_w", f64(p * mask * w[:, None, None]), persistent=False)
        self.register_buffer("p_wc", f64(p * mask * (w / (1 - mu ** 2))[:, None, None]), persistent=False)
        self.register_buffer("h_wc", f64(h * mask * (w / (1 - mu ** 2))[:, None, None]), persistent=False)
        self.register_buffer("mask", f64(mask), persistent=False)
        self.register_buffer("m", f64(m), persistent=False)
        eig = n * (n + 1) / radius ** 2
        self.register_buffer("eig", f64(eig))  # -laplacian eigenvalue per n
        inv = np.zeros(k)
        inv[1:] = 1 / eig[1:]
        self.register_buffer("inv_eig", f64(inv), persistent=False)
        self.data_lat, self.data_nlon = None, None
        if data_lat is not None:
            self.set_data_grid(data_lat, data_nlon or nlon)

    # ----------------------------------------------------------- data grid
    def set_data_grid(self, lat_deg: Sequence[float], nlon: int) -> None:
        lat = np.asarray(lat_deg, np.float64)
        mu = np.sin(np.deg2rad(lat))
        p, h = legendre_tables(self.trunc, mu)
        mask = self.mask.double().numpy()
        p, h = p * mask, h * mask
        w = np.sqrt(np.cos(np.deg2rad(lat)))  # cos-lat weighted least squares
        k = self.trunc + 1
        ana = np.zeros((len(lat), k, k))  # [J, M, N]: c[m, n] = sum_j ana[j, m, n] F[j, m]
        for m in range(k):
            a = p[:, m, m:] * w[:, None]  # [J, N - m]
            ana[:, m, m:] = (np.linalg.pinv(a) * w[None, :]).T
        # winds: U = u cos(lat) needs degree T + 1, fitted on the data grid and
        # evaluated on the Gaussian grid
        pv, _ = legendre_tables(self.trunc + 1, mu)
        kv = k + 1
        maskv = (np.arange(kv)[None, :] >= np.arange(kv)[:, None]).astype(np.float64)
        pv = pv * maskv
        ana_v = np.zeros((len(lat), kv, kv))
        for m in range(kv):
            a = pv[:, m, m:] * w[:, None]
            ana_v[:, m, m:] = (np.linalg.pinv(a) * w[None, :]).T
        pg, _ = legendre_tables(self.trunc + 1, self.mu.double().numpy())
        f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.mu.device)
        self.register_buffer("data_ana_v", f32(ana_v), persistent=False)
        self.register_buffer("gauss_p_v", f32(pg * maskv), persistent=False)
        self.data_lat, self.data_nlon = lat, nlon
        self.register_buffer("data_p", f32(p), persistent=False)
        self.register_buffer("data_h", f32(h), persistent=False)
        self.register_buffer("data_ana", f32(ana), persistent=False)
        self.register_buffer("data_coslat", f32(np.cos(np.deg2rad(lat))), persistent=False)

    # ------------------------------------------------------------- helpers
    def _fourier(self, f: Tensor) -> Tensor:
        return torch.fft.rfft(f, norm="forward")[..., : self.trunc + 1]

    def _grid(self, fm: Tensor, data: bool = False) -> Tensor:
        return torch.fft.irfft(fm, n=self.data_nlon if data else self.nlon, norm="forward")

    def ddlambda(self, c: Tensor) -> Tensor:
        return c * (1j * self.m)[:, None]

    def laplacian(self, c: Tensor) -> Tensor:
        return -c * self.eig

    def inverse_laplacian(self, c: Tensor) -> Tensor:
        return -c * self.inv_eig

    def truncate(self, c: Tensor) -> Tensor:
        return c * self.mask

    # --------------------------------------------------- scalar transforms
    def to_grid(self, c: Tensor, data: bool = False) -> Tensor:
        return self._grid(_matmul_spec(self.data_p if data else self.p, c), data)

    def to_spec(self, f: Tensor) -> Tensor:
        return _project(self.p_w, self._fourier(f))

    def from_data(self, f: Tensor) -> Tensor:
        """Least-squares analysis of a field on the data grid."""
        return _project(self.data_ana, self._fourier(f))

    def grad(self, c: Tensor, data: bool = False) -> Tuple[Tensor, Tensor]:
        """Grid gradient (1/(a cos) d/dlambda, (1/a) d/dphi) of a scalar."""
        p, h, cos = ((self.data_p, self.data_h, self.data_coslat.clamp(min=1e-3)) if data
                     else (self.p, self.h, self.coslat))
        scale = 1 / (self.radius * cos[:, None])
        gx = self._grid(_matmul_spec(p, self.ddlambda(c)), data) * scale
        gy = self._grid(_matmul_spec(h, c), data) * scale
        return gx, gy

    # --------------------------------------------------- vector transforms
    def uv(self, zeta: Tensor, div: Tensor, data: bool = False) -> Tuple[Tensor, Tensor]:
        """Grid winds (u, v) from spectral vorticity and divergence."""
        p, h, cos = ((self.data_p, self.data_h, self.data_coslat.clamp(min=1e-3)) if data
                     else (self.p, self.h, self.coslat))  # pole rows: U = V = 0 there
        psi, chi = self.inverse_laplacian(zeta), self.inverse_laplacian(div)
        a = self.radius
        ucos = (-_matmul_spec(h, psi) + _matmul_spec(p, self.ddlambda(chi))) / a
        vcos = (_matmul_spec(p, self.ddlambda(psi)) + _matmul_spec(h, chi)) / a
        inv = 1 / cos[:, None]
        return self._grid(ucos, data) * inv, self._grid(vcos, data) * inv

    def div_curl(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Spectral divergence and vorticity of a Gaussian-grid vector field."""
        um = self._fourier(u * self.coslat[:, None])
        vm = self._fourier(v * self.coslat[:, None])
        im = 1j * self.m
        a = self.radius
        div = (_project(self.p_wc, um * im) - _project(self.h_wc, vm)) / a
        curl = (_project(self.p_wc, vm * im) + _project(self.h_wc, um)) / a
        return self.truncate(div), self.truncate(curl)

    def div_curl_from_data(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Vorticity and divergence of data-grid winds: U, V are fitted as scalars
        on the data grid, evaluated on the Gaussian grid and differentiated there."""
        cos_d, cos_g = self.data_coslat[:, None], self.coslat[:, None]
        kv = self.trunc + 2

        def regrid(f):
            fm = torch.fft.rfft(f * cos_d, norm="forward")[..., :kv]
            c = _project(self.data_ana_v, fm)
            return self._grid(_matmul_spec(self.gauss_p_v, c)) / cos_g

        return self.div_curl(regrid(u), regrid(v))
