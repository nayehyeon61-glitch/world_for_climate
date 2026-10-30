import math

import pytest
import torch

from climate_wm.geometry import composite_group as cg
from climate_wm.geometry.group_actions import so3_exp

torch.manual_seed(0)
B, S, N, CS, CV = 2, 3, 7, 6, 2


def _latent():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(B, S, N, CS, generator=g)
    v = torch.randn(B, S, N, CV, 3, generator=g)
    return h, v


def _fib(families=cg.FIB_FAMILIES, rank=3):
    cfg = cg.CompositeConfig(enabled=True, families=families, mixing_rank=rank)
    return cg.FibreGenerators(S, CS, CV, cfg)


def _coef(fib, g=None, scale=1.0):
    g = g or torch.Generator().manual_seed(2)
    return cg.FibreCoefficients(
        radiation=scale * torch.randn(B, S, N, generator=g),
        mixing=scale * torch.rand(B, N, fib.num_modes, generator=g),
        surface=scale * torch.randn(B, N, generator=g),
        condensation=scale * torch.rand(B, S, N, generator=g))


def _generator_matrix(fib, coef, b, n):
    """Dense A [S*CS, S*CS] and bias [S*CS] of column (b, n) of the scalar part."""
    q, lam = fib.modes, fib.mode_eig
    m = fib.mix_h @ fib.mix_h.T
    a = torch.zeros(S * CS, S * CS)
    for i in range(fib.num_modes):
        a -= coef.mixing[b, n, i] * torch.kron(torch.outer(q[:, i], q[:, i]) * lam[i], m)
    bias = fib.bias(coef)[b, :, n].reshape(-1)
    return a.detach(), bias.detach()


def test_zero_coefficients_and_zero_shift_are_the_identity():
    h, v = _latent()
    fib = _fib()
    zero = cg.FibreCoefficients(radiation=torch.zeros(B, S, N), mixing=torch.zeros(B, N, fib.num_modes),
                                surface=torch.zeros(B, N), condensation=torch.zeros(B, S, N))
    h2, v2 = fib.flow(h, v, zero, 1.0)
    assert torch.allclose(h2, h, atol=1e-6) and torch.allclose(v2, v, atol=1e-6)
    h3, v3 = cg.vert_shift(h, v, torch.zeros(B, N))
    assert torch.equal(h3, h) and torch.equal(v3, v)


def test_mixing_and_surface_conserve_the_column_and_reduce_vertical_variance():
    h, v = _latent()
    fib = _fib()
    coef = _coef(fib)
    coef = cg.FibreCoefficients(mixing=coef.mixing, surface=coef.surface)  # no sources
    h2, v2 = fib.flow(h, v, coef, 1.0)
    assert torch.allclose(h2.sum(1), h.sum(1), atol=1e-5)
    assert torch.allclose(v2.sum(1), v.sum(1), atol=1e-5)
    # mixing is down-gradient in the range of M: the projected vertical variance shrinks
    e, _ = fib._psd_eig(fib.mix_h)
    proj = lambda x: torch.einsum("bsnc,cj->bsnj", x, e)
    var = lambda x: proj(x).var(1).mean()
    coef_mix = cg.FibreCoefficients(mixing=coef.mixing)
    h3, _ = fib.flow(h, v, coef_mix, 1.0)
    assert var(h3) < var(h)


def test_exact_affine_flow_matches_ode_integration():
    h, v = _latent()
    fib = _fib()
    coef = _coef(fib)
    dt = 0.7
    h2, _ = fib.flow(h, v, coef, dt)
    for b, n in ((0, 0), (1, 3)):
        a, bias = _generator_matrix(fib, coef, b, n)
        z = h[b, :, n].reshape(-1).clone()
        steps = 4000
        for _ in range(steps):  # explicit Euler of dz/dt = A z + b
            z = z + (dt / steps) * (a @ z + bias)
        assert torch.allclose(h2[b, :, n].reshape(-1), z, atol=2e-3, rtol=1e-3)


def test_fibre_flow_is_a_one_parameter_subgroup():
    h, v = _latent()
    fib = _fib()
    coef = _coef(fib)
    h1, v1 = fib.flow(h, v, coef, 0.3)
    h12, v12 = fib.flow(h1, v1, coef, 0.5)
    h2, v2 = fib.flow(h, v, coef, 0.8)
    assert torch.allclose(h12, h2, atol=1e-5) and torch.allclose(v12, v2, atol=1e-5)


def test_mixing_family_commutes_and_brackets_stay_in_the_affine_algebra():
    fib = _fib()
    c1, c2 = _coef(fib, torch.Generator().manual_seed(3)), _coef(fib, torch.Generator().manual_seed(4))
    a1, b1 = _generator_matrix(fib, c1, 0, 0)
    a2, b2 = _generator_matrix(fib, c2, 0, 0)
    comm = a1 @ a2 - a2 @ a1
    assert comm.abs().max() < 1e-5 * max(a1.abs().max(), 1e-8)
    # [(A1,b1),(A2,b2)] = ([A1,A2], A1 b2 - A2 b1): a translation, so inside the algebra
    bracket_b = a1 @ b2 - a2 @ b1
    assert bracket_b.shape == b1.shape


def test_vert_shift_moves_shells_and_is_clamped():
    h, v = _latent()
    w = torch.ones(B, N)
    h2, v2 = cg.vert_shift(h, v, w, dt=1.0, max_shift=1e6)  # shift up by one shell (cap inactive)
    assert torch.allclose(h2[:, 1:], h[:, :-1], atol=1e-6)
    assert torch.allclose(h2[:, 0], h[:, 0], atol=1e-6)  # clamped at the surface
    assert torch.allclose(v2[:, 1:], v[:, :-1], atol=1e-6)
    big = cg.clamp_shift(torch.tensor([50.0]), 0.5)
    assert 0.49 < big.item() <= 0.5
    # differentiable in w
    w.requires_grad_(True)
    h3, _ = cg.vert_shift(h, v, 0.3 * w, dt=1.0, max_shift=0.5)
    h3.sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0


def test_splitting_orders_differ_by_second_order_and_strang_is_closer():
    h, v = _latent()
    fib = _fib()
    coef = _coef(fib, scale=0.5)
    w = 0.4 * torch.randn(B, N)
    rot_alg = 0.3 * torch.randn(B, S, N, 3)

    def rotate_factory(dt):
        def rotate(hh, vv):  # a cheap "rotation": rotate vector features, mix scalars by a rotation matrix
            r = so3_exp(dt * rot_alg)
            vv = torch.einsum("bsnij,bsncj->bsnci", r, vv)
            hh = hh + dt * torch.tanh(hh).roll(1, -1)
            return hh, vv
        return rotate

    shift = lambda hh, vv, dt: cg.vert_shift(hh, vv, w, dt, max_shift=2.0)
    fibre = lambda hh, vv, dt: fib.flow(hh, vv, coef, dt)

    def ref(dt, sub=64):
        hh, vv = h, v
        for _ in range(sub):
            hh, vv = cg.split_step(hh, vv, rotate_factory(dt / sub), shift, fibre, dt / sub, "strang")
        return hh

    errs = {}
    for dt in (0.4, 0.2):
        r = ref(dt)
        s_h, _ = cg.split_step(h, v, rotate_factory(dt), shift, fibre, dt, "strang")
        t_h, _ = cg.split_step(h, v, rotate_factory(dt), shift, fibre, dt, "trotter")
        tr_h, _ = cg.split_step(h, v, rotate_factory(dt), shift, fibre, dt, "trotter", reverse=True)
        errs[dt] = ((s_h - r).norm().item(), (t_h - r).norm().item(), (t_h - tr_h).norm().item())
    for dt in errs:
        assert errs[dt][0] < errs[dt][1]  # Strang closer than Trotter
    ratio = errs[0.4][2] / max(errs[0.2][2], 1e-12)  # order difference ~ dt^2
    assert 2.5 < ratio < 6.0


def test_cartan_split():
    centres = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    omega = torch.randn(4, 5, 3)
    k, p = cg.cartan_split(omega, centres)
    assert torch.allclose(k + p, omega, atol=1e-6)
    assert torch.allclose((p * centres).sum(-1), torch.zeros(4, 5), atol=1e-6)
    assert torch.allclose(torch.cross(k, centres.expand_as(k), dim=-1), torch.zeros_like(k), atol=1e-6)


def test_karcher_mean_and_log_cov_recover_a_wrapped_gaussian():
    g = torch.Generator().manual_seed(5)
    mu = torch.tensor([[0.4, -0.2, 0.7], [1.2, 0.3, -0.5]])
    sd = torch.tensor([0.05, 0.1, 0.02])
    noise = torch.randn(4000, 2, 3, generator=g) * sd
    # samples x_i = mu exp(noise_i)
    from climate_wm.geometry.group_actions import so3_log
    w = so3_log(so3_exp(mu)[None] @ so3_exp(noise))
    est = cg.karcher_mean(w)
    assert cg.geodesic_distance(est, mu).max() < 0.01
    cov = cg.log_cov(w, est)
    assert torch.allclose(cov.diagonal(dim1=-2, dim2=-1), sd.square().expand(2, 3), rtol=0.15, atol=1e-4)
    shift = cg.group_gaussian_shift(w[:2000], w[2000:])
    assert shift["mean_distance"] < 0.02 and shift["cov_change"] < 0.3


def test_coefficients_and_knockout():
    fib = _fib()
    raw = torch.randn(B, S, N, fib.num_raw)
    coef = fib.coefficients(raw, gate_mix=torch.full((B, N), 0.5), gate_cond=torch.zeros(B, S, N))
    assert (coef.mixing >= 0).all() and torch.allclose(coef.condensation, torch.zeros(B, S, N))
    assert coef.mixing.shape == (B, N, fib.num_modes) and coef.surface.shape == (B, N)
    ko = coef.knockout(["mixing", "radiation"])
    assert ko.mixing.abs().sum() == 0 and ko.radiation.abs().sum() == 0
    assert torch.equal(ko.surface, coef.surface)
    small = cg.FibreGenerators(1, CS, CV, cg.CompositeConfig(enabled=True))
    c1 = small.coefficients(torch.randn(B, 1, N, small.num_raw))
    assert c1.mixing is None and c1.surface is None
    assert math.isfinite(sum(v.item() for v in coef.norm2().values()))
