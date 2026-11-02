import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.geometry.spectral import SphericalHarmonics, legendre_tables
from climate_wm.models import primitive_core as pc
from climate_wm.models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel
from climate_wm.train.losses import GraphCastLoss

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
T_REF = np.array([212, 208, 212, 218, 226, 234, 248, 258, 266, 273, 281, 285, 288.0])


def _random_spec(sh, *shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    k = sh.trunc + 1
    c = torch.complex(torch.randn(*shape, k, k, generator=gen), torch.randn(*shape, k, k, generator=gen))
    c = c * sh.mask
    c[..., 0, :] = c[..., 0, :].real  # m = 0 coefficients of a real field are real
    return c


def test_legendre_tables():
    mu, w = np.polynomial.legendre.leggauss(24)
    p, h = legendre_tables(10, mu)
    gram = np.einsum("jn,jk,j->nk", p[:, 3, 3:], p[:, 3, 3:], w)
    assert np.allclose(gram, np.eye(8), atol=1e-12)
    d = 1e-6
    fd = (legendre_tables(10, mu + d)[0] - legendre_tables(10, mu - d)[0]) / (2 * d)
    assert np.allclose(h, (1 - mu ** 2)[:, None, None] * fd, atol=1e-6)


@pytest.mark.parametrize("data_lat", [-87.1875 + 5.625 * np.arange(32), np.linspace(-90, 90, 33)])
def test_spectral_transforms(data_lat):
    sh = SphericalHarmonics(15, 24, 48, data_lat=data_lat, data_nlon=64)
    c = _random_spec(sh, 2)
    assert torch.allclose(sh.to_spec(sh.to_grid(c)), c, atol=1e-5)
    assert torch.allclose(sh.from_data(sh.to_grid(c, data=True)), c, atol=1e-4)
    zeta, div = sh.laplacian(_random_spec(sh, 2, seed=1)), sh.laplacian(_random_spec(sh, 2, seed=2))
    d2, z2 = sh.div_curl(*sh.uv(zeta, div))
    assert torch.allclose(d2, div, atol=1e-5 * div.abs().max()) and torch.allclose(z2, zeta, atol=1e-5 * zeta.abs().max())
    u, v = sh.uv(zeta, div, data=True)
    d3, z3 = sh.div_curl_from_data(u, v)
    assert torch.isfinite(u).all()
    assert torch.allclose(d3, div, atol=1e-4 * div.abs().max()) and torch.allclose(z3, zeta, atol=1e-4 * zeta.abs().max())
    # gradient of a potential = the divergent wind of that potential
    chi = _random_spec(sh, 2, seed=3)
    gx, gy = sh.grad(chi)
    u2, v2 = sh.uv(torch.zeros_like(chi), sh.laplacian(chi))
    assert torch.allclose(gx, u2, atol=1e-5 * gx.abs().max()) and torch.allclose(gy, v2, atol=1e-5 * gy.abs().max())


def test_vertical_operators():
    p, w, hy, dp = pc.vertical_operators(LEVELS)
    d = np.ones(len(p))
    assert np.allclose(-(w @ d), -p)  # omega = -p D for uniform divergence
    t = np.full(len(p), 250.0)
    phi = hy @ t  # isothermal: Phi = R T ln(p_b / p)
    assert np.allclose(phi, pc.R_DRY * 250.0 * np.log(p[-1] / p), rtol=1e-12)
    assert np.allclose(dp @ p, 1.0)


def test_imex_tableau_second_order():
    """SIL3 on dy/dt = i a y (explicit) + b y (implicit, stiff decay)."""
    a, b = 1.0, -20.0

    def solve(dt, t_end=1.0):
        y = 1.0 + 0j
        for _ in range(int(round(t_end / dt))):
            ys, ex, im = [], [], []
            for i in range(4):
                rhs = y + dt * sum(pc.SIL3_EX[i][j] * ex[j] + pc.SIL3_IM[i][j] * im[j] for j in range(i))
                yi = rhs / (1 - dt * pc.SIL3_IM[i][i] * b)
                ys.append(yi)
                ex.append(1j * a * yi)
                im.append(b * yi)
            y = y + dt * sum(pc.SIL3_B_EX[j] * ex[j] + pc.SIL3_B_IM[j] * im[j] for j in range(4))
        return y

    exact = np.exp((1j * a + b) * 1.0)
    e1, e2 = abs(solve(0.02) - exact), abs(solve(0.01) - exact)
    assert e2 < e1 / 3.5  # at least second order
    assert abs(solve(0.5, 10.0)) < 1  # stable with a stiff implicit part


@pytest.fixture(scope="module")
def core():
    return pc.PrimitiveCore(pc.CoreConfig(trunc=21, nlat=32, nlon=64, dt_minutes=30.0), LEVELS, T_REF)


def test_core_keeps_balanced_solid_body_rotation(core):
    sh = core.sh
    u0 = 20.0
    lat = np.deg2rad(sh.lat.double().numpy())
    ug = torch.tensor(u0 * np.cos(lat), dtype=torch.float32)[:, None].expand(32, 64)
    ug = ug[None, None].expand(1, len(LEVELS), 32, 64).contiguous()
    div, zeta = sh.div_curl(ug, torch.zeros_like(ug))
    temp = sh.to_spec(torch.tensor(T_REF, dtype=torch.float32)[None, :, None, None].expand(1, len(LEVELS), 32, 64).contiguous())
    phib = 1000.0 - (pc.RADIUS * pc.OMEGA * u0 + u0 ** 2 / 2) * np.sin(lat) ** 2
    phib = sh.to_spec(torch.tensor(phib, dtype=torch.float32)[:, None].expand(32, 64)[None].contiguous())
    s0 = {"zeta": zeta, "div": div, "temp": temp, "q": torch.zeros_like(temp), "phib": phib}
    st = dict(s0)
    for _ in range(48):  # one day
        st = core.step(st)
    for k in ("zeta", "temp", "phib"):
        assert (st[k] - s0[k]).abs().max() < 1e-4 * s0[k].abs().max(), k
    assert sh.uv(st["zeta"], st["div"])[1].abs().max() < 1e-2  # no meridional wind appears


def test_core_gravity_waves_stay_bounded(core):
    sh = core.sh
    temp = sh.to_spec(torch.tensor(T_REF, dtype=torch.float32)[None, :, None, None].expand(1, len(LEVELS), 32, 64).contiguous())
    blob = torch.zeros(1, len(LEVELS), 32, 64)
    blob[:, 6:11, 12:18, 25:32] = 3.0
    st = {"zeta": torch.zeros_like(temp), "div": torch.zeros_like(temp), "temp": temp + sh.to_spec(blob),
          "q": torch.zeros_like(temp), "phib": torch.zeros_like(temp[:, 0])}
    for _ in range(96):  # two days at 30 min, far beyond the explicit gravity-wave limit
        st = core.step(st)
    u, v = sh.uv(st["zeta"], st["div"])
    assert torch.isfinite(u).all() and u.abs().max() < 30 and v.abs().max() < 30


def test_boundary_layer_drag():
    c = pc.PrimitiveCore(pc.CoreConfig(trunc=10, nlat=16, nlon=32, surface_drag_days=1.0,
                                       terms=("pgf",)), LEVELS, T_REF)
    sigma = np.array(LEVELS) / LEVELS[-1]
    expect = np.clip((sigma - 0.7) / 0.3, 0, None) / 86400.0
    assert np.allclose(c.drag.numpy(), expect) and c.drag[-1] * 86400 == 1 and (c.drag[:9] == 0).all()
    sh = c.sh
    zeta = torch.zeros(1, len(LEVELS), sh.trunc + 1, sh.trunc + 1, dtype=torch.complex64)
    zeta[0, :, 0, 1] = 1e-5  # solid-body rotation at every level
    temp = sh.to_spec(torch.tensor(T_REF, dtype=torch.float32)[None, :, None, None].expand(1, len(LEVELS), 16, 32).contiguous())
    st = {"zeta": zeta, "div": torch.zeros_like(zeta), "temp": temp, "q": torch.zeros_like(temp),
          "phib": torch.zeros_like(temp[:, 0])}
    dz = c.tendencies(st)["zeta"][0, :, 0, 1]
    assert torch.allclose(dz.real, -c.drag * 1e-5, atol=1e-13)  # only the drag acts on the rotation


def test_core_term_knockouts_build_and_run():
    for term in pc.TERMS:
        terms = tuple(t for t in pc.TERMS if t != term)
        c = pc.PrimitiveCore(pc.CoreConfig(trunc=10, nlat=16, nlon=32, terms=terms), LEVELS, T_REF)
        sh = c.sh
        st = {k: _random_spec(sh, 1, len(LEVELS), seed=i) * 1e-6 for i, k in enumerate(("zeta", "div", "temp", "q"))}
        st["temp"] = st["temp"] + sh.to_spec(torch.tensor(T_REF, dtype=torch.float32)[None, :, None, None].expand(1, len(LEVELS), 16, 32).contiguous())
        st["phib"] = torch.zeros_like(st["temp"][:, 0])
        out = c.step(st)
        assert all(torch.isfinite(torch.view_as_real(v)).all() for v in out.values()), term


def _tiny():
    levels = (500, 850)
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", levels))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=levels)
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    return layout, stats, data


def _model(layout, stats, data, column=True, budget=True):
    cfg = PhysicsModelConfig.from_dict({
        "core": {"trunc": 5, "nlat": 8, "nlon": 16, "dt_minutes": 60.0},
        "column": {"enabled": column, "hidden": 16, "layers": 1, "call_minutes": 120.0, "budget": budget}})
    torch.manual_seed(0)
    model = PhysicsWorldModel(cfg, layout, data.lat, data.lon)
    model.set_normalization(stats)
    return model


def test_physics_model_rollout_and_gradients():
    layout, stats, data = _tiny()
    batch = {k: v[None] for k, v in data[0].items()}
    n0 = _model(layout, stats, data, column=False)
    p0 = n0.rollout(batch, 2)
    assert p0.shape == (1, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(p0).all()
    # zero-initialised networks: N1 starts exactly at N0 (without the budget's
    # drag and evaporation, which start small but nonzero)
    free = _model(layout, stats, data, column=True, budget=False)
    assert torch.allclose(free.rollout(batch, 2), p0, atol=1e-5)
    n1 = _model(layout, stats, data, column=True)
    tp = [c.key for c in layout.targets].index("total_precipitation_6hr")
    p1 = n1.rollout(batch, 2)
    assert (p1[..., tp, :, :] > p0[..., tp, :, :]).all()  # evaporation rains out
    keep = [i for i in range(len(layout.targets)) if i != tp]
    assert (p1[..., keep, :, :] - p0[..., keep, :, :]).abs().max() < 0.1  # weak initial drag
    # carried residual: lead 0 of a zero-length forecast reproduces the inputs
    state, obs = n0.initial_state(batch)
    keys = [c.key for c in layout.targets]
    phys = n0.diagnose(state) + (obs - n0.diagnose(state))
    j = keys.index("geopotential@500")
    x = batch["inputs"][0, -1, [c.key for c in layout.dynamic].index("geopotential@500")]
    assert torch.allclose(phys[0, j], x * stats.std["geopotential@500"] + stats.mean["geopotential@500"], rtol=1e-4)
    # gradients reach both networks through the differentiable core
    loss_fn = GraphCastLoss(layout, data.lat)
    for _ in range(2):  # the second backward sees nonzero hidden-layer gradients
        loss = loss_fn(n1.forward_rollout(batch, 2), batch["targets"])
        n1.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in n1.parameters():
                p -= 1e-2 * (p.grad if p.grad is not None else 0)
    grads = {n: p.grad.abs().sum().item() for n, p in n1.named_parameters() if p.grad is not None}
    assert grads["column.0.weight"] > 0 and grads["surface.0.weight"] > 0


def test_evaluation_resumes_where_it_stopped(tmp_path):
    from climate_wm import pipeline
    layout, stats, data = _tiny()
    model = _model(layout, stats, data, column=True)
    cfg = {"eval": {"lead_steps": 2, "batch_size": 1, "climatology_window_days": 1}}
    clim = pipeline.build_climatology(cfg, data)
    idx = list(range(len(data)))
    full = pipeline.evaluate(cfg, model, data, stats, clim, indices=idx)["metrics"]
    path = str(tmp_path / "progress.pt")
    pipeline.evaluate(cfg, model, data, stats, clim, indices=idx, progress_path=path, max_batches=1)
    resumed = pipeline.evaluate(cfg, model, data, stats, clim, indices=idx, progress_path=path)["metrics"]
    for name in ("model", "persistence"):
        a, b = full[name].compute(), resumed[name].compute()
        assert np.allclose(a["rmse"], b["rmse"]) and np.allclose(a["acc"], b["acc"], equal_nan=True)
