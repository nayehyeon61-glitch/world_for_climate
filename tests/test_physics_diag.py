import math

import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.eval import physics_diag as pdg
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig

LEVELS = (500, 850, 925, 1000)
LAT = np.linspace(-80, 80, 17)
LON = np.arange(32) * 360.0 / 32


def _diag(levels=LEVELS, lat=LAT, **kw):
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", levels))
    return layout, pdg.PhysicsDiagnostics(layout, lat, LON, pdg.DiagnosticsConfig(enabled=True, **kw))


def _state(layout, fields):
    """Physical state [1, C_dyn, H, W] from {(name, level): [H, W]} (zeros elsewhere)."""
    x = torch.zeros(1, len(layout.dynamic), len(LAT), len(LON))
    for i, c in enumerate(layout.dynamic):
        if (c.name, c.level) in fields:
            x[0, i] = torch.as_tensor(fields[(c.name, c.level)], dtype=torch.float32)
    return x


def _balanced(diag, t0=250.0, u_scale=1.0):
    """Isothermal, dry, hydrostatic geopotential with a wave; wind = geostrophic wind of Phi."""
    phi = np.deg2rad(LAT)[:, None]
    lam = np.deg2rad(LON)[None, :]
    fields = {}
    for l in LEVELS:
        z = 5000.0 + pdg.R_D * t0 * math.log(1000.0 / l) + 300.0 * np.cos(phi) ** 2 * np.cos(2 * lam)
        zt = torch.tensor(z, dtype=torch.float32)
        gx, gy = diag._grad(zt)
        f = diag.coriolis[:, None]
        f = torch.where(f == 0, torch.ones_like(f), f)  # no geostrophic wind on the equator
        fields[("geopotential", l)] = z
        fields[("temperature", l)] = np.full_like(z, t0)
        fields[("u_component_of_wind", l)] = (-gy / f * u_scale).numpy()
        fields[("v_component_of_wind", l)] = (gx / f * u_scale).numpy()
        fields[("specific_humidity", l)] = np.zeros_like(z)
    return fields


def test_channels_land_in_the_shell_of_their_level():
    layout, diag = _diag()
    keys = [(c.name, c.level) for c in diag.channels]
    assert ("diag_hydrostatic", 1000) in keys and ("diag_hydrostatic", 500) not in keys  # pair -> lower level
    assert ("diag_geostrophic", None) in keys and ("diag_surface", None) in keys
    lay = pdg.with_diagnostics(layout, diag.channels)
    assert lay.num_encoder_channels == layout.num_encoder_channels + len(diag)
    assert layout.diagnostics == []  # the original layout is not modified
    surface = lay.diagnostic_shells[lay.shells.surface_index].scalar
    assert sorted(diag.channels[i].level is None for i in surface) == [True] * len(surface)
    _, only = _diag(families=("kinematics",))
    assert {c.name for c in only.channels} == {"diag_divergence", "diag_vorticity"}
    with pytest.raises(ValueError):
        pdg.DiagnosticsConfig(families=("magic",))


def test_balanced_state_has_zero_hydrostatic_and_geostrophic_residual():
    layout, diag = _diag()
    raw = diag.raw(_state(layout, _balanced(diag)))
    for lo in (1000, 925, 850):
        assert raw[("hydrostatic", lo)].abs().max() < 1e-5
    off_eq = np.abs(LAT) >= 10
    for l in LEVELS:
        assert raw[("geostrophic", l)][0, off_eq].abs().max() < 1e-4
        assert raw[("geostrophic", l)][0, ~off_eq].abs().max() == 0  # masked near the equator
    # doubling the wind leaves half of the Coriolis force unbalanced: ratio 1/3
    raw2 = diag.raw(_state(layout, _balanced(diag, u_scale=2.0)))
    assert torch.allclose(raw2[("geostrophic", 500)][0, off_eq], torch.tensor(1 / 3), atol=1e-3)
    # a warmer column than the geopotential implies is a hydrostatic error
    f = _balanced(diag)
    f[("temperature", 850)] = f[("temperature", 850)] + 10.0
    assert raw[("hydrostatic", 850)].abs().max() < diag.raw(_state(layout, f))[("hydrostatic", 850)].abs().min()


def test_isothermal_stability_matches_g2_over_cp_t():
    layout, diag = _diag()
    raw = diag.raw(_state(layout, _balanced(diag, t0=250.0)))
    expected = pdg.GRAVITY ** 2 / (pdg.CP * 250.0)
    for lo in (1000, 925, 850):
        assert torch.allclose(raw[("stability", lo)], torch.tensor(expected), rtol=0.03)
    f = _balanced(diag)
    f[("temperature", 925)] = f[("temperature", 925)] - 8.0  # cold air above warm: unstable
    assert (diag.raw(_state(layout, f))[("stability", 1000)] < 0).all()


def test_saturation_deficit_and_kinematics():
    layout, diag = _diag()
    phi = np.deg2rad(LAT)[:, None] * np.ones((1, len(LON)))
    t = np.full_like(phi, 293.15)
    es = 611.2 * math.exp(17.67 * 20.0 / (293.15 - 29.65))
    qs = 0.622 * es / (1e5 - 0.378 * es)
    assert abs(es - 2338.0) < 5.0  # Bolton at 20 C
    u0 = 20.0
    fields = {("temperature", 1000): t, ("specific_humidity", 1000): np.full_like(t, 0.5 * qs),
              ("u_component_of_wind", 1000): u0 * np.cos(phi),  # solid-body rotation
              ("v_component_of_wind", 1000): np.zeros_like(t)}
    raw = diag.raw(_state(layout, fields))
    assert torch.allclose(raw[("saturation", 1000)], torch.tensor(0.5), atol=1e-4)
    assert raw[("divergence", 1000)].abs().max() < 1e-9
    interior = slice(1, -1)
    vort = raw[("vorticity", 1000)][0, interior]
    exact = torch.tensor(2 * u0 * np.sin(phi[interior]) / pdg.EARTH_RADIUS, dtype=torch.float32)
    assert torch.allclose(vort, exact, rtol=0.03, atol=1e-8)


def test_pole_rows_and_differentiability():
    lat = np.linspace(-90, 90, 19)
    layout, diag = _diag(lat=lat)
    x = torch.randn(2, len(layout.dynamic), len(lat), len(LON), requires_grad=True)
    feats = diag(x)
    assert feats.shape == (2, len(diag), len(lat), len(LON))
    assert torch.isfinite(feats).all() and feats.abs().max() <= 1
    names = [c.name for c in diag.channels]
    for k in ("diag_divergence", "diag_vorticity", "diag_geostrophic"):
        i = names.index(k)
        assert feats[:, i, [0, -1]].abs().max() == 0
    feats.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    batch = torch.utils.data.default_collate([data[0], data[1]])
    return layout, data, stats, batch


TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 2, "patch_splits": 0,
             "pyramid_levels": (1, 2), "pyramid_detail_level": 2},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1, "shell_coupling": "independent"},
    "diffusion": {"layers": 1, "cond_dim": 16, "num_steps_denoising": 2},
    "pyramid": {"level_weights": (1.0, 0.5), "skip_dropout": 0.5, "level_layers": 1},
    "transition": "pyramid_diffusion",
}


def test_world_model_with_diagnostic_inputs(setup):
    layout, data, stats, batch = setup
    torch.manual_seed(0)
    base = ClimateWorldModel(WorldModelConfig.from_dict(TINY), layout, data.lat, data.lon)
    assert base.diagnostics is None and base.layout.num_encoder_channels == layout.num_encoder_channels
    torch.manual_seed(0)
    cfg = dict(TINY, diagnostics={"enabled": True})
    model = ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)
    model.diagnostics.set_stats(stats)
    assert layout.diagnostics == []
    dyn = batch["inputs"][:, -1]
    x = model.encoder_input(dyn, batch["forcings_in"][:, -1], batch["static"])
    assert x.shape[1] == model.layout.num_encoder_channels == layout.num_encoder_channels + len(model.diagnostics)
    # the diagnostics see physical units: the normalised input maps back to the raw frame
    phys = model.diagnostics.physical(dyn)
    assert torch.allclose(phys, stats.unnormalize(dyn, layout.dynamic), rtol=1e-5, atol=1e-3)
    model.train()
    loss, _ = model.losses(batch, "dynamics", {"rec": 1, "forecast": 1, "latent": 0.1,
                                               "geo": 0.01, "joint": 0.01}, rollout_steps=1)
    loss.backward()
    grads = [p.grad for p in model.encoder.in_scalar.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # the normalisation buffers travel with the checkpoint
    state = model.state_dict()
    assert "diagnostics.mean" in state and "diagnostics.std" in state
