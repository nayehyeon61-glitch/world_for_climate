import math

import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.eval.physics_diag import EARTH_RADIUS
from climate_wm.models.critic import CriticConfig, LieFisherCritic, member_weights
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig

TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 2, "patch_splits": 0,
             "pyramid_levels": (1, 2), "pyramid_detail_level": 1},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1, "shell_coupling": "independent"},
    "diffusion": {"layers": 1, "cond_dim": 16, "num_steps_denoising": 2},
    "pyramid": {"level_weights": (1.0, 0.5), "skip_dropout": 0.5, "level_layers": 1},
    "transition": "pyramid_diffusion",
    "critic": {"enabled": True, "ramp_steps": 2},
}
WEIGHTS = {"rec": 1, "forecast": 1, "latent": 0.1, "geo": 0.01, "joint": 0.01, "phys": 0.1, "fisher": 0.1}


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=4, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    batch = torch.utils.data.default_collate([data[i] for i in range(4)])
    torch.manual_seed(0)
    model = ClimateWorldModel(WorldModelConfig.from_dict(TINY), layout, data.lat, data.lon)
    model.set_normalization(stats)
    return layout, data, stats, batch, model


def test_diagnostic_critic_prefers_observed_states_over_blurred_ones(setup):
    layout, data, stats, batch, model = setup
    crit = model.critic.diagnostic
    frames = torch.cat([batch["inputs"][:, j] for j in range(batch["inputs"].shape[1])], 0)
    for _ in range(3):
        crit.observe(frames)
    truth = crit.score(frames).mean()
    k = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 1, 3)
    blurred = torch.nn.functional.conv2d(
        torch.nn.functional.pad(frames.flatten(0, 1)[:, None], (1, 1, 0, 0), mode="circular"), k)
    blurred = blurred.view_as(frames)
    zonal = frames.mean(-1, keepdim=True).expand_as(frames)
    s_blur, s_zonal = crit.score(blurred).mean(), crit.score(zonal).mean()
    assert truth < s_blur < s_zonal
    assert crit.score(frames, per_channel=True).shape == (frames.shape[0], len(crit.diag))


def test_lie_fisher_separability_and_penalty():
    g = torch.Generator().manual_seed(0)
    centres = [torch.nn.functional.normalize(torch.randn(6, 3, generator=g), dim=-1)]
    fisher = LieFisherCritic(2, centres, CriticConfig(momentum=0.0))
    cov = torch.tensor([0.04, 0.01, 0.09])
    shift = torch.tensor([0.2, 0.0, 0.0])  # in the Cartan frame: spin only
    frame = fisher.frame  # [P, 3, 3], rows (n, east, north)

    def sample(mean_frame, n=4000):
        f = mean_frame + cov.sqrt() * torch.randn(n, 2, 6, 3, generator=g)
        return torch.einsum("pji,bspj->bspi", frame, f)  # back to ambient coordinates

    obs, mod = sample(shift), sample(torch.zeros(3))
    fisher.observe(obs, mod)
    expected = float((shift.square() / cov).sum())  # 1.0
    assert torch.allclose(fisher.separability(), torch.full((2, 1), expected), rtol=0.1)
    assert fisher.penalty(obs) < 0.01 < fisher.penalty(mod)
    assert fisher.score(obs[:200]).mean() > 0 > fisher.score(mod[:200]).mean()


def test_observed_generators_recover_solid_body_rotation(setup):
    layout, data, stats, batch, model = setup
    lat = np.deg2rad(data.lat)[:, None] * np.ones((1, len(data.lon)))
    u0 = 15.0
    phys = torch.zeros(1, len(layout.dynamic), len(data.lat), len(data.lon))
    for i, c in enumerate(layout.dynamic):
        if c.name in ("u_component_of_wind", "10m_u_component_of_wind"):
            phys[0, i] = torch.tensor(u0 * np.cos(lat), dtype=torch.float32)
    mean, std = stats.arrays(layout.dynamic)
    dyn = (phys - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    omega = model.observed_generators(dyn)
    parts = model.lie.split_parts(model.graphs, omega)
    rate = u0 * 6 * 3600 / EARTH_RADIUS  # rad per 6 h about the polar axis
    expect = torch.tensor([0.0, 0.0, rate])
    coarse = parts[0][0]  # [S, 12, 3]
    assert torch.allclose(coarse, expect.expand_as(coarse), atol=0.15 * rate)
    assert parts[1].abs().max() < 0.15 * rate  # a rigid field has no detail rotations


def test_critic_losses_ramp_and_member_weights(setup):
    layout, data, stats, batch, model = setup
    model.train()
    model.critic.steps.zero_()
    _, logs = model.losses(batch, "dynamics", WEIGHTS, rollout_steps=2)
    assert logs["phys"] == 0.0 and logs["fisher"] == 0.0  # ramp starts at zero
    assert int(model.critic.steps) == 1 and int(model.critic.fisher.count) == 2
    model.zero_grad()
    loss, logs = model.losses(batch, "dynamics", WEIGHTS, rollout_steps=1)
    assert logs["phys"] > 0 and math.isfinite(logs["fisher"])
    loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert "fisher_J" in model.critic.summary()
    w = member_weights(torch.tensor([[1.0, 0.2], [3.0, 0.1]]), tau=1.0)
    assert torch.allclose(w.sum(0), torch.ones(2)) and w[0, 0] > w[1, 0] and w[0, 1] < w[1, 1]
    # standardised: only the ranking within a forecast matters, not the score scale
    assert torch.allclose(member_weights(torch.tensor([[1.0], [3.0]])),
                          member_weights(torch.tensor([[10.0], [30.0]])))
    # checkpoints carry the observed statistics
    assert any(k.startswith("critic.diagnostic.mean") for k in model.state_dict())


def test_weighted_crps():
    from climate_wm.eval.stage5 import weighted_crps
    g = torch.Generator().manual_seed(0)
    ens, y = torch.randn(5, 7, generator=g), torch.randn(7, generator=g)
    eq = torch.full((5, 1), 0.2)
    biased = (ens - y).abs().mean(0) - 0.5 * (ens[:, None] - ens[None]).abs().mean((0, 1))
    assert torch.allclose(weighted_crps(ens, y, eq), biased, atol=1e-6)
    one = torch.zeros(5, 1)
    one[2] = 1.0
    assert torch.allclose(weighted_crps(ens, y, one), (ens[2] - y).abs(), atol=1e-6)  # a single member
