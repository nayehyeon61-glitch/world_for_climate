import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.models.lie_generator import compose
from climate_wm.models.world_model import TRANSITIONS, ClimateWorldModel, WorldModelConfig

TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 1, "patch_splits": 0},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1},
    "diffusion": {"layers": 1, "cond_dim": 16, "num_steps_denoising": 2},
    "residual_layers": 1,
}
WEIGHTS = {"rec": 1, "forecast": 1, "latent": 0.1, "geo": 0.01, "joint": 0.01, "gen": 1}


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    batch = torch.utils.data.default_collate([data[0], data[1]])
    return layout, data, batch


def _model(layout, data, transition, coupling="independent"):
    torch.manual_seed(0)
    cfg = dict(TINY, transition=transition)
    cfg["lie"] = dict(TINY["lie"], shell_coupling=coupling)
    return ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_stages_and_rollout(setup, transition):
    layout, data, batch = setup
    model = _model(layout, data, transition)
    stages = ["repr", "dynamics"] + (["stochastic"] if model.denoiser is not None else [])
    for stage in stages:
        if stage == "stochastic":
            model.calibrate_residual_scale(batch)
        model.zero_grad()
        loss, logs = model.losses(batch, stage, WEIGHTS, rollout_steps=2)
        assert torch.isfinite(loss)
        loss.backward()
        trained = model.denoiser if stage == "stochastic" else model.encoder
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained.parameters())
    pred = model.rollout(batch, steps=2, members=3)
    assert pred.shape == (3, 2, 2, len(layout.targets), 9, 16)
    assert torch.isfinite(pred).all()


def test_lie_transition_starts_at_identity(setup):
    layout, data, batch = setup
    model = _model(layout, data, "lie")
    history = model.encode_history(batch)
    out = model.step(history, batch["forcings_tgt"][:, 0], sample=False)
    torch.testing.assert_close(out.zbar.h, history[-1].h, atol=1e-5, rtol=0)
    torch.testing.assert_close(out.zbar.v, history[-1].v, atol=1e-5, rtol=0)


def test_chain_coupling_is_forward_kinematics(setup):
    layout, data, batch = setup
    model = _model(layout, data, "lie", coupling="chain")
    with torch.no_grad():
        for p in (model.lie.velocity.weight, model.lie.spin.weight):
            p.normal_(0, 0.5)
    history = model.encode_history(batch)
    nf = model.node_forcing(batch["forcings_tgt"][:, 0])
    omega, _, joints = model.lie(model.graphs, history, nf)
    assert omega.abs().sum() > 0
    torch.testing.assert_close(omega[:, 0], joints[:, 0])
    for s in range(1, omega.shape[1]):
        torch.testing.assert_close(omega[:, s], compose(omega[:, s - 1], joints[:, s]),
                                   atol=1e-5, rtol=1e-4)
