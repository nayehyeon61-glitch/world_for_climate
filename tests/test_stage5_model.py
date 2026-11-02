import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.geometry import composite_group as cg
from climate_wm.models.latent import Latent
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig

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
WEIGHTS = {"rec": 1, "forecast": 1, "latent": 0.1, "geo": 0.01, "joint": 0.01, "fib": 0.01}


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    batch = torch.utils.data.default_collate([data[0], data[1]])
    return layout, data, stats, batch


def _model(layout, data, **composite):
    torch.manual_seed(0)
    cfg = dict(TINY, composite=dict({"enabled": True}, **composite))
    return ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)


def _inputs(model, batch):
    hist = model.encode_history(batch)
    return hist, model.node_forcing(batch["forcings_tgt"][:, 0])


def test_composite_model_starts_as_the_rotation_only_step(setup):
    layout, data, _, batch = setup
    model = _model(layout, data)
    lie = model.lie
    assert lie.composite is not None and lie.fibre_gain.abs().sum() == 0
    with torch.no_grad():
        for p in (lie.velocity.weight, lie.spin.weight):  # some rotation so the test is not trivial
            p.normal_(0, 0.1)
        hist, nf = _inputs(model, batch)
        zbar, omega, joints, terms = model.structural_terms(hist, nf)
        _, omega_node, _ = lie(model.graphs, hist, nf)
        plain = lie.rotate(model.graphs, hist[-1], omega_node)
    assert terms is not None and terms.shift.abs().max() == 0
    assert torch.allclose(zbar.h, plain.h, atol=1e-6) and torch.allclose(zbar.v, plain.v, atol=1e-6)
    assert float(terms.norm2()) == 0


def test_composite_step_is_the_strang_split_and_conserves_columns(setup):
    layout, data, _, batch = setup
    model = _model(layout, data, families=("mixing", "surface"), vert=False)
    lie = model.lie
    with torch.no_grad():
        lie.fibre_gain.fill_(0.5)
        lie.velocity.weight.normal_(0, 0.1)
        hist, nf = _inputs(model, batch)
        _, omega_node, _, terms = lie.generate(model.graphs, hist, nf)
        z = hist[-1]
        out = lie.act(model.graphs, z, omega_node, terms)
        # with no rotation the step is the fibre flow alone, which conserves column sums
        still = lie.act(model.graphs, z, torch.zeros_like(omega_node), terms)
        manual = cg.split_step(z.h, z.v, lambda h, v: (h, v), None,
                               lambda h, v, dt: lie.fibre.flow(h, v, terms.coef, dt), 1.0, "strang")
    assert terms.coef.mixing.min() >= 0 and terms.coef.mixing.abs().sum() > 0
    assert torch.allclose(still.h, manual[0], atol=1e-5)
    assert torch.allclose(still.h.sum(1), z.h.sum(1), atol=1e-4)
    assert torch.allclose(still.v.sum(1), z.v.sum(1), atol=1e-4)
    assert not torch.allclose(out.h, still.h)


def test_composite_losses_and_gradients(setup):
    layout, data, _, batch = setup
    model = _model(layout, data)
    model.train()
    loss, logs = model.losses(batch, "dynamics", WEIGHTS, rollout_steps=2)
    assert "fib" in logs and logs["fib"] == 0.0  # identity start
    loss.backward()
    lie = model.lie
    # zero-initialised gains and shift still receive gradient
    assert lie.fibre_gain.grad is not None and lie.fibre_gain.grad.abs().sum() > 0
    assert lie.shift.weight.grad is not None and lie.shift.weight.grad.abs().sum() > 0
    # once the gains move, the coefficient heads learn too and L_fib is positive
    model.zero_grad()
    with torch.no_grad():
        lie.fibre_gain.fill_(0.3)
        lie.shift.bias.fill_(0.2)
    loss, logs = model.losses(batch, "dynamics", WEIGHTS, rollout_steps=1)
    loss.backward()
    assert logs["fib"] > 0
    for p in (lie.coef_head.weight, lie.gate_head.weight, lie.fibre.mix_h, lie.fibre.u_rad):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    # a negative gain of a non-negative family is projected to zero in the forward pass
    with torch.no_grad():
        lie.fibre_gain.fill_(-1.0)
        hist, nf = _inputs(model, batch)
        terms = lie.generate(model.graphs, hist, nf)[3]
    assert terms.coef.mixing.abs().max() == 0 and terms.coef.condensation.abs().max() == 0
    assert terms.coef.radiation.abs().max() > 0  # signed family


def test_composite_needs_a_pyramid_and_diagnostics_combine(setup):
    layout, data, stats, batch = setup
    cfg = dict(TINY, transition="lie_diffusion", composite={"enabled": True})
    cfg["mesh"] = {k: v for k, v in TINY["mesh"].items() if not k.startswith("pyramid")}
    with pytest.raises(ValueError):
        ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)
    torch.manual_seed(0)
    cfg = dict(TINY, composite={"enabled": True}, diagnostics={"enabled": True})
    model = ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)
    model.diagnostics.set_stats(stats)
    out = model.step(model.encode_history(batch), batch["forcings_tgt"][:, 0], sample=True)
    assert isinstance(out.z, Latent) and torch.isfinite(out.z.h).all()
    assert out.composite is not None
