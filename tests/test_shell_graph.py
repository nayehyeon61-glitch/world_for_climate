import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.eval import causes
from climate_wm.geometry.group_actions import so3_exp
from climate_wm.models.layers import ShellGraphMixer, VerticalShellMixer
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig

TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 2, "patch_splits": 0,
             "pyramid_levels": (1, 2), "pyramid_detail_level": 2},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1, "shell_coupling": "independent"},
    "diffusion": {"layers": 1, "cond_dim": 16, "num_steps_denoising": 2},
    "pyramid": {"level_weights": (1.0, 0.5), "skip_dropout": 0.0, "level_layers": 1},
    "transition": "pyramid_diffusion",
}
WEIGHTS = {"rec": 1, "forecast": 1, "latent": 0.1, "geo": 0.01, "joint": 0.01}


def _cfg(vertical="chain", coupling="independent"):
    return dict(TINY, gnn=dict(TINY["gnn"], vertical=vertical),
                lie=dict(TINY["lie"], shell_coupling=coupling))


def test_mixer_starts_on_the_chain_and_is_equivariant():
    torch.manual_seed(0)
    s, d, cv = 5, 8, 3
    mix = ShellGraphMixer(d, cv, s)
    h, v = torch.randn(2, s, 7, d), torch.randn(2, s, 7, cv, 3)
    a = mix.weights(h)
    assert torch.allclose(a.sum(-1), torch.ones(2, 7, s))
    assert (a.diagonal(dim1=-2, dim2=-1) == 0).all()  # no self-edges
    # zero-initialised query: the start is the bias, peaked on the chain neighbours
    assert torch.allclose(a[0, 0], a[1, 3])
    assert a[0, 0, 2, 1] > 5 * a[0, 0, 2, 4] and a[0, 0, 2, 3] > 5 * a[0, 0, 2, 0]
    mix.local_only = True
    a_loc = mix.weights(h)
    assert (a_loc[..., ~mix.chain_edge] == 0).all() and torch.allclose(a_loc.sum(-1), torch.ones(2, 7, s))
    mix.local_only = False
    # rotating every vector by R rotates the output vectors and leaves the scalars
    with torch.no_grad():
        mix.query.weight.normal_()
    rot = so3_exp(torch.tensor([0.3, -0.5, 0.8]))
    h1, v1 = mix(h, v)
    h2, v2 = mix(h, v @ rot.T)
    assert torch.allclose(h1, h2, atol=1e-5) and torch.allclose(v1 @ rot.T, v2, atol=1e-5)


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    batch = torch.utils.data.default_collate([data[0], data[1]])
    return layout, data, stats, batch


def _model(layout, data, **kw):
    torch.manual_seed(0)
    return ClimateWorldModel(WorldModelConfig.from_dict(_cfg(**kw)), layout, data.lat, data.lon)


def test_graph_model_replaces_every_chain_mixer(setup):
    layout, data, _, batch = setup
    chain = _model(layout, data)
    graph = _model(layout, data, vertical="graph", coupling="graph")
    n_chain = sum(isinstance(m, VerticalShellMixer) for m in chain.modules())
    assert n_chain > 4  # encoder, decoder, V-Net blocks, transition, denoiser
    assert not any(isinstance(m, VerticalShellMixer) for m in graph.modules())
    assert sum(isinstance(m, ShellGraphMixer) for m in graph.modules()) == n_chain
    assert graph.lie.shell_graph is not None and chain.lie.shell_graph is None
    graph.train()
    loss, _ = graph.losses(batch, "dynamics", WEIGHTS, rollout_steps=1)
    loss.backward()
    mixer = next(m for m in graph.modules() if isinstance(m, ShellGraphMixer))
    assert mixer.bias.grad is not None and torch.isfinite(mixer.bias.grad).all()
    # the checkpoint of a graph model loads into a graph model
    again = _model(layout, data, vertical="graph", coupling="graph")
    again.load_state_dict(graph.state_dict())
    with pytest.raises(ValueError):
        WorldModelConfig.from_dict(dict(TINY, gnn=dict(TINY["gnn"], vertical="tree")))
    with pytest.raises(ValueError):  # graph coupling of generators needs the pyramid
        ClimateWorldModel(WorldModelConfig.from_dict(
            dict(TINY, mesh={"mesh_splits": 2, "latent_splits": 2, "patch_splits": 0},
                 transition="lie_diffusion", lie={"layers": 1, "shell_coupling": "graph"})),
            layout, data.lat, data.lon)


def test_generator_coupling(setup):
    layout, data, _, batch = setup
    model = _model(layout, data, coupling="graph")
    lie = model.lie
    s = len(layout.shells)
    with torch.no_grad():
        for p in (lie.velocity.weight, lie.spin.weight):
            p.normal_(0, 0.1)
        hist = model.encode_history(batch)
        nf = model.node_forcing(batch["forcings_tgt"][:, 0])
        h, v = lie.features(model.graphs, hist, nf)
        omega0, _, _ = lie.algebra(model.graphs, h, v)  # A = 0: independent shells
        u = (lie.velocity(h)[..., None] * v).sum(-2)
        plain, _, _ = lie.project(model.graphs, u, lie.spin(h))
        assert torch.allclose(omega0, plain)
        # strictly lower-triangular ones: each shell adds the generators below it,
        # the first-order (BCH) form of the stage 1 chain g_s = g_{s-1} exp(delta_s)
        lie.shell_graph.copy_(torch.tril(torch.ones(s, s), -1) + 7 * torch.eye(s))  # diagonal ignored
        chained, _, _ = lie.algebra(model.graphs, h, v)
        assert torch.allclose(chained, plain.cumsum(1), atol=1e-6)
        # the observed-wind generators of the stage 5 critic are not coupled
        observed, _, _ = lie.project(model.graphs, u, lie.spin(h))
        assert torch.allclose(observed, plain)
        with causes.knocked_out(model, ["shell_graph"]):
            assert lie.shell_graph.abs().sum() == 0
            assert torch.allclose(lie.algebra(model.graphs, h, v)[0], plain)
        assert lie.shell_graph[2, 0] == 1


def test_cause_analysis_of_a_graph_model(setup):
    layout, data, stats, _ = setup
    model = _model(layout, data, vertical="graph", coupling="graph")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ShellGraphMixer):
                m.query.weight.normal_(0, 0.5)
        model.lie.shell_graph.normal_(0, 0.3)
        model.lie.velocity.weight.normal_(0, 0.1)
    variants = causes.available_variants(model)
    assert variants == ["no_denoiser", "nonlocal", "shell_graph"]
    res = causes.error_decomposition(model, data, [0, 1], stats, lead_steps=2, variants=variants)
    graphs = res["shell_graph"]
    assert len(graphs["mixers"]) == sum(isinstance(m, ShellGraphMixer) for m in model.modules())
    entry = next(iter(graphs["mixers"].values()))
    mean = np.array(entry["mean"])
    assert mean.shape == (len(layout.shells),) * 2 and np.allclose(mean.sum(-1), 1, atol=1e-5)
    assert 0 < entry["nonlocal_share"] < 1 and 0 < entry["prior_nonlocal_share"] < 0.2
    assert np.array(graphs["generator_coupling"]).shape == (len(layout.shells),) * 2
    for name in ("nonlocal", "shell_graph"):
        assert np.abs(np.array(res["variant_change"][name]["mean"])).max() > 0
    assert not any(m.local_only for m in model.modules() if isinstance(m, ShellGraphMixer))
