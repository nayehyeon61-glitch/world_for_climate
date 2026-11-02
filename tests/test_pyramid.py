import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.geometry import spherical_mesh
from climate_wm.geometry.group_actions import killing_field, so3_exp
from climate_wm.geometry.shells import PatchPyramid, SpherePatches
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig

TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 2, "patch_splits": 0,
             "pyramid_levels": (1, 2), "pyramid_detail_level": 2},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1, "shell_coupling": "independent"},
    "diffusion": {"layers": 1, "cond_dim": 16, "num_steps_denoising": 2},
    "pyramid": {"level_weights": (1.0, 0.5), "skip_dropout": 0.5, "level_layers": 1},
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


def _model(layout, data, transition="pyramid_diffusion", levels=(1, 2)):
    torch.manual_seed(0)
    cfg = dict(TINY, transition=transition)
    cfg["mesh"] = dict(TINY["mesh"], pyramid_levels=levels)
    return ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)


def _nodes(splits=3):
    return spherical_mesh.icosphere_hierarchy(splits)[-1].vertices


def test_pyramid_sizes_and_parents():
    pyr = PatchPyramid(_nodes(), levels=(0, 1, 2))
    assert [p.num_patches for p in pyr.patches] == [12, 42, 162]
    for level in (1, 2):
        parent = pyr.parent(level)
        assert parent.shape == (pyr.patches[level].num_patches,)
        assert parent.min() >= 0 and parent.max() < pyr.patches[level - 1].num_patches
        # every coarse patch is somebody's parent
        assert len(torch.unique(parent)) == pyr.patches[level - 1].num_patches


def test_global_rotation_projects_to_same_element_at_every_level():
    nodes = _nodes()
    pyr = PatchPyramid(nodes, levels=(0, 1, 2))
    omega = torch.tensor([[0.02, -0.05, 0.03]])
    u = killing_field(omega[:, None, :].expand(1, len(nodes), 3),
                      torch.tensor(nodes, dtype=torch.float32)[None])
    assert torch.allclose(pyr.root(u), omega, atol=1e-5)
    omegas = pyr.project(u)
    # the hierarchical prior supplies the axial component small patches cannot
    # see, so the rigid field is recovered exactly at every level
    for om in omegas:
        assert torch.allclose(om, omega[:, None, :].expand_as(om), atol=1e-4)
    parts = pyr.details(omegas)
    assert torch.allclose(parts[0], omega[:, None, :].expand_as(parts[0]), atol=1e-4)
    for d in parts[1:]:
        assert d.abs().max() < 2e-4  # a rigid field has no detail rotations
    node_omega = pyr.node_algebra(parts)
    assert torch.allclose(node_omega, omega[:, None, :].expand_as(node_omega), atol=2e-4)
    # without the prior the finest level loses up to ~20% of the axial component
    flat = SpherePatches(nodes, 2).killing_projection(u)
    assert (flat - omega[:, None, :]).abs().max() > 5e-3


def test_details_round_trip():
    pyr = PatchPyramid(_nodes(), levels=(0, 1, 2))
    torch.manual_seed(0)
    omegas = [0.3 * torch.randn(2, p.num_patches, 3) for p in pyr.patches]
    rec = pyr.reconstruct(pyr.details(omegas))
    for a, b in zip(rec, omegas):
        assert torch.allclose(a, b, atol=1e-4)


def test_coarse_projection_matches_direct_least_squares():
    """The coarse Killing projection is the least-squares fit of the node field,
    which is also the inertia-weighted pooling of finer fits for a field that is
    itself a blend of patch rotations."""
    nodes = _nodes()
    pyr = PatchPyramid(nodes, levels=(0, 2))
    torch.manual_seed(1)
    fine = 0.1 * torch.randn(1, pyr.patches[1].num_patches, 3)
    u = killing_field(pyr.patches[1].blend(fine), torch.tensor(nodes, dtype=torch.float32)[None])
    root = pyr.root(u)[:, None, :].expand(1, 12, 3)
    direct = SpherePatches(nodes, 0).killing_projection(u, root)
    assert torch.allclose(pyr.project(u)[0], direct, atol=1e-6)
    # pooled (fine -> coarse) energy is not larger than the fine energy
    energy = pyr.energy(pyr.project(u))
    assert energy.shape == (2,) and (energy >= 0).all()
    assert energy[0] <= energy[1] + 1e-6


def test_node_algebra_is_scale_chain():
    nodes = _nodes()
    pyr = PatchPyramid(nodes, levels=(0, 1))
    torch.manual_seed(2)
    parts = [0.2 * torch.randn(1, 12, 3), 0.05 * torch.randn(1, 42, 3)]
    out = pyr.node_algebra(parts)
    coarse = pyr.patches[0].blend(parts[0])
    detail = pyr.patches[1].blend(parts[1])
    expected = so3_exp(coarse) @ so3_exp(detail)
    assert torch.allclose(so3_exp(out), expected, atol=1e-5)


# ----------------------------------------------------------- world model


def test_pyramid_graphs_shapes(setup):
    layout, data, batch = setup
    model = _model(layout, data)
    pyr = model.graphs.pyramid
    assert pyr.levels == (1, 2) and pyr.counts == (42, 162)
    assert model.graphs.num_latent == 204 and model.graphs.latent_struct.shape == (204, 6)
    assert pyr.rot_levels == (0, 1, 2) and pyr.rot_source == (0, 0, 1)
    assert pyr.patch_counts == (12, 42, 162)
    # every stacked node gets a forcing value; a constant field stays constant per level
    const = torch.ones(2, 3, 9, 16)
    nf = model.graphs.grid_to_latent_mean(const)
    assert nf.shape == (2, 204, 3)
    assert torch.allclose(nf, torch.ones_like(nf), atol=1e-5)
    # the stacked mesh links the levels in both directions
    kinds = model.graphs.latent_mesh.edge_attr[:, -1]
    assert (kinds > 0).any() and (kinds < 0).any() and (kinds == 0).any()


@pytest.mark.parametrize("transition,levels", [("pyramid_diffusion", (1, 2)),
                                               ("pyramid", (1, 2)),
                                               ("pyramid_diffusion", (1,))])
def test_pyramid_stages_and_rollout(setup, transition, levels):
    layout, data, batch = setup
    model = _model(layout, data, transition, levels)
    stages = ["repr", "dynamics"] + (["stochastic"] if model.denoiser is not None else [])
    for stage in stages:
        if stage == "stochastic":
            model.calibrate_residual_scale(batch)
        model.zero_grad()
        loss, logs = model.losses(batch, stage, WEIGHTS, rollout_steps=2)
        assert torch.isfinite(loss), (stage, logs)
        loss.backward()
        trained = model.denoiser if stage == "stochastic" else model.encoder
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained.parameters())
        if stage == "dynamics":
            assert "joint" in logs and "geo" in logs
    pred, omegas = model.rollout(batch, steps=2, members=2, return_omega=True)
    assert pred.shape == (2, 2, 2, len(layout.targets), 9, 16)
    assert torch.isfinite(pred).all()
    pyr = model.graphs.pyramid
    assert omegas[0].shape[-2] == sum(pyr.patch_counts)


def test_pyramid_transition_starts_at_identity(setup):
    layout, data, batch = setup
    model = _model(layout, data, "pyramid", (1, 2))
    model.eval()
    with torch.no_grad():
        hist = model.encode_history(batch)
        nf = model.node_forcing(batch["forcings_tgt"][:, 0])
        zbar, omega, joints = model.structural(hist, nf)
    assert omega.abs().max() < 1e-6 and joints.abs().max() < 1e-6
    assert torch.allclose(zbar.h, hist[-1].h, atol=1e-4)
    assert torch.allclose(zbar.v, hist[-1].v, atol=1e-4)


def test_pyramid_latent_error_is_level_weighted(setup):
    layout, data, batch = setup
    model = _model(layout, data)
    pyr = model.graphs.pyramid
    with torch.no_grad():
        z = model.encode_history(batch)[-1]
    # perturb only the fine level: the weighted error is 0.5 / 1.5 of the plain one
    delta = torch.zeros_like(z.h)
    delta[:, :, pyr.slice(1)] = 1.0
    other = type(z)(z.h + delta, z.v)
    plain = z.squared_error(other)
    fine_only = type(z)(z.h[:, :, pyr.slice(1)], z.v[:, :, pyr.slice(1)]).squared_error(
        type(z)(other.h[:, :, pyr.slice(1)], other.v[:, :, pyr.slice(1)]))
    weighted = model.latent_error(z, other)
    assert abs(weighted.item() - 0.5 * fine_only.item() / 1.5) < 1e-6
    assert weighted.item() != plain.item()


def test_pyramid_skip_dropout_changes_training_decode(setup):
    layout, data, batch = setup
    model = _model(layout, data)
    with torch.no_grad():
        z = model.encode_history(batch)[-1]
        model.eval()
        a = model.decode(z)
        b = model.decode(z)
        assert torch.allclose(a, b)
        model.train()
        torch.manual_seed(1)
        outs = [model.decode(z) for _ in range(4)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])


def test_pyramid_evaluation_energy_and_transfer(setup):
    from climate_wm.eval import pyramid as pyr_eval
    layout, data, batch = setup
    model = _model(layout, data, "pyramid", (1, 2))
    pairs = pyr_eval.collect_pairs(model, data, range(len(data)), batch_size=3)
    assert len(pairs) == len(data) and len(pairs.hist) == 2
    parts = pyr_eval.explain_all(model, pairs, batch_size=3)
    assert [p.shape[-2] for p in parts] == [12, 42, 162]
    energy = pyr_eval.level_energy(parts, pyr_eval.patch_latitudes(model), [s.name for s in layout.shells.shells])
    assert len(energy["levels"]) == 3 and all(r["energy"] >= 0 for r in energy["levels"])
    analogs = pyr_eval.find_analogs(pairs, min_gap_hours=6.0)
    assert (analogs >= 0).any()
    tr = pyr_eval.level_transfer(model, pairs, parts, analogs, batch_size=3)
    assert tr["pairs"] > 0 and tr["coarse_levels"] == [0, 1] and tr["fine_level"] == 2
    for name in ("persistence", "own", "all", "coarse", "fine", "random_all"):
        assert np.isfinite(tr["latent_" + name]) and len(tr["levels_" + name]) == 2
    assert "coarse_gap" in tr and "fine_gap" in tr
    assert pyr_eval.markdown({"experiment": "t", "energy": energy, "transfer": tr}).startswith("#")
