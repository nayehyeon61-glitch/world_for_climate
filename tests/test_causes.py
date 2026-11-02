import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.eval import causes
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
    "composite": {"enabled": True},
}


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    torch.manual_seed(0)
    model = ClimateWorldModel(WorldModelConfig.from_dict(TINY), layout, data.lat, data.lon)
    with torch.no_grad():  # give the composite terms something to do
        model.lie.fibre_gain.normal_(0, 1.0)
        for p in model.lie.shift.parameters():
            p.normal_(0, 0.5)
    return layout, data, stats, model


def test_knock_out_restores_the_parameters(setup):
    _, _, _, model = setup
    lie = model.lie
    gain = lie.fibre_gain.detach().clone()
    weight = lie.shift.weight.detach().clone()
    assert causes.available_variants(model) == ["no_denoiser", "composite", "shift", "radiation",
                                                "mixing", "surface", "condensation"]
    with causes.knocked_out(model, ["mixing"]):
        assert lie.fibre_gain[1] == 0 and lie.fibre_gain[0] == gain[0]
        assert torch.equal(lie.shift.weight, weight)
    with causes.knocked_out(model, ["composite"]):
        assert lie.fibre_gain.abs().sum() == 0 and lie.shift.weight.abs().sum() == 0
    assert torch.equal(lie.fibre_gain, gain) and torch.equal(lie.shift.weight, weight)
    with causes.knocked_out(model, ["full", "no_denoiser"]):  # not composite parts: no change
        assert torch.equal(lie.fibre_gain, gain)


def test_composite_off_equals_the_rotation_only_step(setup):
    _, data, _, model = setup
    batch = torch.utils.data.default_collate([data[0], data[1]])
    with torch.no_grad():
        hist = model.encode_history(batch)
        nf = model.node_forcing(batch["forcings_tgt"][:, 0])
        _, omega_node, _ = model.lie(model.graphs, hist, nf)
        plain = model.lie.rotate(model.graphs, hist[-1], omega_node)
        full = model.structural(hist, nf)[0]
        with causes.knocked_out(model, ["composite"]):
            off = model.structural(hist, nf)[0]
    assert torch.allclose(off.h, plain.h, atol=1e-6) and torch.allclose(off.v, plain.v, atol=1e-6)
    assert not torch.allclose(full.h, plain.h, atol=1e-4)


def test_error_decomposition(setup):
    _, data, stats, model = setup
    res = causes.error_decomposition(model, data, [0, 1, 2], stats, lead_steps=2,
                                     variants=["no_denoiser", "composite", "shift"], batch_size=2)
    assert res["inits"] == 3 and res["lead_h"] == [6, 12]
    rmse = {k: np.array(v) for k, v in res["rmse"].items()}
    assert rmse["full"].shape == (2, len(res["variables"]))
    # triangle inequality of the exact split total = rec + dyn
    assert (rmse["full"] <= rmse["rec"] + rmse["dyn"] + 1e-5).all()
    share = np.array(res["rec_share"])
    assert np.isfinite(share).all() and (share >= 0).all()
    for name in ("no_denoiser", "composite", "shift"):
        ch = {k: np.array(v) for k, v in res["variant_change"][name].items()}
        assert (ch["lo"] <= ch["mean"] + 1e-9).all() and (ch["mean"] <= ch["hi"] + 1e-9).all()
        assert np.abs(ch["mean"]).max() > 0  # the part matters in this random model
    # the same noise for every variant: running twice gives the same numbers
    again = causes.error_decomposition(model, data, [0, 1, 2], stats, lead_steps=2,
                                       variants=["composite"], batch_size=2)
    assert np.allclose(again["rmse"]["full"], res["rmse"]["full"])
    assert np.allclose(again["variant_change"]["composite"]["mean"],
                       res["variant_change"]["composite"]["mean"])
