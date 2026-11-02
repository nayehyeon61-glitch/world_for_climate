import json
import os
import sys

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.eval import topology as topo
from climate_wm.geometry.shells import SpherePatches
from climate_wm.models.commute_time import CommuteConfig, embed_all, fit_head
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

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


def _ring_walk(n=24, steps=6000, seed=0):
    gen = np.random.default_rng(seed)
    s = np.cumsum(gen.choice([-1, 1], steps)) % n
    return s


def test_markov_commute_times_and_spectrum():
    a, b = 0.1, 0.25
    counts = np.array([[(1 - a) * 1000, a * 1000], [b * 1000, (1 - b) * 1000]])
    ct, keep = topo.commute_times(counts)
    assert keep.all() and np.isclose(ct[0, 1], 1 / a + 1 / b, rtol=1e-6) and np.isclose(ct[0, 0], 0)
    n = 12
    ring = np.zeros((n, n))
    for i in range(n):
        ring[i, (i + 1) % n] = ring[i, (i - 1) % n] = 1.0
    ct, _ = topo.commute_times(ring)
    assert np.allclose([ct[0, k] for k in range(n)], [2 * k * (n - k) for k in range(n)])  # gambler's ruin
    lam = topo.laplacian_spectrum(ring, np.ones(n, bool))
    assert np.isclose(lam[0], 1 - np.cos(2 * np.pi / n))
    # an unreachable cluster is dropped, not given an infinite commute time
    ring3 = np.zeros((n + 1, n + 1))
    ring3[:n, :n] = ring
    ring3[n, 0] = 1.0
    ct, keep = topo.commute_times(ring3)
    assert not keep[n] and np.isnan(ct[n]).all() and np.isfinite(ct[:n, :n]).all()


def test_commute_head_recovers_a_ring():
    n = 24
    s = _ring_walk(n)
    gen = np.random.default_rng(1)
    x = np.eye(n)[s] @ gen.normal(size=(n, 20)) + 0.05 * gen.normal(size=(len(s), 20))
    pairs = np.stack([np.arange(len(s) - 1), np.arange(1, len(s))], 1)
    k = np.minimum(np.arange(n), n - np.arange(n))
    truth = 2 * k * (n - k)  # commute time from node 0
    rhos = {}
    for reg in ("logdet", "vicreg"):
        head, hist = fit_head(x.astype(np.float32), pairs, None,
                              CommuteConfig(dim=4, hidden=32, steps=600, regulariser=reg, use_action=False))
        emb = embed_all(head, x)
        cent = np.stack([emb[s == i].mean(0) for i in range(n)])
        d = np.sqrt(((cent - cent[0]) ** 2).sum(-1))
        rhos[reg] = spearmanr(d[1:], truth[1:]).statistic
        assert np.isfinite(hist[-1]["loss"])
    assert rhos["logdet"] > 0.8, rhos


def test_helpers():
    t = np.arange(np.datetime64("2016-01-01T00"), np.datetime64("2017-01-01T00"), np.timedelta64(6, "h"))
    h = topo.harmonics(t)
    x = (3 + 2 * h[:, 1] - h[:, 8])[:, None] * np.ones((1, 4))
    fit = np.ones(len(t), bool)
    assert np.abs(topo.deseasonalise(x, t, fit)).max() < 1e-4
    gen = np.random.default_rng(0)
    blobs = np.concatenate([gen.normal(c, 0.1, (60, 2)) for c in ((0, 0), (5, 0), (0, 5))])
    lab, _ = topo.kmeans(blobs, 3)
    assert len(set(lab[:60])) == len(set(lab[60:120])) == len(set(lab[120:])) == 1 and len(set(lab)) == 3
    assert topo.h0_persistence(blobs)["excess"] >= 2
    assert topo.h0_persistence(gen.normal(size=(180, 2)))["excess"] <= 1
    slow = np.repeat(gen.normal(size=(20, 2)) * 5, 20, 0)
    assert topo.residence_time(slow, 4)["mean_run"] > topo.residence_time(gen.permutation(slow), 4)["mean_run"]
    ref = 1 + gen.random((100, 3))
    ch = topo.block_bootstrap_change(0.9 * ref, ref, 50, 10, 0)
    assert np.allclose(ch["mean"], -0.1) and np.allclose(ch["lo"], -0.1) and np.allclose(ch["hi"], -0.1)
    rows = np.ones(len(t), bool)
    rows[10] = False
    pairs = topo.transition_pairs(t, rows)
    assert len(pairs) == len(t) - 3 and not np.isin(10, pairs)
    lat, lon = np.linspace(-78.75, 78.75, 8), np.arange(16) * 22.5
    normal = 5600 * 9.80665 - 30 * 9.80665 * np.abs(lat)[:, None] * np.ones(16)
    blocked = normal.copy()
    blocked[[5, 6]] = normal[[6, 5]]  # reversed gradient between 34N and 56N at every longitude
    reg = topo.blocking_and_jet(np.stack([normal, blocked]), lat, lon)
    assert not reg["blocked"][0] and reg["blocked"][1]
    assert np.isclose(topo.coarsen(np.ones((2, 9, 16))).mean(), 1) and topo.coarsen(np.ones((9, 16))).shape == (4, 8)


@pytest.fixture(scope="module")
def tiny():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=4, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=1)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=1)
    torch.manual_seed(0)
    model = ClimateWorldModel(WorldModelConfig.from_dict(TINY), layout, data.lat, data.lon)
    model.set_normalization(stats)
    model.eval()
    return layout, ds, stats, data, model


def test_encode_states(tiny):
    layout, ds, stats, data, model = tiny
    patches = SpherePatches(model.graphs.latent_pos.cpu().numpy(), 0)
    store = topo.encode_states(model, data, stats, patches, batch_size=3)
    n, s = len(data), len(layout.shells)
    assert store["latent"].shape == (n, s, 12, 8 + 2) and np.isfinite(store["latent"]).all()
    assert store["action"].shape == (n, 12 + 2) and store["fields"].shape == (n, 3, 9, 16)
    frames, _, _ = data.raw_window(0)  # physical units, target order
    z500 = frames[data.input_steps - 1, [c.key for c in layout.targets].index("geopotential@500")]
    assert np.allclose(store["fields"][0, 0], z500, rtol=1e-4, atol=1e-1)
    # energy: 2 levels x (u, v, T) + mean sea level pressure on the 4 x 8 coarse grid
    assert store["energy"].shape == (n, 7 * 4 * 8)
    assert (np.diff(store["times"].astype(np.int64)) == 6 * 3600).all()
    wm = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    err = topo.world_model_errors(model, wm, stats, [0, 1], 2)
    assert err.shape == (2,) and np.isfinite(err).all() and (err > 0).all()


def _synthetic_store(seed=0):
    gen = np.random.default_rng(seed)
    t = np.arange(np.datetime64("2016-09-01T00"), np.datetime64("2018-03-01T00"), np.timedelta64(6, "h"))
    n = len(t)
    regime = np.repeat(gen.integers(0, 4, n // 20 + 1), 20)[:n]  # slow regimes
    base = gen.normal(size=(4, 72))[regime]
    ar = np.zeros((n, 72))
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + 0.3 * gen.normal(size=72)
    lat_feat = (base + ar).reshape(n, 2, 12, 3).astype(np.float32)
    lat, lon = np.linspace(-78.75, 78.75, 8), np.arange(16) * 22.5
    fields = (lat_feat.reshape(n, -1)[:, :48] @ gen.normal(size=(48, 3 * 8 * 16))).reshape(n, 3, 8, 16)
    fields[:, 0] += 5600 * 9.80665 - 300 * np.abs(lat)[:, None]
    centres = gen.normal(size=(12, 3))
    return {"times": t.astype("datetime64[s]"), "latent": lat_feat,
            "action": gen.normal(size=(n, 14)).astype(np.float32), "fields": fields.astype(np.float32),
            "energy": (lat_feat.reshape(n, -1)[:, :40] * 3).astype(np.float32),
            "patch_centres": centres / np.linalg.norm(centres, axis=1, keepdims=True), "lat": lat, "lon": lon,
            "wm_err": gen.random(12), "wm_rows": np.arange(len(t) - 300, len(t) - 12, 24)}


def test_analyse_and_report(tmp_path):
    import report_topology
    store = _synthetic_store()
    arms = {"psi_ct": CommuteConfig(dim=4, hidden=16, steps=40),
            "psi_patch": CommuteConfig(dim=2, hidden=16, steps=40, mode="patch")}
    res = topo.analyse(store, arms, n_clusters=8, leads=(1, 2), pca_dim=16, topo_dim=4, n_boot=20, log=lambda *_: None)
    assert set(res["arms"]) == {"raw_z", "energy", "psi_ct", "psi_patch"}
    ct = res["arms"]["psi_ct"]
    assert {"latent", "energy", "latent_minus_raw_z", "energy_minus_energy"} <= set(ct["t1"])
    assert -1 <= ct["t1"]["latent"]["mean"] <= 1 and np.isfinite(ct["t2"]["spectrum"]["slope"])
    assert np.asarray(ct["t3"]["rmse"]).shape == (2, 3) and "all" in ct["t3"]["vs_raw_z"]
    assert "speed_vs_error" in ct["predictability"]
    assert res["arms"]["psi_patch"]["dim"] == 24 and "psi_patch" in res["regional"]["north_atlantic"]
    res["config"] = {"topology": {"analogs": 10}}
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(res))
    sys.argv = ["report_topology.py", "--topology", str(path), "--out-dir", str(tmp_path)]
    report_topology.main()
    md = (tmp_path / "topology.md").read_text()
    assert "T1." in md and "psi_patch" in md and (tmp_path / "topology.png").exists()
