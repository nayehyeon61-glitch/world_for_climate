import math

import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig
from climate_wm.train.trainer import StageConfig, train_stage

TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 1, "patch_splits": 0},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1},
    "transition": "lie_program",
}


def _setup(codebook_size, coupling="independent", target_steps=1, physics_codes=0):
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=3, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=target_steps)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=target_steps)
    batch = torch.utils.data.default_collate([data[0], data[1]])
    torch.manual_seed(0)
    cfg = dict(TINY, theory={"codebook_size": codebook_size, "max_steps": 2,
                             "physics_codes": physics_codes, "physics_dim": 4},
               lie=dict(TINY["lie"], shell_coupling=coupling))
    model = ClimateWorldModel(WorldModelConfig.from_dict(cfg), layout, data.lat, data.lon)
    return model, batch, data


def test_codebook_frames_and_identity_code():
    model, batch, _ = _setup(8)
    cb = model.theory.codebook
    frames = cb.frames
    eye = torch.einsum("pij,pkj->pik", frames, frames)  # rows are orthonormal
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-5)
    omega = torch.randn(2, 3, cb.frames.shape[0], 3)
    assert torch.allclose(cb.to_world(cb.to_local(omega)), omega, atol=1e-5)
    assert torch.all(cb.embedding[0] == 0)  # code 0 = stay


def test_explain_selects_lengths_and_executes_transfer():
    for coupling in ("independent", "chain"):
        model, batch, _ = _setup(8, coupling)
        th = model.theory
        with torch.no_grad():
            s = model.encode(batch["inputs"][:, -1], batch["forcings_in"][:, -1], batch["static"])
            s_y = model.encode_target(batch, 0)
            nf = model.node_forcing(batch["forcings_tgt"][:, 0])
            exp = th.explain(model.graphs, s, s_y, nf)
        K, P = th.cfg.max_steps, model.graphs.patches.num_patches
        assert exp.program.shape == (K, 2, 3, P, 3) and exp.indices.shape == (K, 2, 3, P)
        assert exp.lengths.min() >= 0 and exp.lengths.max() <= K
        assert exp.patch_error.shape == (K + 1, 2, 3, P)
        # MDL picks the per-patch argmin of lambda^k * error (lambda = 1 here)
        assert torch.equal(exp.lengths, exp.patch_error.argmin(0))
        # the assembled state equals state k* on every node
        hard = model.graphs.patches.hard
        for b in range(2):
            for sh in range(3):
                for n in range(hard.shape[0]):
                    k = int(exp.lengths[b, sh, hard[n]])
                    assert torch.allclose(exp.selected.h[b, sh, n], exp.states[k].h[b, sh, n])
        # executing the inferred program on the same state reproduces the states
        with torch.no_grad():
            replay = th.execute(model.graphs, s, exp.program, torch.full_like(exp.lengths, K))
        assert torch.allclose(replay.h, exp.states[K].h, atol=1e-5)
        # the identity program leaves the state unchanged
        with torch.no_grad():
            still = th.execute(model.graphs, s, torch.zeros_like(exp.program), exp.lengths)
        assert torch.allclose(still.h, s.h, atol=1e-5)


def test_theory_stage_trains_only_the_programmer():
    for size in (0, 8):
        model, batch, _ = _setup(size)
        frozen = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        stage = StageConfig(name="theory", steps=2, lr=1e-3,
                            weights={"rec": 1.0, "latent": 0.1, "vq": 1.0, "entropy": 0.1})
        hist = train_stage(model, [batch], stage, log_every=0)
        assert len(hist) == 2 and all(torch.isfinite(torch.tensor(h["loss"])) for h in hist)
        assert "mean_len" in hist[0] and ("codes_used" in hist[0]) == (size > 0)
        assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in frozen.items())
        if size:
            cb = model.theory.codebook
            assert torch.isfinite(cb.embedding).all() and torch.all(cb.embedding[0] == 0)


def test_transfer_evaluation_runs_end_to_end():
    from climate_wm.eval import transfer

    model, _, data = _setup(8, "chain", target_steps=2)
    K, idx = model.theory.cfg.max_steps, range(len(data))
    pairs = transfer.collect_pairs(model, data, idx, 0, batch_size=3)
    assert len(pairs) == len(data) and pairs.lead_hours == 6.0
    exp = transfer.explain_all(model, pairs, batch_size=3)
    assert exp.loss_by_length.shape == (len(data), K + 1) and exp.indices.shape[0] == K
    se = transfer.self_explainability(exp, 8)
    assert se["pairs"] == len(data) and len(se["code_usage"]) == 8
    assert len(se["length_histogram"]) == K + 1 and abs(sum(se["length_histogram"]) - 1) < 1e-5
    analogs = transfer.find_analogs(pairs, min_gap_hours=12.0)
    ok = analogs >= 0
    assert analogs.shape == (len(data),) and ok.any()
    gap = abs(pairs.times[ok.numpy()] - pairs.times[analogs[ok].numpy()])
    assert (gap >= 12 * 3600).all()  # never itself, never a neighbour in time
    tr = transfer.analog_transfer(model, pairs, exp, analogs, batch_size=3)
    assert tr["pairs"] == int(ok.sum())
    assert all(math.isfinite(v) for v in tr.values() if isinstance(v, float))
    pairs12 = transfer.collect_pairs(model, data, idx, 1, batch_size=3)
    assert pairs12.lead_hours == 12.0
    ood = transfer.length_ood(model, pairs12, [K, 2 * K], batch_size=3)
    assert set(ood["by_steps"]) == {str(K), str(2 * K)}
    md = transfer.markdown({"experiment": "tiny", "lead_hours": 6.0, "self_explainability": se,
                            "analog_transfer": tr, "length_ood": ood})
    assert "Analog transfer" in md and "Length OOD" in md


def test_physics_executor_is_column_local_and_conservative():
    from climate_wm.models.latent import Latent
    from climate_wm.models.physics_codes import PhysicsExecutor, code_family, family_weights

    torch.manual_seed(0)
    B, S, N, Cs, Cv, F, St, dz = 2, 4, 7, 6, 3, 5, 2, 4
    ex = PhysicsExecutor(F, St, Cs, Cv, hidden=16, hidden_layers=1, code_dim=dz, scale=1.0)
    for m in ex.modules():  # the executor starts at zero; give it something to do
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.3)
    z = Latent(torch.randn(B, S, N, Cs), torch.randn(B, S, N, Cv, 3))
    e, nf, ns = torch.randn(B, S, N, dz), torch.randn(B, N, F), torch.randn(B, N, St)
    fam = lambda f: torch.nn.functional.one_hot(torch.full((B, S, N), f), 3).float()
    # identity weights leave the state unchanged
    out = ex(z, e, torch.zeros(B, S, N, 3), nf, ns)
    assert torch.equal(out.h, z.h) and torch.equal(out.v, z.v)
    # radiation changes scalars only
    out = ex(z, e, fam(0), nf, ns)
    assert not torch.allclose(out.h, z.h) and torch.equal(out.v, z.v)
    # mixing conserves the column sum of scalars and vectors
    out = ex(z, e, fam(1), nf, ns)
    assert not torch.allclose(out.h, z.h)
    assert torch.allclose((out.h - z.h).sum(1), torch.zeros(B, N, Cs), atol=1e-5)
    assert torch.allclose((out.v - z.v).sum(1), torch.zeros(B, N, Cv, 3), atol=1e-5)
    # the surface code only exchanges between shells 0 and 1
    out = ex(z, e, fam(2), nf, ns)
    d = out.h - z.h
    assert torch.allclose(d[:, 2:], torch.zeros_like(d[:, 2:])) and not torch.allclose(d[:, 0], torch.zeros_like(d[:, 0]))
    assert torch.allclose(d[:, 0] + d[:, 1], torch.zeros_like(d[:, 0]), atol=1e-6)
    # column locality: changing another node leaves this node's tendency unchanged
    z2 = Latent(z.h.clone(), z.v.clone())
    z2.h[:, :, 3] += 1.0
    out1, out2 = ex(z, e, fam(1), nf, ns), ex(z2, e, fam(1), nf, ns)
    assert torch.allclose((out1.h - z.h)[:, :, :3], (out2.h - z2.h)[:, :, :3], atol=1e-6)
    # families split the codebook in three equal blocks, identity has none
    idx = torch.arange(7)
    assert code_family(idx, 6).tolist() == [-1, 0, 0, 1, 1, 2, 2]
    assert torch.equal(family_weights(idx, 6)[0], torch.zeros(3))


def test_physics_only_arm_has_no_transport_and_no_vq_gradient():
    model, batch, _ = _setup(1, "chain", physics_codes=6)  # identity-only Lie vocabulary
    th = model.theory
    with torch.no_grad():
        s = model.encode(batch["inputs"][:, -1], batch["forcings_in"][:, -1], batch["static"])
        s_y = model.encode_target(batch, 0)
        nf = model.node_forcing(batch["forcings_tgt"][:, 0])
        exp = th.explain(model.graphs, s, s_y, nf)
    assert torch.all(exp.program == 0) and torch.all(exp.indices == 0)
    assert float(exp.vq_loss) >= 0 and abs(th.summary(exp)["phys_share"] - 1.0) < 1e-6
    stage = StageConfig(name="theory", steps=3, lr=1e-2,
                        weights={"rec": 1.0, "latent": 1.0, "vq": 1.0, "entropy": 0.1})
    hist = train_stage(model, [batch], stage, log_every=0)
    assert all(h["vq"] < 1.0 and torch.isfinite(torch.tensor(h["loss"])) for h in hist)


def test_hybrid_program_explains_trains_and_transfers():
    from climate_wm.eval import transfer

    model, batch, data = _setup(8, "chain", target_steps=2, physics_codes=6)
    th = model.theory
    assert th.physics is not None and th.phys_codebook.size == 7
    with torch.no_grad():
        s = model.encode(batch["inputs"][:, -1], batch["forcings_in"][:, -1], batch["static"])
        s_y = model.encode_target(batch, 0)
        nf = model.node_forcing(batch["forcings_tgt"][:, 0])
        ns = model.graphs.grid_to_latent_mean(batch["static"])
        exp = th.explain(model.graphs, s, s_y, nf, ns=ns)
    K, P = th.cfg.max_steps, model.graphs.patches.num_patches
    assert exp.phys_indices.shape == (K, 2, 3, P) and exp.sub_error.shape == (K, 2, 3, P)
    assert exp.phys_gate.shape == (K, 2, 3, P) and float(exp.phys_gate.mean()) < 0.1  # starts closed
    assert "phys_frac" in th.summary(exp) and "phys_share" in th.summary(exp)
    with torch.no_grad():  # replaying Lie + physics codes (with their gates) reproduces the final state
        replay = th.execute(model.graphs, s, exp.program, torch.full_like(exp.lengths, K),
                            exp.phys_indices, nf, ns, exp.phys_gate)
    assert torch.allclose(replay.h, exp.states[K].h, atol=1e-5)
    # the theory stage trains with physics codes and logs their statistics
    stage = StageConfig(name="theory", steps=2, lr=1e-3,
                        weights={"rec": 1.0, "latent": 1.0, "vq": 1.0, "entropy": 0.1})
    hist = train_stage(model, [batch], stage, log_every=0)
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in hist) and "phys_frac" in hist[0]
    # the evaluation carries physics codes through explanation and transfer
    pairs = transfer.collect_pairs(model, data, range(len(data)), 0, batch_size=3)
    assert pairs.ns is not None
    ex = transfer.explain_all(model, pairs, batch_size=3)
    lat = torch.rad2deg(torch.asin(model.graphs.patches.centres[:, 2].clamp(-1, 1)))
    se = transfer.self_explainability(ex, 8, lat)
    assert "phys_share" in se and len(se["phys_usage"]["per_shell"]) == 3
    tr = transfer.analog_transfer(model, pairs, ex, transfer.find_analogs(pairs, 12.0), batch_size=3)
    assert all(math.isfinite(v) for v in tr.values() if isinstance(v, float))
    # intervention tests: forcing counterfactual and family knock-outs
    ca = transfer.causal_tests(model, pairs, ex, lat, 0, -1.5, {"t2m": 0}, batch_size=3)
    assert set(ca["knockout"]) == {"radiation", "mixing", "surface"}
    assert "t2m" in ca["counterfactual"]["decoded_channel_change"]
    assert math.isfinite(ca["counterfactual"]["effect"]["all"])
    md = transfer.markdown({"experiment": "t", "lead_hours": 6.0, "self_explainability": se,
                            "analog_transfer": tr, "causal": ca})
    assert "physics codes" in md and "knocked-out family" in md
