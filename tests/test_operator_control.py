import pytest
import torch

pytest.importorskip("neuralop")
pytest.importorskip("torch_harmonics")

from climate_wm.models.operator_control import OperatorBankController, OperatorControlConfig
from climate_wm.models.tendency_control import controller_state, load_controller
from climate_wm.train.physics_game import play_physics
from climate_wm.train.prediction_game import GameConfig, episode_masks, evaluate_game, train_controller
from test_physics_core import _model, _tiny


@pytest.fixture(scope="module")
def setup():
    layout, stats, data = _tiny()
    model = _model(layout, stats, data, column=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return layout, data, model


def _bank(model, **kw):
    cfg = OperatorControlConfig(max_degree=2, layers=((500, 500), (850, 850)), hidden=8, n_operators=2,
                                operator_hidden=4, operator_layers=1, operator_modes=4, out_degree=3, **kw)
    torch.manual_seed(0)
    return OperatorBankController(model, cfg)


def test_untrained_bank_does_nothing_and_choices_are_banded(setup):
    layout, data, model = setup
    ctrl = _bank(model)
    assert ctrl.n_options == 3 and ctrl.n_bands == 3
    assert torch.allclose(ctrl.band_w.sum(0), torch.ones(ctrl.band_w.shape[1]))
    batch = {k: v[None] for k, v in data[0].items()}
    gen = torch.Generator().manual_seed(0)
    masks, _ = episode_masks(layout, 1, GameConfig(), data.lat, data.lon, gen)
    h1, a1 = play_physics(model, ctrl, batch, masks, 2, obs_prob=0.5, gen=torch.Generator().manual_seed(1))
    h0, a0 = play_physics(model, ctrl, batch, masks, 2, control=False, obs_prob=0.5,
                          gen=torch.Generator().manual_seed(1))
    assert torch.allclose(h1, h0) and torch.allclose(a1, a0)
    # an operator with a nonzero output acts only through the bands that chose it
    with torch.no_grad():
        ctrl.operators[0].projection.fcs[-1].bias.fill_(1.0)
    state, _ = model.begin(batch)
    ctrl.set_mask(masks)
    x = ctrl.fields(state, 0)
    assert x.shape[1] == ctrl.n_in
    w = torch.zeros(1, 3, 3)
    w[0, 2, 0] = 1.0  # the northern band takes operator 0, the others nothing
    u = ctrl.control_field(x, w)
    north = ctrl.band_w[2] > 0.99
    south = ctrl.band_w[0] > 0.99
    assert u[0, :, north].abs().min() > 0.9 and u[0, :, south].abs().max() < 1e-2
    out, cost = ctrl.apply_field(state, u)
    assert cost[0] > 0 and not torch.allclose(out["temp"], state["temp"])
    assert torch.equal(out["div"], state["div"])
    # the choice log-probability sums over bands (uniform at the start)
    feats = ctrl.features(state, 0)
    lp = ctrl.log_prob(feats, torch.zeros(1, 3, dtype=torch.long))
    assert torch.allclose(lp, torch.tensor([3 * torch.log(torch.tensor(1 / 3))]))
    ctrl.set_mask(None)


@pytest.mark.parametrize("mode,selection", [("hybrid", "choice"), ("grad", "soft")])
def test_operator_bank_training_and_eval(setup, mode, selection):
    layout, data, model = setup
    ctrl = _bank(model, selection=selection)
    with torch.no_grad():  # start from a bank that acts, so both losses have something to move
        for op in ctrl.operators:
            op.projection.fcs[-1].weight.normal_(0, 0.1)
    ctrl.cost_weight = 1e-4
    before = {k: v.detach().clone() for k, v in ctrl.named_parameters()}
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = GameConfig(mode=mode, steps=2, batch_size=1, members=2, horizon=2, obs_prob=0.5, log_every=1,
                     warmup_steps=0, cap=10.0)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    hist = train_controller(model, ctrl, loader, cfg, play_fn=play_physics)
    assert len(hist) == 2 and all(torch.isfinite(torch.tensor(h["loss"])) for h in hist)
    if mode == "hybrid":
        assert "loss_rl" in hist[0] and "loss_grad" in hist[0]
    moved = lambda prefix: any(not torch.equal(before[k], v) for k, v in ctrl.named_parameters()
                               if k.startswith(prefix))
    assert moved("operators.") and moved("net.")
    assert all(torch.equal(frozen[k], v) for k, v in model.state_dict().items())
    # save / load, then the evaluation with the stage 11 knock-outs and usage
    ctrl2 = load_controller(model, controller_state(ctrl))
    assert isinstance(ctrl2, OperatorBankController)
    res = evaluate_game(model, ctrl2, data, [0, 1], ["region"], 2, obs_prob=0.5, play_fn=play_physics,
                        variants=["no_control", "random_control", "option_0", "option_2"], cap=10.0)
    r = res["kinds"]["region"]
    assert "option_2_hidden" in r["change"]
    if selection == "choice":
        share = r["usage"]["share"]
        assert len(share) == 3 and abs(sum(share[0]) - 1) < 1e-6
    # option 2 (= no control) equals the uncontrolled game
    assert torch.allclose(torch.tensor(r["mse"]["option_2_hidden"]), torch.tensor(r["mse"]["no_control_hidden"]))
