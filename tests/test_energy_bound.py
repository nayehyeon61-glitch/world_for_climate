import torch

from climate_wm.eval.energy import diagnose_energy, set_energy
from climate_wm.models.physics_world_model import CP, GRAV, L_V
from test_physics_core import _model, _tiny


def _flux_and_work(model, out, u, v):
    dp = model.dp_layer[None, :, None, None]
    flux = (CP * out["Q"] * dp).sum(1) / GRAV + (L_V * out["S"] * dp).sum(1) / GRAV
    work = ((out["F_u"] * u + out["F_v"] * v) * dp).sum(1) / GRAV
    return flux, work


def test_energy_bound_clamps_flux_and_work_and_tracks():
    layout, stats, data = _tiny()
    model = _model(layout, stats, data)
    torch.manual_seed(1)
    with torch.no_grad():  # a column with a large energy input (the fresh column's last layer is zero)
        for m in model.column.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.5)
                torch.nn.init.normal_(m.bias, std=0.5)
    batch = {k: v[None] for k, v in data[0].items()}
    state, ctx = model.begin(batch)
    forc = model.forcing_grid_at(ctx, 0)
    x, u, v = model.column_inputs(state, ctx["static_g"], forc)
    free = model.column_forcing(state, ctx["static_g"], forc)
    flux0, work0 = _flux_and_work(model, free, u, v)
    assert flux0.abs().max() > 10.0  # something to clamp
    set_energy(model, {"energy_max_flux": 10.0, "energy_no_work": True})
    model.energy_track = []
    bound = model.column_forcing(state, ctx["static_g"], forc)
    flux, work = _flux_and_work(model, bound, u, v)
    assert (flux.abs() <= 10.0 + 1e-3).all() and (work <= 1e-4).all()
    # the vertical structure of Q is kept: the correction is uniform over the column
    d = free["Q"] - bound["Q"]
    assert torch.allclose(d, d.mean(1, keepdim=True).expand_as(d), atol=1e-8)
    assert len(model.energy_track) == 1 and model.energy_track[0].shape == (1, 6)
    assert 0 < model.energy_track[0][0, 5] <= 1  # the clamped fraction
    set_energy(model, {})
    model.energy_track = None
    assert model.cfg.column.energy_max_flux is None and not model.cfg.column.energy_no_work


def test_diagnose_energy_records_terms_and_blowups(tmp_path):
    layout, stats, data = _tiny()
    model = _model(layout, stats, data)
    res = diagnose_energy(model, data, [0, 1], 2, {"none": {}, "flux500": {"energy_max_flux": 500.0}},
                          kinds=("none", "region"), progress_path=str(tmp_path / "e.pt"))
    r = res["runs"]["region:flux500"]
    assert set(res["runs"]) == {"none:none", "none:flux500", "region:none", "region:flux500"}
    assert len(r["err"]) == 2 and len(r["energy"]) == 2 and len(r["energy"][0]) == 6
    assert r["blowups"] == 0 and "err_ratio_finite" in r and r["finite_both"] == 2
    assert model.energy_track is None and model.cfg.column.energy_max_flux is None
