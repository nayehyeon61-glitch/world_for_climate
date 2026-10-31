import numpy as np
import pytest
import torch

from climate_wm.eval import physics_causes as pc
from test_physics_core import _model, _tiny

MODEL = {"core": {"trunc": 5, "nlat": 8, "nlon": 16, "dt_minutes": 60.0, "surface_drag_days": 1.0},
         "column": {"enabled": True, "hidden": 16, "layers": 1, "call_minutes": 120.0}}


@pytest.fixture(scope="module")
def setup():
    layout, stats, data = _tiny()
    model = _model(layout, stats, data, column=True)
    torch.manual_seed(1)
    with torch.no_grad():  # a column that does something (the zero init would hide every cut)
        model.column[-1].weight.normal_(0, 0.05)
        model.surface[-1].weight.normal_(0, 0.05)
    return layout, stats, data, model.state_dict()


def test_interventions_cut_their_paths(setup):
    layout, stats, data, state = setup
    batch = {k: v[None] for k, v in data[0].items()}
    intact = pc.build(MODEL, state, layout, data.lat, data.lon, None)
    full = pc.rollout(intact, batch, 2)
    assert torch.isfinite(full).all()
    for name in ("no_temp_advection", "no_moisture", "no_heating", "q_to_heating", "sun_off", "sun_to_heating",
                 "ocean_column", "no_drag", "no_carry"):
        iv = pc.intervention(name)
        m = pc.build(MODEL, state, layout, data.lat, data.lon, iv)
        assert pc.available(name, intact)
        pred = pc.rollout(m, batch, 2, sun_off=iv.sun)
        assert torch.isfinite(pred).all() and not torch.allclose(pred, full), name
    # the path cut keeps the other outputs: only the heating differs at the first call
    m = pc.build(MODEL, state, layout, data.lat, data.lon, pc.intervention("q_to_heating"))
    m0 = pc.build(MODEL, state, layout, data.lat, data.lon, None)
    st, ctx = m0.begin(batch)
    f0 = m0.column_forcing(st, ctx["static_g"], None if not m0.cfg.column.use_forcings else
                           m0._forcing_grid(ctx["init"][:1] + 3600)[None, 0])
    f1 = m.column_forcing(st, ctx["static_g"], None if not m.cfg.column.use_forcings else
                          m._forcing_grid(ctx["init"][:1] + 3600)[None, 0])
    assert not torch.allclose(f0["Q"], f1["Q"])
    for key in ("S", "F_u", "F_v", "P_b", "precip"):
        assert torch.allclose(f0[key], f1[key]), key
    # pairs merge their cuts; N0 has nothing for the column to cut
    both = pc.intervention("sun_off+q_to_heating")
    assert both.sun and both.paths == (("heating", ("q",)),)
    n0 = pc.build(dict(MODEL, column={"enabled": False}), state, layout, data.lat, data.lon, None)
    assert not pc.available("sun_off", n0) and pc.available("no_pgf", n0)


def test_physics_causes_evaluation(setup, tmp_path):
    layout, stats, data, state = setup
    names = ["no_temp_advection", "sun_off", "no_heating", "sun_off+no_heating", "no_coriolis"]
    path = str(tmp_path / "causes.progress")
    res = pc.evaluate_physics_causes(MODEL, state, stats, data, [0, 1], names=names, steps=2,
                                     progress_path=path)
    assert res["leads_h"] == [6, 12] and set(res["interventions"]) == {"full", *names}
    s = res["interventions"]["sun_off+no_heating"]
    p = s["pair"]
    assert p["a"] == "sun_off" and len(p["interaction_all"]["mean"]) == 2
    assert set(p["eta"]) == {"thermodynamics", "motion", "water"}
    r = res["interventions"]["sun_off"]["response"]
    assert np.isfinite(r["thermodynamics"]["mean"]).all() and r["thermodynamics"]["mean"][0] > 0
    assert "t2m_day_K" in r and len(r["t2m_corr_energy"]["mean"]) == 2
    # resume: everything is read back
    again = pc.evaluate_physics_causes(MODEL, state, stats, data, [0, 1], names=names, steps=2,
                                       progress_path=path)
    assert again["interventions"]["sun_off"]["change_all"] == res["interventions"]["sun_off"]["change_all"]
