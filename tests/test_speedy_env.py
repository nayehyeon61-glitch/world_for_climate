import numpy as np

from climate_wm.sim.speedy import (GAMMA, GRAV, R_D, Intervention, Regridder, WB2_LAT,
                                   mean_sea_level_pressure, omega_sigma, to_pressure)

SIGMA = np.array([0.025, 0.095, 0.2, 0.34, 0.51, 0.685, 0.835, 0.95])
LAT = np.linspace(-87.16, 87.16, 48)


def test_to_pressure_interpolates_and_extrapolates():
    ps = np.full((1, 4, 6), 1.0e5)
    ps[0, 0, 0] = 8.0e4                       # a mountain column
    x = np.log(SIGMA[None, :, None, None] * ps[:, None])   # linear in log p
    out = to_pressure(x, ps, SIGMA, [500, 850], "u")
    assert np.allclose(out[0, 0, 1:, 1:], np.log(5.0e4))
    # below the lowest level (0.95 * 800 hPa = 760 hPa) other fields are held constant
    assert np.isclose(out[0, 1, 0, 0], x[0, -1, 0, 0])
    t = np.full((1, 8, 4, 6), 280.0)
    tt = to_pressure(t, ps, SIGMA, [1000], "t")
    p_low = 0.95 * 8.0e4
    assert np.isclose(tt[0, 0, 0, 0], 280.0 * (1.0e5 / p_low) ** (R_D * GAMMA / GRAV))
    z = to_pressure(np.zeros_like(t), ps, SIGMA, [1000], "z", t_low=t[:, -1])
    assert z[0, 0, 0, 0] < 0                  # 1000 hPa lies below the column's lowest level


def test_omega_vanishes_for_nondivergent_flow():
    u = np.ones((1, 8, 48, 96)) * 10.0 * np.cos(np.deg2rad(LAT))[None, None, :, None]
    v = np.zeros_like(u)
    ps = np.full((1, 48, 96), 1.0e5)
    w = omega_sigma(u, v, ps, LAT)
    assert np.abs(w).max() < 1e-6


def test_regridder_preserves_constants_and_mslp_at_sea_level():
    rg = Regridder(LAT)
    f = np.full((2, 48, 96), 3.0)
    g = rg(f)
    assert g.shape == (2, len(WB2_LAT), 64) and np.allclose(g, 3.0)
    ps = np.full((48, 96), 1.0e5)
    assert np.allclose(mean_sea_level_pressure(ps, ps * 0 + 280.0, 0.95 * ps, ps * 0), ps)


def test_intervention_namelist():
    iv = Intervention(solar_factor=1.02, patches=((240.0, 0.0, 15.0, 2.0),))
    text = iv.namelist()
    assert "n_patch = 1" in text and "patch_amp(1) = 2.0" in text
    assert Intervention().is_null() and not iv.is_null()
