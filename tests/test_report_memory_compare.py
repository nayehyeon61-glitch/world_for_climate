import json
import os
import subprocess
import sys

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "report_memory_compare.py")
VARS = ["geopotential@500", "temperature@850", "2m_temperature", "10m_u_component_of_wind"]


def _file(path, scale, gc=1.0, leads=(6, 12, 18, 24, 36, 72, 120)):
    rows = lambda s: [{"variable": v, "lead_h": h, "rmse": s * h * (1 + i), "rmse_trad": s * h, "acc": 0.9}
                      for i, v in enumerate(VARS) for h in leads]
    with open(path, "w") as f:
        json.dump({"model": rows(scale), "graphcast": rows(gc), "persistence": rows(5.0), "climatology": rows(6.0)}, f)
    return str(path)


def test_report(tmp_path):
    cold = _file(tmp_path / "cold.json", 2.0)
    n1 = _file(tmp_path / "n1.json", 2.2)
    _file(tmp_path / "cm_warm.json", 1.8)            # 10% below the cold start at every lead
    _file(tmp_path / "cm_warm_no_memory.json", 2.0)  # the spin-up alone changes nothing
    with open(tmp_path / "cm_summary.json", "w") as f:
        json.dump({"inits": 31, "variants": {"warm": {"blowups": 1}, "warm_no_memory": {"blowups": 0}}}, f)
    cold18 = _file(tmp_path / "cold18.json", 0.5)
    _file(tmp_path / "obs_warm.json", 1.5, gc=1.0)     # on the cap: 1.5x GraphCast at every lead
    _file(tmp_path / "obs_nudging.json", 2.0, gc=1.0)  # 25% below nudging
    with open(tmp_path / "obs_summary.json", "w") as f:
        json.dump({"inits": 31, "region_fraction": 0.25, "variants": {"warm": {"blowups": 2, "hidden_fraction": 0.25}}}, f)
    out = tmp_path / "compare.md"
    subprocess.run([sys.executable, SCRIPT, "--cold2020", f"memory={cold}", f"n1={n1}",
                    "--warm2020", str(tmp_path / "cm.json"), "--cold2018", f"memory={cold18}",
                    "--obs2020", str(tmp_path / "obs.json"), "--out", str(out)],
                   check=True, capture_output=True)
    text = out.read_text()
    assert "| z500 | 24 h | 24 | 52.8 | 48 | 43.2 | 48 | 120 | 144 |" in text
    assert "| 24 h | -10.0% | +0.0% | -10.0% |" in text
    assert "supported (warm vs cold -10.0%, spin-up alone +0.0%)" in text
    assert "the gain stays above 1% to 120 h" in text
    assert "memory cold: 24 h 2.00x, 72 h 2.00x, 120 h 2.00x" in text
    assert "| z500 | 24 h | 193 | 333 | 365 | 51 | 24 | 12 |" in text  # ClimODE row with GraphCast and ours (2018)
    assert "not supported (memory cold below ClimODE at 24 h on 1 of 4 variables)" in text  # 12 < 193 only
    assert "warm 1, warm_no_memory 0 of 31" in text
    assert "| z500 | 24 h | 24 | 48 | 36 | 1.50x | 0.75x |" in text
    assert "supported (memory / nudging on the cap: 6 h 0.75x, 24 h 0.75x, 72 h 0.75x, 120 h 0.75x; blow-ups" in text
    assert "no crossing within the forecast (memory / GraphCast on the cap: 6 h 1.50x" in text
