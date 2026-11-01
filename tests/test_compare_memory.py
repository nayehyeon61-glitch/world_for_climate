import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset  # noqa: E402
from climate_wm.data.synthetic import make_synthetic_era5  # noqa: E402
from compare_memory import STEP_S, SpinUpWindows, WarmStart  # noqa: E402
from test_dreamer_game import _mem_model  # noqa: E402
from test_physics_core import _tiny  # noqa: E402


def test_warm_start_pairs_the_windows_and_the_memory_changes_the_free_run():
    layout, stats, data = _tiny()
    ds = make_synthetic_era5(num_days=5, num_lat=9, num_lon=16, levels=(500, 850))
    short = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    long = GraphCastWindowDataset(ds, layout, stats, target_steps=5)
    joined = SpinUpWindows(short, long, 3)
    assert 0 < len(joined) < len(short)  # the first forecast windows have no spin-up
    item = joined[0]
    assert torch.allclose(item["spin_targets"][3:], item["targets"])
    assert int(item["spin_init_time"]) == int(item["init_time"]) - 3 * STEP_S
    assert joined.init_seconds()[0] == int(item["init_time"])

    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    torch.manual_seed(1)
    for p in model.memory.parameters():  # an untrained kernel does nothing
        p.data.normal_(0, 0.3)
    batch = {k: v[None] for k, v in item.items()}
    warm = WarmStart(model, 3).eval().rollout(batch, 2, members=2)
    assert warm.shape == (2, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(warm).all()
    assert torch.equal(warm[0], warm[1])
    knocked = WarmStart(model, 3, ("memory",)).rollout(batch, 2)
    assert knocked.shape == (1, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(knocked).all()
    assert not torch.allclose(warm[0], knocked[0])  # the filled memory acts on the free run
    assert model.memory.knock == set()  # restored
    with torch.no_grad():  # the same free run twice: the spin-up is deterministic
        assert torch.allclose(WarmStart(model, 3).rollout(batch, 2)[0], warm[0])
