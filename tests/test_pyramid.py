import numpy as np
import torch

from climate_wm.geometry import spherical_mesh
from climate_wm.geometry.group_actions import killing_field, so3_exp
from climate_wm.geometry.shells import PatchPyramid, SpherePatches


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
