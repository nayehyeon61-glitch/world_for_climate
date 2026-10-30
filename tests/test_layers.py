import torch
from scipy.spatial.transform import Rotation

from climate_wm.geometry import spherical_mesh as sm
from climate_wm.geometry.tangent_frames import project_tangent
from climate_wm.models.layers import GeoGraph, GeometricInteraction, ShellProcessor


def _graph(vertices, meshes, break_symmetry):
    s, r = sm.multimesh_edges(meshes)
    return GeoGraph(vertices, vertices, s, r, break_symmetry)


def test_geometric_interaction_is_so3_equivariant():
    torch.manual_seed(0)
    meshes = sm.icosphere_hierarchy(2)
    x = meshes[-1].vertices
    q = Rotation.random(random_state=1).as_matrix()
    g, g_rot = _graph(x, meshes, False), _graph(x @ q.T, meshes, False)
    qt = torch.tensor(q, dtype=torch.float32)
    layer = GeometricInteraction(16, 4)
    n = len(x)
    pos = torch.tensor(x, dtype=torch.float32)
    h = torch.randn(2, n, 16)
    v = project_tangent(torch.randn(2, n, 4, 3), pos.expand(2, -1, -1))
    e = torch.randn(1, g.num_edges, 16)
    h1, v1, _ = layer(g, h, v, h, v, e)
    h2, v2, _ = layer(g_rot, h, v @ qt.T, h, v @ qt.T, e)
    torch.testing.assert_close(h2, h1, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v2, v1 @ qt.T, atol=1e-4, rtol=1e-4)
    assert (v1 * pos[None, :, None]).sum(-1).abs().max() < 1e-5  # stays tangent


def test_shell_processor_equivariant_with_vertical_mixing():
    torch.manual_seed(0)
    meshes = sm.icosphere_hierarchy(2)
    x = meshes[-1].vertices
    q = Rotation.random(random_state=2).as_matrix()
    g, g_rot = _graph(x, meshes, False), _graph(x @ q.T, meshes, False)
    qt = torch.tensor(q, dtype=torch.float32)
    proc = ShellProcessor(16, 4, g.edge_dim, num_layers=2, num_shells=3, cond_dim=8)
    n = len(x)
    pos = torch.tensor(x, dtype=torch.float32)
    h = torch.randn(2, 3, n, 16)
    v = project_tangent(torch.randn(2, 3, n, 4, 3), pos.expand(2, 3, -1, -1))
    cond = torch.randn(2, 8)
    h1, v1 = proc(g, h, v, cond)
    h2, v2 = proc(g_rot, h, v @ qt.T, cond)
    torch.testing.assert_close(h2, h1, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v2, v1 @ qt.T, atol=1e-4, rtol=1e-4)
