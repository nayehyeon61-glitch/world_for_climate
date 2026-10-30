import numpy as np
import torch
from scipy.spatial.transform import Rotation

from climate_wm.geometry import group_actions as ga
from climate_wm.geometry import shells
from climate_wm.geometry import spherical_mesh as sm
from climate_wm.geometry import tangent_frames as tf


def test_icosphere_counts_and_nesting():
    meshes = sm.icosphere_hierarchy(3)
    assert [m.num_vertices for m in meshes] == [12, 42, 162, 642]
    assert [len(m.faces) for m in meshes] == [20, 80, 320, 1280]
    for coarse, fine in zip(meshes[:-1], meshes[1:]):
        np.testing.assert_allclose(coarse.vertices, fine.vertices[:coarse.num_vertices])
    np.testing.assert_allclose(np.linalg.norm(meshes[-1].vertices, axis=1), 1.0)
    s, r = sm.multimesh_edges(meshes)
    pairs = set(zip(s.tolist(), r.tolist()))
    assert all((b, a) in pairs for a, b in pairs)  # symmetric


def test_grid_connectivity():
    lat, lon = np.linspace(-90, 90, 19), np.arange(36) * 10.0
    _, _, grid = sm.grid_nodes(lat, lon)
    mesh = sm.icosphere_hierarchy(2)[-1]
    s, r = sm.radius_edges(grid, mesh.vertices, 0.6 * sm.max_edge_length(mesh))
    assert set(r.tolist()) == set(range(mesh.num_vertices))
    s, r, w = sm.mesh2grid_edges(grid, mesh)
    assert np.all(np.bincount(r) == 3)
    np.testing.assert_allclose(np.bincount(r, weights=w), 1.0, atol=1e-6)
    # Barycentric interpolation reproduces the point itself.
    recon = np.zeros_like(grid)
    np.add.at(recon, r, w[:, None] * mesh.vertices[s])
    recon /= np.linalg.norm(recon, axis=1, keepdims=True)
    np.testing.assert_allclose(recon, grid, atol=1e-6)


def test_frames_and_transport():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-80, 80, 50), rng.uniform(0, 360, 50)
    east, north = tf.east_north_frames(lat, lon)
    p = sm.lat_lon_to_xyz(lat, lon)
    np.testing.assert_allclose((east * p).sum(-1), 0, atol=1e-12)
    np.testing.assert_allclose((north * p).sum(-1), 0, atol=1e-12)
    u, v = torch.randn(50, 2), torch.randn(50, 2)
    e, n = torch.tensor(east), torch.tensor(north)
    u2, v2 = tf.xyz_to_uv(tf.uv_to_xyz(u.double(), v.double(), e, n), e, n)
    torch.testing.assert_close(u2, u.double())
    torch.testing.assert_close(v2, v.double())
    q = sm.lat_lon_to_xyz(lat + 3, lon + 2)
    t = tf.transport_matrices(p, q)
    np.testing.assert_allclose(np.einsum("eij,ej->ei", t, p), q, atol=1e-10)
    np.testing.assert_allclose(t @ np.transpose(t, (0, 2, 1)), np.eye(3)[None].repeat(50, 0), atol=1e-10)


def test_so3_exp_log():
    w = torch.randn(100, 3, dtype=torch.float64)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(100, 1, dtype=torch.float64) * 3.0  # |w| < pi
    r = ga.so3_exp(w)
    eye = torch.eye(3, dtype=torch.float64).expand(100, 3, 3)
    torch.testing.assert_close(r @ r.transpose(-1, -2), eye)
    torch.testing.assert_close(torch.det(r), torch.ones(100, dtype=torch.float64))
    torch.testing.assert_close(ga.so3_log(r), w)
    tiny = torch.randn(10, 3, dtype=torch.float64) * 1e-7
    torch.testing.assert_close(ga.so3_log(ga.so3_exp(tiny)), tiny)
    # exp(hat w) agrees with scipy.
    ref = Rotation.from_rotvec(w.numpy()).as_matrix()
    np.testing.assert_allclose(r.numpy(), ref, atol=1e-10)


def test_rotation_action_identity_and_rotation():
    mesh = sm.icosphere_hierarchy(3)[-1]
    act = ga.SphereRotationAction(mesh)
    n = mesh.num_vertices
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    h = torch.randn(2, n, 4)
    v = tf.project_tangent(torch.randn(2, n, 3, 3), pos.expand(2, -1, -1))
    h2, v2 = act(h, v, ga.so3_exp(torch.zeros(2, n, 3)))
    torch.testing.assert_close(h2, h, atol=1e-5, rtol=0)
    torch.testing.assert_close(v2, v, atol=1e-5, rtol=0)
    # Smooth field rotated about z: compare with the analytic rotation.
    lat, lon = sm.xyz_to_lat_lon(mesh.vertices)
    f = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    angle = 0.1
    rot = ga.so3_exp(torch.tensor([0., 0., angle]).expand(1, n, 3))
    fr, _ = act(torch.tensor(f, dtype=torch.float32)[None, :, None],
                torch.zeros(1, n, 1, 3), rot)
    exact = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon) - angle)
    assert np.abs(fr[0, :, 0].numpy() - exact).max() < 1e-2
    # Gradients reach the Lie algebra element.
    w = torch.full((1, n, 3), 0.02, requires_grad=True)
    out, _ = act(torch.tensor(f, dtype=torch.float32)[None, :, None], torch.zeros(1, n, 1, 3),
                 ga.so3_exp(w))
    out.sum().backward()
    assert w.grad.abs().sum() > 0


def test_killing_projection_recovers_rigid_rotation():
    mesh = sm.icosphere_hierarchy(3)[-1]
    patches = shells.SpherePatches(mesh.vertices, patch_splits=1)
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    w = torch.tensor([0.03, -0.02, 0.01])
    u = ga.killing_field(w.expand_as(pos), pos)[None]
    omega = patches.killing_projection(u)[0]
    c = patches.centres
    perp = lambda x: x - (x * c).sum(-1, keepdim=True) * c
    err = (perp(omega) - perp(w.expand_as(omega))).norm(dim=-1) / w.norm()
    assert err.max() < 0.02  # the advective part is recovered (spin is a separate head)
    blended = patches.blend(w.expand(1, patches.num_patches, 3))
    torch.testing.assert_close(blended[0], w.expand(mesh.num_vertices, 3))


def test_shell_spec_default_grouping():
    spec = shells.ShellSpec.from_levels(
        (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000))
    assert [s.name for s in spec.shells] == [
        "surface", "boundary_layer", "lower_troposphere", "upper_troposphere", "stratosphere"]
    assert spec.shells[1].levels == (1000, 925, 850)
    assert spec.shells[4].levels == (150, 100, 50)
