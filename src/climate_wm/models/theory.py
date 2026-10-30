"""Learning-to-Theorize on the latent sphere with Lie-algebra primitives.

Following NEO (Baek et al., "Learning to Theorize the World from Observation",
ICML 2026) the 6 h change Z_t -> Z_{t+1} is not regressed in one shot but
*explained* by a short program tau = (z_1, ..., z_k) of discrete primitives:

    s_1 = Z_t,   s_{k+1} = f(s_k, z_k),   D(s_{k*+1}) ~ X_{t+1}

Here the primitives are elements of so(3): a shared codebook of small
rotations written in each patch's local (east, north, up) frame, so the same
code means the same motion everywhere on the sphere. Their executor is the
group action Phi_{exp(omega)} the model already uses: fixed, exactly
equivariant, isometric, and closed under composition (exp(a) exp(b) is a
rotation), so the executor needs no learning and NEO's state grounding is not
needed for these codes. Programs are a *field*: every (shell, patch) has its
own program over the shared codebook, and the MDL criterion picks the
shortest accurate length k* per patch (k* = 0 keeps the patch still).

The theory programmer q(z_k | s_k, s_y, c) is the Lie generator fed with the
current and the *target* latent (NEO's goal-conditioned policy). With
`codebook_size = 0` the proposals stay continuous: that is NEO's Cont-Mono
baseline, i.e. the current generator with a target input, and the control
for the VQ-Lie hypothesis. With `shell_coupling = chain` the codes are the
joint rotations of the shell chain (liquid-centipede reading): a code field
that is "still" above a shell means the column moves rigidly.

Programs inferred from one (support) pair can be executed on another (query)
state with `execute`, which is the transfer test of the hypothesis.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .graphs import WorldGraphs
from .latent import Latent
from .lie_generator import LieConfig, LieGenerator, compose
from .physics_codes import PhysicsExecutor, family_weights


@dataclasses.dataclass
class TheoryConfig:
    codebook_size: int = 16        # 0: continuous programs (Cont-Mono control)
    max_steps: int = 3             # K, the longest program during training
    lambda_mdl: float = 1.0        # > 1 prefers shorter explanations (NEO eq. 5)
    commitment: float = 0.25       # VQ commitment weight (inside the vq term)
    entropy_weight: float = 0.1    # NEO's codebook entropy regulariser
    entropy_temperature: float = 1.0
    ema_decay: float = 0.99
    dead_code_steps: int = 200     # re-seed a code unused for this many updates (0: off)
    code_scale: float = 0.5        # initial code magnitude in units of the rotation cap
    # Phase 2: column-local residual physics codes executed after the Lie step
    physics_codes: int = 0         # non-identity physics codes (0: Lie only); split in
                                   # three equal families: radiation, mixing, surface
    physics_dim: int = 8           # code embedding width (FiLM input of the executor)
    physics_scale: float = 0.1     # tendency scale in latent units per step


class VectorQuantiser(nn.Module):
    """EMA vector quantiser (NEO's VQ): code 0 is a fixed identity, the other
    codes are seeded from the first training batch and re-seeded when dead."""

    def __init__(self, size: int, dim: int, cfg: TheoryConfig, codes: Optional[Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.size, self.dim = size, dim
        if codes is None:
            codes = torch.zeros(size, dim)
            if size > 1:
                codes[1:] = 0.1 * torch.randn(size - 1, dim, generator=torch.Generator().manual_seed(0))
        self.register_buffer("embedding", codes)
        self.register_buffer("ema_weight", codes.clone())
        self.register_buffer("ema_size", torch.ones(size))
        self.register_buffer("seeded", torch.zeros(()))  # codes re-seeded from the first batch
        self.register_buffer("unused_steps", torch.zeros(size))

    def forward(self, local: Tensor, update: bool = True
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Quantises proposals [..., dim].

        Returns the straight-through quantised values, code indices, the
        per-element commitment loss and the entropy regulariser (scalar)."""
        flat = local.reshape(-1, self.dim).float()
        if self.training and update and not bool(self.seeded):
            self._seed(flat)
        emb = self.embedding.clone()  # snapshot: the EMA update below edits the buffer in place
        d2 = (flat.square().sum(1, keepdim=True) - 2 * flat @ emb.T + emb.square().sum(1)[None])
        idx = d2.argmin(1)
        quant = emb[idx]
        if self.training and update:
            self._ema_update(flat, idx)
        logits = -d2 / (self.cfg.entropy_temperature * max(float(emb.square().sum(1).mean()), 1e-6))
        probs = torch.softmax(logits, 1)
        sample_entropy = -(probs * torch.log(probs + 1e-6)).sum(1).mean()
        avg = probs.mean(0)
        codebook_entropy = -(avg * torch.log(avg + 1e-6)).sum()
        entropy = sample_entropy - codebook_entropy
        commitment = self.cfg.commitment * (flat - quant.detach()).square().sum(1)
        out = flat + (quant - flat).detach()  # straight-through estimator
        return (out.view_as(local), idx.view(local.shape[:-1]),
                commitment.view(local.shape[:-1]), entropy)

    @torch.no_grad()
    def _seed(self, flat: Tensor) -> None:
        """Places the non-identity codes on proposals of the first training batch
        (the usual data-dependent VQ initialisation), so that the vocabulary
        starts at the scale of real 6 h changes instead of the geometric cap."""
        n = self.size - 1
        if n <= 0 or flat.shape[0] == 0:
            self.seeded.fill_(1.0)
            return
        pick = flat[torch.randperm(flat.shape[0], device=flat.device)[:n]]
        if pick.shape[0] < n:  # fewer proposals than codes: repeat
            pick = pick.repeat((n + pick.shape[0] - 1) // pick.shape[0], 1)[:n]
        pick = pick + 1e-3 * pick.std(0, keepdim=True) * torch.randn_like(pick)
        self.embedding[1:] = pick
        self.ema_weight[1:] = pick
        self.ema_size.fill_(1.0)
        self.seeded.fill_(1.0)

    @torch.no_grad()
    def _ema_update(self, flat: Tensor, idx: Tensor) -> None:
        d = self.cfg.ema_decay
        onehot = F.one_hot(idx, self.size).float()
        counts = onehot.sum(0)
        self.ema_size.mul_(d).add_(counts, alpha=1 - d)
        self.ema_weight.mul_(d).add_(onehot.T @ flat, alpha=1 - d)
        n = self.ema_size.sum()
        size = (self.ema_size + 1e-5) / (n + self.size * 1e-5) * n  # Laplace smoothing
        self.embedding[1:].copy_((self.ema_weight / size[:, None])[1:])  # code 0 stays the identity
        self.unused_steps.add_(1.0).mul_((counts == 0).float())
        if self.cfg.dead_code_steps:
            dead = self.unused_steps >= self.cfg.dead_code_steps
            dead[0] = False  # the identity code stays
            if dead.any():
                pick = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
                self.embedding[dead] = pick
                self.ema_weight[dead] = pick
                self.ema_size[dead] = 1.0
                self.unused_steps[dead] = 0.0

    def lookup(self, idx: Tensor) -> Tensor:
        return self.embedding[idx]


class LieCodebook(VectorQuantiser):
    """Vector quantiser over so(3) elements written in patch-local frames."""

    def __init__(self, patch_centres: Tensor, size: int, max_angle: float, cfg: TheoryConfig):
        super().__init__(size, 3, cfg, self._initial_codes(size, cfg.code_scale * max_angle))
        up = F.normalize(patch_centres, dim=-1)                      # [P, 3]
        z = torch.tensor([0.0, 0.0, 1.0]).expand_as(up)
        east = torch.cross(z, up, dim=-1)
        polar = east.norm(dim=-1, keepdim=True) < 1e-6
        east = torch.where(polar, torch.tensor([1.0, 0.0, 0.0]).expand_as(up), east)
        east = F.normalize(east - (east * up).sum(-1, keepdim=True) * up, dim=-1)
        north = torch.cross(up, east, dim=-1)
        self.register_buffer("frames", torch.stack([east, north, up], -2))  # [P, 3, 3] rows

    @staticmethod
    def _initial_codes(size: int, scale: float) -> Tensor:
        """Code 0 is the identity; then +-east, +-north, +-spin at two sizes."""
        axes = torch.eye(3)
        basis = torch.cat([axes, -axes], 0)  # [6, 3]
        codes = [torch.zeros(1, 3), scale * basis, 2 * scale * basis]
        codes = torch.cat(codes, 0)[:size]
        if len(codes) < size:
            extra = scale * torch.randn(size - len(codes), 3, generator=torch.Generator().manual_seed(0))
            codes = torch.cat([codes, extra], 0)
        return codes

    def to_local(self, omega: Tensor) -> Tensor:
        """World-frame algebra elements [..., P, 3] -> local (east, north, up)."""
        return torch.einsum("pij,...pj->...pi", self.frames, omega)

    def to_world(self, local: Tensor) -> Tensor:
        return torch.einsum("pji,...pj->...pi", self.frames, local)


@dataclasses.dataclass
class Explanation:
    states: List[Latent]        # s_1 .. s_{K+1}
    program: Tensor             # local algebra elements [K, B, S, P, 3] (joints for chain)
    indices: Optional[Tensor]   # code indices [K, B, S, P] (None when continuous)
    lengths: Tensor             # MDL-selected k* per (b, s, P), in 0..K
    patch_error: Tensor         # latent error per length [K+1, B, S, P]
    selected: Latent            # the assembled state (each node at its patch's k*)
    vq_loss: Tensor             # scalar
    entropy: Tensor             # scalar
    phys_indices: Optional[Tensor] = None  # physics code per step [K, B, S, P] (phase 2)
    sub_error: Optional[Tensor] = None     # patch error after the Lie sub-step [K, B, S, P]
    phys_gate: Optional[Tensor] = None     # physics gate per step [K, B, S, P] in (0, 1)


class TheoryProgram(nn.Module):
    """Programmer + so(3) codebook + fixed executor + MDL selection."""

    def __init__(self, graphs: WorldGraphs, num_shells: int, forcing_dim: int,
                 cs: int, cv_latent: int, d: int, cv: int, hidden_layers: int,
                 lie_cfg: LieConfig, cfg: TheoryConfig, static_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.static_dim = static_dim
        self.chain = lie_cfg.shell_coupling == "chain"
        # the programmer sees (s_k, s_y): two latent frames, like a 2-frame history
        self.proposer = LieGenerator(graphs, num_shells, 2, forcing_dim, cs, cv_latent,
                                     d, cv, hidden_layers, lie_cfg)
        self.codebook = (LieCodebook(graphs.patches.centres, cfg.codebook_size,
                                     self.proposer.max_angle, cfg)
                         if cfg.codebook_size > 0 else None)
        if self.codebook is None:  # frames are still needed to store continuous programs
            self.register_buffer("frames", LieCodebook(graphs.patches.centres, 1,
                                                       self.proposer.max_angle, cfg).frames)
        # phase 2: physics codes (identity + physics_codes) executed after the Lie step
        self.physics = self.phys_head = self.phys_codebook = None
        if cfg.physics_codes > 0:
            self.phys_head = nn.Linear(d, cfg.physics_dim)
            self.phys_codebook = VectorQuantiser(1 + cfg.physics_codes, cfg.physics_dim, cfg)
            # learned "use physics here" gate per (shell, patch); starts almost
            # closed, so the model begins as the Lie-only arm and opens the
            # physics step only where it lowers the error
            self.phys_gate = nn.Linear(d, 1)
            nn.init.zeros_(self.phys_gate.weight)
            nn.init.constant_(self.phys_gate.bias, -3.0)
            self.physics = PhysicsExecutor(forcing_dim, static_dim, cs, cv_latent, d,
                                           hidden_layers, cfg.physics_dim, cfg.physics_scale)

    def _static(self, s: Latent, ns: Optional[Tensor]) -> Tensor:
        if ns is None:
            b, n = s.h.shape[0], s.h.shape[2]
            ns = s.h.new_zeros(b, n, self.static_dim)
        return ns

    # ----------------------------------------------------------- primitives
    def _frames(self) -> Tensor:
        return self.codebook.frames if self.codebook is not None else self.frames

    def to_local(self, omega: Tensor) -> Tensor:
        return torch.einsum("pij,...pj->...pi", self._frames(), omega)

    def to_world(self, local: Tensor) -> Tensor:
        return torch.einsum("pji,...pj->...pi", self._frames(), local)

    def _omega_from_program(self, local: Tensor) -> Tensor:
        """Local program elements [B, S, P, 3] -> world omega per patch [B, S, P, 3]."""
        world = self.to_world(local)
        if not self.chain:
            return world
        chain = [world[:, 0]]
        for k in range(1, world.shape[1]):
            chain.append(compose(chain[-1], world[:, k]))
        return torch.stack(chain, 1)

    def apply(self, graphs: WorldGraphs, s: Latent, local: Tensor,
              phys: Optional[Tuple[Tensor, Tensor]] = None,
              nf: Optional[Tensor] = None, ns: Optional[Tensor] = None) -> Latent:
        """Executes one program step on every patch: s -> Phi_{exp(omega)}(s),
        then the physics code (`phys` = (code embeddings [B, S, P, dz], code
        indices [B, S, P])) as a column-local tendency."""
        omega = self._omega_from_program(local)
        b, sh = omega.shape[:2]
        omega_node = graphs.patches.blend(omega.flatten(0, 1)).unflatten(0, (b, sh))
        out = self.proposer.act(graphs, s, omega_node)
        if phys is not None and self.physics is not None:
            out = self.apply_physics(graphs, out, *phys, nf, ns)
        return out

    def apply_physics(self, graphs: WorldGraphs, s: Latent, e_patch: Tensor, idx: Tensor,
                      gate: Tensor, nf: Tensor, ns: Optional[Tensor]) -> Latent:
        """e_patch [B, S, P, dz] code embeddings, idx [B, S, P] codes, gate [B, S, P, 1]."""
        b, sh = e_patch.shape[:2]
        blend = lambda x: graphs.patches.blend(x.flatten(0, 1)).unflatten(0, (b, sh))
        e_node = blend(e_patch)                                              # [B, S, N, dz]
        w_node = blend(family_weights(idx, self.cfg.physics_codes) * gate)   # [B, S, N, 3]
        return self.physics(s, e_node, w_node, nf, self._static(s, ns))

    def propose(self, graphs: WorldGraphs, s: Latent, s_y: Latent, nf: Tensor
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]], Tensor, Tensor]:
        """One Lie primitive per (shell, patch) in local frame (quantised if a
        codebook exists) and, in phase 2, one physics code per (shell, patch).
        Returns (lie_local, lie_idx, (phys_embed, phys_idx) | None, vq, entropy)."""
        h, v = self.proposer.features(graphs, [s, s_y], nf)
        omega, _, joints = self.proposer.algebra(graphs, h, v)
        local = self.to_local(joints if self.chain else omega)
        zero = torch.zeros((), device=local.device)
        vq, ent, idx = zero, zero, None
        if self.codebook is not None and self.codebook.size == 1:
            # identity-only vocabulary (physics-only arm): no transport, and no
            # straight-through gradient into proposals that can never be used
            local = torch.zeros_like(local).detach()
            idx = torch.zeros(local.shape[:-1], dtype=torch.long, device=local.device)
        elif self.codebook is not None:
            local, idx, commitment, ent = self.codebook(local)
            vq = commitment.mean()
        phys = None
        if self.physics is not None:
            b, sh = h.shape[:2]
            pooled = graphs.patches.pool(h.float().flatten(0, 1)).unflatten(0, (b, sh))  # [B, S, P, d]
            e_q, p_idx, p_commit, p_ent = self.phys_codebook(self.phys_head(pooled))
            gate = torch.sigmoid(self.phys_gate(pooled))                                  # [B, S, P, 1]
            vq, ent = vq + p_commit.mean(), ent + p_ent
            phys = (e_q, p_idx, gate)
        return local, idx, phys, vq, ent

    # ---------------------------------------------------------- explanation
    @staticmethod
    def _patch_error(graphs: WorldGraphs, s: Latent, s_y: Latent) -> Tensor:
        """Mean squared latent difference pooled per patch: [B, S, P]."""
        b, sh = s.h.shape[:2]
        err = ((s.h.float() - s_y.h.float()).square().mean(-1)
               + (s.v.float() - s_y.v.float()).square().flatten(-2).mean(-1))  # [B, S, N]
        return graphs.patches.pool(err.flatten(0, 1)[..., None])[..., 0].unflatten(0, (b, sh))

    @staticmethod
    def _assemble(graphs: WorldGraphs, states: Sequence[Latent], lengths: Tensor) -> Latent:
        """Every node takes the state at its (hard) patch's selected length."""
        hard = graphs.patches.hard  # [N]
        k_node = lengths[..., hard]  # [B, S, N]
        h = torch.stack([z.h for z in states], 0)  # [K+1, B, S, N, Cs]
        v = torch.stack([z.v for z in states], 0)
        ih = k_node[None, ..., None].expand(1, *k_node.shape, h.shape[-1])
        iv = k_node[None, ..., None, None].expand(1, *k_node.shape, *v.shape[-2:])
        return Latent(h.gather(0, ih)[0], v.gather(0, iv)[0])

    def explain(self, graphs: WorldGraphs, s: Latent, s_y: Latent, nf: Tensor,
                steps: Optional[int] = None, ns: Optional[Tensor] = None) -> Explanation:
        """Infers a program field for the pair (s, s_y) and selects k* by MDL."""
        K = steps or self.cfg.max_steps
        states, program, indices, phys_indices, sub_errors, gates = [s], [], [], [], [], []
        vq_total = torch.zeros((), device=s.h.device)
        ent_total = torch.zeros((), device=s.h.device)
        errors = [self._patch_error(graphs, s, s_y)]
        cur = s
        for _ in range(K):
            local, idx, phys, vq, ent = self.propose(graphs, cur, s_y, nf)
            if phys is None:
                cur = self.apply(graphs, cur, local)
            else:  # Lie sub-step, then the physics code; keep the intermediate error
                cur = self.apply(graphs, cur, local)
                sub_errors.append(self._patch_error(graphs, cur, s_y))
                cur = self.apply_physics(graphs, cur, *phys, nf, ns)
                phys_indices.append(phys[1])
                gates.append(phys[2][..., 0])
            states.append(cur)
            program.append(local)
            indices.append(idx)
            errors.append(self._patch_error(graphs, cur, s_y))
            vq_total = vq_total + vq / K
            ent_total = ent_total + ent / K
        patch_error = torch.stack(errors, 0)  # [K+1, B, S, P]
        with torch.no_grad():
            penalty = self.cfg.lambda_mdl ** torch.arange(K + 1, device=s.h.device, dtype=torch.float32)
            lengths = (patch_error.detach() * penalty[:, None, None, None]).argmin(0)
        selected = self._assemble(graphs, states, lengths)
        return Explanation(states, torch.stack(program, 0),
                           torch.stack(indices, 0) if indices[0] is not None else None,
                           lengths, patch_error, selected, vq_total, ent_total,
                           torch.stack(phys_indices, 0) if phys_indices else None,
                           torch.stack(sub_errors, 0) if sub_errors else None,
                           torch.stack(gates, 0) if gates else None)

    @torch.no_grad()
    def execute(self, graphs: WorldGraphs, s: Latent, program: Tensor, lengths: Tensor,
                phys_indices: Optional[Tensor] = None, nf: Optional[Tensor] = None,
                ns: Optional[Tensor] = None, phys_gates: Optional[Tensor] = None) -> Latent:
        """Runs a stored program field on a (new) state: transfer.

        program [K, B, S, P, 3] local elements, lengths [B, S, P], optional
        physics codes [K, B, S, P] with their gates [K, B, S, P] (need the
        query state's forcing `nf`); a patch whose length is exceeded receives
        the identity for the rest."""
        cur = s
        for k in range(program.shape[0]):
            active = lengths > k
            phys = None
            if phys_indices is not None and self.physics is not None:
                idx = torch.where(active, phys_indices[k], torch.zeros_like(phys_indices[k]))
                gate = (phys_gates[k] if phys_gates is not None else torch.ones_like(idx, dtype=torch.float32))
                phys = (self.phys_codebook.lookup(idx), idx, gate[..., None].float())
            cur = self.apply(graphs, cur, program[k] * active.float()[..., None], phys, nf, ns)
        return cur

    def summary(self, exp: Explanation) -> Dict[str, float]:
        out = {"mean_len": float(exp.lengths.float().mean())}
        if exp.indices is not None:
            used = torch.unique(exp.indices)
            out["codes_used"] = float(len(used))
            counts = torch.bincount(exp.indices.flatten(), minlength=self.cfg.codebook_size).float()
            p = counts / counts.sum()
            out["perplexity"] = float(torch.exp(-(p * torch.log(p + 1e-9)).sum()))
        if exp.phys_indices is not None:
            active = exp.phys_indices > 0
            if exp.phys_gate is not None:
                active = active & (exp.phys_gate > 0.5)
                out["phys_gate"] = float(exp.phys_gate.mean())
            out["phys_codes_used"] = float(len(torch.unique(exp.phys_indices)))
            out["phys_frac"] = float(active.float().mean())
            # share of the per-step error reduction done by the physics sub-step
            lie_gain = exp.patch_error[:-1] - exp.sub_error
            phys_gain = exp.sub_error - exp.patch_error[1:]
            total = lie_gain.sum() + phys_gain.sum()
            out["phys_share"] = float(phys_gain.sum() / total) if abs(float(total)) > 1e-12 else 0.0
        return out
