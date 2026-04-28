from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from dpies.model.encoders import ActionEncoder, EvidenceEncoder, SceneEncoder, make_mlp


@dataclass
class DPIESConfig:
    ego_dim: int = 9
    agent_dim: int = 8
    map_dim: int = 4
    action_state_dim: int = 6
    action_meta_dim: int = 8
    evidence_dim: int = 32
    query_dim: int = 24
    num_action_modes: int = 9
    num_evidence_types: int = 6
    hidden_dim: int = 256
    dropout: float = 0.1
    pair_chunk_size: int = 64


class DPIESNetwork(nn.Module):
    def __init__(self, cfg: DPIESConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.scene_encoder = SceneEncoder(cfg.ego_dim, cfg.agent_dim, cfg.map_dim, d, cfg.dropout)
        self.action_encoder = ActionEncoder(cfg.action_state_dim, cfg.action_meta_dim, cfg.num_action_modes, d, cfg.dropout)
        self.evidence_encoder = EvidenceEncoder(cfg.evidence_dim, cfg.num_evidence_types, d, cfg.dropout)
        self.pair_encoder = make_mlp(d * 5, [d], d, cfg.dropout)
        self.rival_head = nn.Linear(d, 1)
        fuse_in = d + d + d + d + d + cfg.query_dim * 4
        self.fuse = make_mlp(fuse_in, [d, d], d, cfg.dropout)
        self.signed_head = nn.Linear(d, 1)

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scene = self.scene_encoder(batch["ego_history"], batch["agent_history"], batch["agent_mask"],
                                   batch["map_polylines"], batch["map_masks"],
                                   batch.get("agent_history_mask", None))
        action_h = self.action_encoder(batch["actions"], batch["action_meta"], scene)
        evidence_h = self.evidence_encoder(batch["evidence_features"], batch["evidence_type"], scene)
        return {"scene": scene, "action_h": action_h, "evidence_h": evidence_h}

    def pair_representations(self, action_h: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        b, k, d = action_h.shape
        ha = action_h[:, :, None, :].expand(-1, -1, k, -1)
        hb = action_h[:, None, :, :].expand(-1, k, -1, -1)
        hx = scene[:, None, None, :].expand(-1, k, k, -1)
        pair_in = torch.cat([ha, hb, ha - hb, ha * hb, hx], dim=-1)
        return self.pair_encoder(pair_in)

    def forward_rival(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 1: encode scene/actions/evidence and compute dense rival logits.

        This stage is much cheaper than signed-evidence fusion because it only
        scores action pairs, without evidence-pair fusion over [N,K,K].
        """
        enc = self.encode(batch)
        scene = enc["scene"]
        action_h = enc["action_h"]
        evidence_h = enc["evidence_h"]

        pair_h = self.pair_representations(action_h, scene)
        rival_logits = self.rival_head(pair_h).squeeze(-1)

        return {
            "rival_logits": rival_logits,
            "rival_scores": torch.sigmoid(rival_logits),
            "scene": scene,
            "action_h": action_h,
            "evidence_h": evidence_h,
            "pair_h": pair_h,
        }

    def _signed_evidence_for_flat_pairs(
            self,
            batch: Dict[str, torch.Tensor],
            scene: torch.Tensor,
            action_h: torch.Tensor,
            evidence_h: torch.Tensor,
            pair_h: torch.Tensor,
            flat_pair_idx: torch.Tensor,
            pair_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw directed evidence for padded flattened pair indices.

        flat_pair_idx: [B,P], each value is a*K+b.
        pair_valid: [B,P], marks real non-padding entries.
        Returns raw directed evidence [B,N,P].
        """
        b, k, d = action_h.shape
        n = evidence_h.shape[1]
        pmax = flat_pair_idx.shape[1]

        if pmax == 0:
            return action_h.new_zeros((b, n, 0))

        ai = torch.div(flat_pair_idx, k, rounding_mode="floor")
        bi = flat_pair_idx.remainder(k)

        # Keep query dtype aligned with encoded tensors under AMP.
        q = batch["geometry_query"].to(dtype=action_h.dtype)
        qdim = q.shape[-1]

        chunk_size = int(self.cfg.pair_chunk_size)
        raw_chunks = []

        pair_flat = pair_h.reshape(b, k * k, d)

        for start in range(0, pmax, chunk_size):
            end = min(start + chunk_size, pmax)
            ai_c = ai[:, start:end]
            bi_c = bi[:, start:end]
            valid_c = pair_valid[:, start:end]
            pc = end - start

            ha = torch.gather(action_h, 1, ai_c[:, :, None].expand(-1, -1, d))
            hb = torch.gather(action_h, 1, bi_c[:, :, None].expand(-1, -1, d))
            hp = torch.gather(pair_flat, 1, (ai_c * k + bi_c)[:, :, None].expand(-1, -1, d))

            qa = torch.gather(q, 2, ai_c[:, None, :, None].expand(-1, n, -1, qdim))
            qb = torch.gather(q, 2, bi_c[:, None, :, None].expand(-1, n, -1, qdim))

            e = evidence_h[:, :, None, :].expand(-1, -1, pc, -1)
            ha_e = ha[:, None, :, :].expand(-1, n, -1, -1)
            hb_e = hb[:, None, :, :].expand(-1, n, -1, -1)
            hp_e = hp[:, None, :, :].expand(-1, n, -1, -1)
            hx_e = scene[:, None, None, :].expand(-1, n, pc, -1)

            fuse_in = torch.cat(
                [e, ha_e, hb_e, hp_e, hx_e, qa, qb, qa - qb, torch.abs(qa - qb)],
                dim=-1,
            )
            z = self.fuse(fuse_in)
            raw = self.signed_head(z).squeeze(-1)  # [B,N,P]
            raw = raw * valid_c[:, None, :].to(raw.dtype)
            raw_chunks.append(raw)

        return torch.cat(raw_chunks, dim=-1)

    def signed_evidence_for_pair_mask(
            self,
            batch: Dict[str, torch.Tensor],
            enc: Dict[str, torch.Tensor],
            pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 2: compute signed evidence only for screened pairs.

        Returns dense [B,N,K,K], but only screened pairs and their reverse pairs
        are populated. Unscreened entries are zero and must remain masked by
        pair_mask in losses and Q computation.
        """
        scene = enc["scene"]
        action_h = enc["action_h"]
        evidence_h = enc["evidence_h"]
        pair_h = enc["pair_h"]

        b, k, _ = action_h.shape
        n = evidence_h.shape[1]
        device = action_h.device

        # Need both directions to preserve signed antisymmetry:
        # signed(a,b) = 0.5 * (raw(a,b) - raw(b,a)).
        needed = pair_mask.bool() | pair_mask.bool().transpose(1, 2)
        eye = torch.eye(k, dtype=torch.bool, device=device)[None]
        needed = needed & (~eye)

        flat_needed = needed.reshape(b, k * k)
        counts = flat_needed.sum(dim=1)
        pmax = int(counts.max().item()) if b > 0 else 0

        if pmax == 0:
            return action_h.new_zeros((b, n, k, k))

        flat_pair_idx = torch.zeros((b, pmax), dtype=torch.long, device=device)
        pair_valid = torch.zeros((b, pmax), dtype=torch.bool, device=device)

        for i in range(b):
            cur = torch.where(flat_needed[i])[0]
            p = int(cur.numel())
            if p > 0:
                flat_pair_idx[i, :p] = cur
                pair_valid[i, :p] = True

        raw_padded = self._signed_evidence_for_flat_pairs(
            batch=batch,
            scene=scene,
            action_h=action_h,
            evidence_h=evidence_h,
            pair_h=pair_h,
            flat_pair_idx=flat_pair_idx,
            pair_valid=pair_valid,
        )

        raw_flat = raw_padded.new_zeros((b, n, k * k))
        scatter_idx = flat_pair_idx[:, None, :].expand(-1, n, -1)
        raw_flat = raw_flat.scatter(2, scatter_idx, raw_padded)

        raw_full = raw_flat.reshape(b, n, k, k)
        signed = 0.5 * (raw_full - raw_full.transpose(-1, -2))
        return signed

    def signed_evidence_full(
            self,
            batch: Dict[str, torch.Tensor],
            enc: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Original full K x K signed-evidence computation."""
        scene = enc["scene"]
        action_h = enc["action_h"]
        evidence_h = enc["evidence_h"]
        pair_h = enc["pair_h"]

        b, k, _ = action_h.shape
        device = action_h.device
        flat_pair_idx = torch.arange(k * k, dtype=torch.long, device=device)[None].expand(b, -1)
        pair_valid = torch.ones((b, k * k), dtype=torch.bool, device=device)

        raw_all = self._signed_evidence_for_flat_pairs(
            batch=batch,
            scene=scene,
            action_h=action_h,
            evidence_h=evidence_h,
            pair_h=pair_h,
            flat_pair_idx=flat_pair_idx,
            pair_valid=pair_valid,
        ).reshape(b, evidence_h.shape[1], k, k)

        signed = 0.5 * (raw_all - raw_all.transpose(-1, -2))
        return signed

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Backward-compatible full forward."""
        out = self.forward_rival(batch)
        out["signed_evidence"] = self.signed_evidence_full(batch, out)
        return out
