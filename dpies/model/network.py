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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch)
        scene, action_h, evidence_h = enc["scene"], enc["action_h"], enc["evidence_h"]
        b, k, d = action_h.shape
        n = evidence_h.shape[1]
        pair_h = self.pair_representations(action_h, scene)
        rival_logits = self.rival_head(pair_h).squeeze(-1)
        # Raw directed signed evidence score in chunks over ordered action pairs.
        pair_indices = [(i, j) for i in range(k) for j in range(k)]
        raw_chunks = []
        q = batch["geometry_query"]
        chunk_size = int(self.cfg.pair_chunk_size)
        for start in range(0, len(pair_indices), chunk_size):
            chunk = pair_indices[start:start + chunk_size]
            ai = torch.tensor([p[0] for p in chunk], dtype=torch.long, device=action_h.device)
            bi = torch.tensor([p[1] for p in chunk], dtype=torch.long, device=action_h.device)
            pc = len(chunk)
            ha = action_h.index_select(1, ai)  # [B,P,D]
            hb = action_h.index_select(1, bi)
            hp = pair_h[:, ai, bi, :]
            qa = q.index_select(2, ai)  # [B,N,P,Q]
            qb = q.index_select(2, bi)
            e = evidence_h[:, :, None, :].expand(-1, -1, pc, -1)
            ha_e = ha[:, None, :, :].expand(-1, n, -1, -1)
            hb_e = hb[:, None, :, :].expand(-1, n, -1, -1)
            hp_e = hp[:, None, :, :].expand(-1, n, -1, -1)
            hx_e = scene[:, None, None, :].expand(-1, n, pc, -1)
            fuse_in = torch.cat([e, ha_e, hb_e, hp_e, hx_e, qa, qb, qa - qb, torch.abs(qa - qb)], dim=-1)
            z = self.fuse(fuse_in)
            raw = self.signed_head(z).squeeze(-1)  # [B,N,P]
            raw_chunks.append(raw)
        raw_all = torch.cat(raw_chunks, dim=-1).reshape(b, n, k, k)
        signed = 0.5 * (raw_all - raw_all.transpose(-1, -2))
        return {
            "rival_logits": rival_logits,
            "rival_scores": torch.sigmoid(rival_logits),
            "signed_evidence": signed,
            "scene": scene,
            "action_h": action_h,
            "evidence_h": evidence_h,
        }
