from __future__ import annotations

import torch
from torch import nn


def make_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0, final_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    if final_act:
        layers += [nn.LayerNorm(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SceneEncoder(nn.Module):
    def __init__(self, ego_dim: int, agent_dim: int, map_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ego = make_mlp(ego_dim, [hidden_dim], hidden_dim, dropout)
        self.agent = make_mlp(agent_dim, [hidden_dim], hidden_dim, dropout)
        self.map = make_mlp(map_dim, [hidden_dim], hidden_dim, dropout)
        self.fuse = make_mlp(hidden_dim * 3, [hidden_dim], hidden_dim, dropout)

    def forward(self, ego_history: torch.Tensor, agent_history: torch.Tensor, agent_mask: torch.Tensor,
                map_polylines: torch.Tensor, map_masks: torch.Tensor) -> torch.Tensor:
        ego_token = self.ego(ego_history).mean(dim=1)
        # Agent temporal mean then masked scene mean.
        b, a, h, _ = agent_history.shape
        agent_tok = self.agent(agent_history.reshape(b * a * h, -1)).reshape(b, a, h, -1).mean(dim=2)
        am = agent_mask.float().unsqueeze(-1)
        agent_scene = (agent_tok * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)
        # Map point mean per polyline then scene mean.
        b, p, l, _ = map_polylines.shape
        map_tok = self.map(map_polylines.reshape(b * p * l, -1)).reshape(b, p, l, -1)
        pm = map_masks.float().unsqueeze(-1)
        poly = (map_tok * pm).sum(dim=2) / pm.sum(dim=2).clamp_min(1.0)
        poly_mask = map_masks.any(dim=2).float().unsqueeze(-1)
        map_scene = (poly * poly_mask).sum(dim=1) / poly_mask.sum(dim=1).clamp_min(1.0)
        return self.fuse(torch.cat([ego_token, agent_scene, map_scene], dim=-1))


class ActionEncoder(nn.Module):
    def __init__(self, state_dim: int, meta_dim: int, num_modes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.state_mlp = make_mlp(state_dim, [hidden_dim // 2], hidden_dim // 2, dropout)
        self.gru = nn.GRU(hidden_dim // 2, hidden_dim // 2, batch_first=True)
        self.mode_emb = nn.Embedding(num_modes, hidden_dim // 8)
        self.meta_mlp = make_mlp(meta_dim, [hidden_dim // 2], hidden_dim // 2, dropout)
        self.out = make_mlp(hidden_dim // 2 + hidden_dim // 8 + hidden_dim // 2 + hidden_dim, [hidden_dim], hidden_dim, dropout)

    def forward(self, actions: torch.Tensor, action_meta: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        b, k, t, d = actions.shape
        x = self.state_mlp(actions.reshape(b * k * t, d)).reshape(b * k, t, -1)
        _, h = self.gru(x)
        h = h[-1].reshape(b, k, -1)
        mode = action_meta[..., 0].long().clamp_min(0).clamp_max(self.mode_emb.num_embeddings - 1)
        mode_h = self.mode_emb(mode)
        meta_h = self.meta_mlp(action_meta)
        scene_h = scene[:, None, :].expand(-1, k, -1)
        return self.out(torch.cat([h, mode_h, meta_h, scene_h], dim=-1))


class EvidenceEncoder(nn.Module):
    def __init__(self, evidence_dim: int, num_types: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, hidden_dim // 8)
        self.feat = make_mlp(evidence_dim, [hidden_dim], hidden_dim // 2, dropout)
        self.out = make_mlp(hidden_dim // 8 + hidden_dim // 2 + hidden_dim, [hidden_dim], hidden_dim, dropout)

    def forward(self, evidence_features: torch.Tensor, evidence_type: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        b, n, _ = evidence_features.shape
        typ = evidence_type.long().clamp_min(0).clamp_max(self.type_emb.num_embeddings - 1)
        type_h = self.type_emb(typ)
        feat_h = self.feat(evidence_features)
        scene_h = scene[:, None, :].expand(-1, n, -1)
        return self.out(torch.cat([type_h, feat_h, scene_h], dim=-1))
