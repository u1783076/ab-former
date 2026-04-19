import json
import os

import sentence_transformers
import torch
import torch.nn as nn

from safetensors.torch import load_model as _st_load, save_model as _st_save
from modules._STProjectionBase import _STProjectionBase

class QFormerLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            self.norm1(query), context, context,
            key_padding_mask=key_padding_mask,
        )
        query = query + attn_out
        query = query + self.ffn(self.norm2(query))
        return query


class QFormerProjectionModule(_STProjectionBase):

    config_keys = ["sbert_dim", "num_layers", "num_heads", "ffn_ratio"]

    def __init__(self, sbert_dim: int, num_layers: int = 2, num_heads: int = 8,
                 ffn_ratio: float = 4.0):
        super().__init__()
        self.sbert_dim = sbert_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.ffn_dim = int(sbert_dim * ffn_ratio)

        self.q_token = nn.Parameter(torch.randn(1, 1, sbert_dim) * 0.02)
        self.v_token = nn.Parameter(torch.randn(1, 1, sbert_dim) * 0.02)

        self.layers = nn.ModuleList([
            QFormerLayer(sbert_dim, num_heads, self.ffn_dim) for _ in range(num_layers)
        ])
        self.context_norm = nn.LayerNorm(sbert_dim)

    def forward(self, features: dict) -> dict:
        token_embs = self.context_norm(features["token_embeddings"]) 
        key_padding_mask = (features["attention_mask"] == 0)  

        batch_size = token_embs.shape[0]
        queries = torch.cat([
            self.q_token.expand(batch_size, -1, -1),
            self.v_token.expand(batch_size, -1, -1),
        ], dim=1)

        for layer in self.layers:
            queries = layer(queries, token_embs, key_padding_mask=key_padding_mask)

        features["q_sentence_embedding"] = queries[:, 0, :]  
        features["v_sentence_embedding"] = queries[:, 1, :] 
        features["sentence_embedding"] = torch.zeros_like(features["q_sentence_embedding"])
        return features