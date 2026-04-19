import json
import os

import sentence_transformers
import torch
import torch.nn as nn

from safetensors.torch import load_model as _st_load, save_model as _st_save
from modules._STProjectionBase import _STProjectionBase

class AsymProjectionModule(_STProjectionBase):

    config_keys = ["sbert_dim", "hidden_dim"]

    def __init__(self, sbert_dim: int, hidden_dim: int):
        super().__init__()
        self.sbert_dim = sbert_dim
        self.hidden_dim = hidden_dim

        self.q_head = nn.Sequential(
            nn.LayerNorm(sbert_dim),
            nn.Linear(sbert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Linear(hidden_dim, sbert_dim),
        )
        self.v_head = nn.Sequential(
            nn.LayerNorm(sbert_dim),
            nn.Linear(sbert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, sbert_dim),
        )
        nn.init.zeros_(self.q_head[-1].weight)
        nn.init.zeros_(self.q_head[-1].bias)
        nn.init.zeros_(self.v_head[-1].weight)
        nn.init.zeros_(self.v_head[-1].bias)

    def forward(self, features: dict) -> dict:
        emb = features["sentence_embedding"]
        features["q_sentence_embedding"] = emb + self.q_head(emb)
        features["v_sentence_embedding"] = emb + self.v_head(emb)
        return features
