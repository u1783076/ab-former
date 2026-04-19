import json
import os

import sentence_transformers
import torch
import torch.nn as nn

from safetensors.torch import load_model as _st_load, save_model as _st_save
from modules._STProjectionBase import _STProjectionBase

class SplitQVModule(_STProjectionBase):

    config_keys = []

    def __init__(self):
        super().__init__()

    def forward(self, features: dict) -> dict:
        emb = features["sentence_embedding"] 
        B = emb.shape[0] // 2
        features["q_sentence_embedding"] = emb[:B]
        features["v_sentence_embedding"] = emb[B:]
        features["sentence_embedding"] = torch.zeros_like(features["q_sentence_embedding"])
        return features