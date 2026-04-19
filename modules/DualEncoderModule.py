import json
import os

import sentence_transformers
import torch
import torch.nn as nn

from safetensors.torch import load_model as _st_load, save_model as _st_save
from modules._STProjectionBase import _STProjectionBase

class DualEncoderModule(nn.Module):

    def __init__(
        self,
        q_model: sentence_transformers.SentenceTransformer,
        v_model: sentence_transformers.SentenceTransformer,
    ):
        super().__init__()
        self.q_model = q_model
        self.v_model = v_model

    def tokenize(self, texts):
        return self.q_model.tokenize(texts)

    def forward(self, features: dict) -> dict:
        q_features = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in features.items()}
        v_features = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in features.items()}
        q_out = self.q_model(q_features)
        v_out = self.v_model(v_features)
        features["q_sentence_embedding"] = q_out["sentence_embedding"]
        features["v_sentence_embedding"] = v_out["sentence_embedding"]
        features["sentence_embedding"] = torch.zeros_like(features["q_sentence_embedding"])
        return features

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump({}, f)
        self.q_model.save(os.path.join(output_path, "q_encoder"))
        self.v_model.save(os.path.join(output_path, "v_encoder"))

    @classmethod
    def load(cls, input_path: str) -> "DualEncoderModule":
        q_model = sentence_transformers.SentenceTransformer(
            os.path.join(input_path, "q_encoder")
        )
        v_model = sentence_transformers.SentenceTransformer(
            os.path.join(input_path, "v_encoder")
        )
        return cls(q_model, v_model)