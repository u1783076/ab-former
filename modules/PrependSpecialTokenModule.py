import json
import os

import sentence_transformers
import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict

from safetensors.torch import load_model as _st_load, save_model as _st_save
from modules._STProjectionBase import _STProjectionBase

class PrependSpecialTokenModule(_STProjectionBase):

    config_keys = ["n_prefix_tokens", "sbert_dim"]

    def __init__(self, n_prefix_tokens: int, sbert_dim: int):
        super().__init__()
        if n_prefix_tokens < 0:
            raise ValueError("n_prefix_tokens must be >= 0")
        self.n_prefix_tokens = int(n_prefix_tokens)
        self.sbert_dim = int(sbert_dim)

        self.q_prefix = nn.Parameter(torch.randn(self.n_prefix_tokens, self.sbert_dim) * 0.02)
        self.v_prefix = nn.Parameter(torch.randn(self.n_prefix_tokens, self.sbert_dim) * 0.02)

        self._word_embeddings: Optional[nn.Module] = None

    def bind_transformer(self, transformer_module) -> "PrependSpecialTokenModule":
        hf_model = getattr(transformer_module, "auto_model", None)
        if hf_model is None:
            hf_model = getattr(transformer_module, "model", None)
        if hf_model is None or not hasattr(hf_model, "get_input_embeddings"):
            raise ValueError(
                "transformer_module must expose a HF model with `.get_input_embeddings()` "
                "(expected `.auto_model` or `.model`)."
            )
        self._word_embeddings = hf_model.get_input_embeddings()
        return self

    def forward(self, features: dict) -> dict:

        if self._word_embeddings is None:
            raise RuntimeError(
                "PrependSpecialTokenModule is not bound to the Transformer embeddings. "
                "Call `bind_transformer(transformer_module)` after inserting it into the pipeline."
            )

        input_ids = features["input_ids"]          
        attention_mask = features["attention_mask"] 
        batch_size, _seq_len = input_ids.shape
        device = input_ids.device

        token_embs = self._word_embeddings(input_ids)  

        q_prefix = self.q_prefix.to(device).unsqueeze(0).expand(batch_size, -1, -1) 
        v_prefix = self.v_prefix.to(device).unsqueeze(0).expand(batch_size, -1, -1) 
        prefix_mask = torch.ones(batch_size, self.n_prefix_tokens, device=device, dtype=attention_mask.dtype)

        q_inputs = torch.cat([q_prefix, token_embs], dim=1)  
        v_inputs = torch.cat([v_prefix, token_embs], dim=1)  

        features.pop("input_ids", None)
        features["inputs_embeds"] = torch.cat([q_inputs, v_inputs], dim=0)  
        features["attention_mask"] = torch.cat(
            [torch.cat([prefix_mask, attention_mask], dim=1),
             torch.cat([prefix_mask, attention_mask], dim=1)],
            dim=0,
        )  

        if "token_type_ids" in features:
            tti = features["token_type_ids"]  
            prefix_tti = torch.zeros(batch_size, self.n_prefix_tokens, device=device, dtype=tti.dtype)
            features["token_type_ids"] = torch.cat([torch.cat([prefix_tti, tti], dim=1)] * 2, dim=0)

        return features

    def load_state_dict(self, state_dict, strict: bool = True): 
        if any(k.startswith("_word_embeddings.") for k in state_dict.keys()):
            state_dict = OrderedDict((k, v) for k, v in state_dict.items() if not k.startswith("_word_embeddings."))
        return super().load_state_dict(state_dict, strict=strict)
