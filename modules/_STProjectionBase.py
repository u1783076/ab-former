import json
import os

import sentence_transformers
import torch
import torch.nn as nn

from safetensors.torch import load_model as _st_load, save_model as _st_save

class _STProjectionBase(nn.Module):

    config_keys: list = []

    def get_config_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f, indent=2)
        if safe_serialization:
            _st_save(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path: str) -> "_STProjectionBase":
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)
        module = cls(**config)
        safetensors_path = os.path.join(input_path, "model.safetensors")
        bin_path = os.path.join(input_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            _st_load(module, safetensors_path)
        else:
            module.load_state_dict(torch.load(bin_path, map_location="cpu", weights_only=True))
        return module