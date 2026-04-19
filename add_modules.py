import os 
from modules import (
    AsymProjectionModule,
    PrependSpecialTokenModule,
    SplitQVModule,
    QFormerProjectionModule,
    DualEncoderModule,
)
import sentence_transformers
from sentence_transformers import SentenceTransformer
import torch
import time
import torch.nn as nn
from typing import Optional
import torch.distributed as dist


def add_dual_encoder(
    model: sentence_transformers.SentenceTransformer,
) -> sentence_transformers.SentenceTransformer:
    import copy
    if any(isinstance(m, DualEncoderModule) for m in model._modules.values()):
        return model
    q_model = copy.deepcopy(model)
    v_model = copy.deepcopy(model)
    dual_module = DualEncoderModule(q_model, v_model)
    model._modules.clear()
    model._modules["0"] = dual_module
    return model


def add_asym_projection(
    model: sentence_transformers.SentenceTransformer,
    hidden_dim: int = None,
) -> sentence_transformers.SentenceTransformer:
    if any(isinstance(m, AsymProjectionModule) for m in model._modules.values()):
        return model
    sbert_dim = model.get_sentence_embedding_dimension()
    proj = AsymProjectionModule(
        sbert_dim,
        hidden_dim=hidden_dim if hidden_dim is not None else sbert_dim,
    )
    model._modules[str(len(model._modules))] = proj
    return model


def add_prepend_projection(
    model: sentence_transformers.SentenceTransformer,
    n_prefix_tokens: int = 1,
) -> sentence_transformers.SentenceTransformer:
    if any(isinstance(m, PrependSpecialTokenModule) for m in model._modules.values()):
        return model

    transformer_module = next(
        (m for m in model._modules.values()
         if isinstance(m, sentence_transformers.models.Transformer)),
        None,
    )
    if transformer_module is None:
        raise ValueError("Could not find sentence_transformers.models.Transformer in the pipeline.")
    sbert_dim = transformer_module.get_word_embedding_dimension()

    existing = list(model._modules.items())
    model._modules.clear()
    prepend = PrependSpecialTokenModule(n_prefix_tokens=int(n_prefix_tokens), sbert_dim=int(sbert_dim))
    prepend.bind_transformer(transformer_module)
    model._modules["0"] = prepend

    out_i = 1
    for _k, m in existing:
        model._modules[str(out_i)] = m
        out_i += 1

    model._modules[str(len(model._modules))] = SplitQVModule()
    module_with_tokenize = next(
        (m for m in model._modules.values()
        if callable(getattr(m, 'tokenize', None))),
        None,
    ) 
    model.tokenize = module_with_tokenize.tokenize
    print('Add prepend projection')
    return model

def _bind_prepend_if_present(model: SentenceTransformer) -> None:
    prepend = next((m for m in model._modules.values() if isinstance(m, PrependSpecialTokenModule)), None)
    if prepend is None:
        return
    transformer_module = next(
        (m for m in model._modules.values() if isinstance(m, sentence_transformers.models.Transformer)),
        None,
    )
    if transformer_module is None:
        return
    prepend.bind_transformer(transformer_module)

def build_sbert(
    model_id: str,
    device: str,
    asym_type: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    mlp_hidden_dim: Optional[int] = None,
    qformer_layers: int = 2,
    qformer_heads: int = 8,
    qformer_ffn_ratio: float = 4.0,
    prepend_n_tokens: int = 1,
) -> SentenceTransformer:
    if int(os.environ.get("RANK", 0)) == 0:
        sbert = SentenceTransformer(model_id, device=device)
    time.sleep(15)
    if "WORLD_SIZE" in os.environ:
        dist.barrier()
    sbert = SentenceTransformer(model_id, device=device, local_files_only=True)
    module_with_tokenize = next(
        (m for m in sbert._modules.values()
        if callable(getattr(m, 'tokenize', None))),
        None,
    )
    sbert._modules["0"].tokenize = module_with_tokenize.tokenize
    _bind_prepend_if_present(sbert)

    if max_seq_length is not None:
        sbert.max_seq_length = max_seq_length
    if asym_type is None:
        return sbert
    if asym_type == "mlp":
        add_asym_projection(sbert, hidden_dim=mlp_hidden_dim)
    elif asym_type == "qformer":
        add_qformer_projection(sbert, num_layers=qformer_layers,
                               num_heads=qformer_heads, ffn_ratio=qformer_ffn_ratio)
    elif asym_type == "prepend":
        add_prepend_projection(sbert, n_prefix_tokens=int(prepend_n_tokens))
    elif asym_type == "dual":
        add_dual_encoder(sbert)
    else:
        raise ValueError(f"Unknown asym_type: '{asym_type}'")
    sbert = sbert.to(device)
    return sbert


def asym_head_parameter_ids(model: SentenceTransformer) -> frozenset:
    ids: set = set()
    for m in model.modules():
        if isinstance(m, (AsymProjectionModule, QFormerProjectionModule, PrependSpecialTokenModule)):
            for p in m.parameters():
                ids.add(id(p))
    return frozenset(ids)


def add_qformer_projection(
    model: sentence_transformers.SentenceTransformer,
    num_layers: int = 2,
    num_heads: int = 8,
    ffn_ratio: float = 4.0,
) -> sentence_transformers.SentenceTransformer:
    if any(isinstance(m, QFormerProjectionModule) for m in model._modules.values()):
        return model

    sbert_dim = model.get_sentence_embedding_dimension()

    pooling_key = next(
        (k for k, m in model._modules.items()
         if isinstance(m, sentence_transformers.models.Pooling)),
        None,
    )
    if pooling_key is not None:
        keys_to_remove = [k for k in model._modules if int(k) >= int(pooling_key)]
        for k in keys_to_remove:
            del model._modules[k]
    proj = QFormerProjectionModule(
        sbert_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
    )
    model._modules[str(len(model._modules))] = proj
    return model
