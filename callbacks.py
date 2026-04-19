import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
import torch

from models import SparseKerasELSA, SparseKerasAsymELSA
from time import time
from utils import *

def _build_eval_model(sbert, is_asym, texts, items_idx, device):
    """Build evaluation model from current SBERT embeddings."""
    if is_asym:
        results = sbert.encode(texts, output_value=None, show_progress_bar=True)
        q_embs = torch.stack([r["q_sentence_embedding"] for r in results])
        v_embs = torch.stack([r["v_sentence_embedding"] for r in results])
        emb_dim = q_embs.shape[1]
        model = SparseKerasAsymELSA(len(items_idx), emb_dim, items_idx, device=device)
        model.to(device)
        model.set_weights([q_embs, v_embs])
    else:
        embs = sbert.encode(texts, show_progress_bar=True)
        emb_dim = embs.shape[1]
        model = SparseKerasELSA(len(items_idx), emb_dim, items_idx, device=device)
        model.to(device)
        model.set_weights([embs])
    return model

class evaluateWriter(keras.callbacks.Callback):
    def __init__(
        self,
        items_idx,
        sbert,
        texts,
        evaluator,
        logdir,
        DEVICE,
        is_asym_sbert=False,
        sbert_name="sbert_temp_model",
        evaluate_epoch="false",
        save_every_epoch="false",
        eval_model = None,
        coldstart_evaluator = None,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.coldstart_evaluator = coldstart_evaluator
        self.logdir = logdir
        self.sbert = sbert
        self.texts = texts
        self.items_idx = items_idx
        self.DEVICE = DEVICE
        self.is_asym = is_asym_sbert
        self.results_list = []
        self.sbert_name = sbert_name
        self.evaluate_epoch = evaluate_epoch
        self.save_every_epoch = save_every_epoch
        self.eval_model = eval_model

    def on_epoch_end(self, epoch, logs=None, eval_model=None, **kwargs):
        print()
        eval_model = eval_model if eval_model is not None else self.eval_model
        if self.save_every_epoch == "true":
            print("saving sbert model")
            if self.sbert:
                self.sbert.save(f"{self.sbert_name}-epoch-{epoch}")
        if self.evaluate_epoch == "true":
            model = eval_model if eval_model is not None else _build_eval_model(
                self.sbert, self.is_asym, self.texts, self.items_idx, self.DEVICE
            )
            if isinstance(self.evaluator, ColdStartEvaluation):
                df_preds = model.predict_df(
                    self.evaluator.test_src,
                    candidates_df=(
                        self.evaluator.cold_start_candidates_df
                        if hasattr(self.evaluator, "cold_start_candidates_df")
                        else None
                    ),
                    k=1000,
                )
                df_preds = df_preds[
                    ~df_preds.set_index(["item_id", "user_id"]).index.isin(
                        self.evaluator.test_src.set_index(["item_id", "user_id"]).index
                    )
                ]
            else:
                df_preds = model.predict_df(self.evaluator.test_src, **kwargs)
            results = self.evaluator(df_preds)
            
            if self.coldstart_evaluator:
                df_preds_coldstart = model.predict_df(self.coldstart_evaluator.test_src, **kwargs)
                coldstart_results = self.coldstart_evaluator(df_preds_coldstart)
                coldstart_results = {('cold_start_' + k) : v for k, v in coldstart_results.items()}
                results.update(coldstart_results)

            print(results)
            pd.Series(results).to_csv(f"{self.logdir}/result-epoch-{epoch}.csv")
            print("results file written")
            self.results_list.append(results)