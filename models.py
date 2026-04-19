import json
import logging
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import math
import numpy as np
import scipy.sparse as sp
import torch

from dataloaders import *
from layers import *
from utils import get_first_item

logger = logging.getLogger(__name__)

def filter_params(trainable_weights, asym_param_ids: frozenset):
    asym_w, back_w = [], []
    for w in trainable_weights:
        (asym_w if id(w.value) in asym_param_ids else back_w).append(w)
    return asym_w, back_w


def _scale_asym_head_gradients(trainable_weights, asym_param_ids: frozenset, asym_params_lr_scaling: float) -> None:
    scale = float(asym_params_lr_scaling)
    if scale == 1.0 or not asym_param_ids:
        return
    asym_w, _ = filter_params(trainable_weights, asym_param_ids)
    for w in asym_w:
        g = w.value.grad
        if g is not None:
            g.mul_(scale)


def _frob_sq_q_vt(Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    n = Q.shape[0]
    if n <= 1:
        return Q.new_zeros((), dtype=Q.dtype, device=Q.device)
    qtq = Q.mT @ Q
    vtv = V.mT @ V
    full_scaled = torch.trace((qtq / n) @ (vtv / n))
    diag_dots = (Q * V).sum(dim=-1)
    diag_sq_sum = (diag_dots * diag_dots).sum()  
    numer = n * full_scaled - diag_sq_sum / n
    numer = numer.clamp(min=0.0)
    return numer / (n - 1)


class NMSEbeeformer(keras.models.Model):
    def __init__(self, tokenized_sentences, items_idx, sbert, device, top_k=0, sbert_batch_size=128):
        super().__init__()
        self.device = device
        self.sbert = LayerSBERT(sbert, device, tokenized_sentences)
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.top_k = top_k
        self.sbert_batch_size = sbert_batch_size

    def call(self, x):
        return self.sbert(x)

    def train_step(self, data):
        a, b = data
        x, y = a
        y = torch.hstack((x, y))
        x_out = y
        tokenized_items, slicer, negative_slicer = b
        slicer = slicer.to(self.device)
        if negative_slicer is not None:
            negative_slicer = negative_slicer.to(self.device)

        # init everything for training
        self.zero_grad()
        sbert_batch_size = self.sbert_batch_size
        len_sentences = get_first_item(tokenized_items).shape[0]
        max_i = math.ceil(len_sentences / sbert_batch_size)

        with torch.no_grad():
            # we are doing it in batches because of memory
            batched_results = []
            for i in range(max_i):
                ind = i * sbert_batch_size
                ind_min = ind
                ind_max = ind + sbert_batch_size
                batch_result = self.sbert({k: v[ind_min:ind_max] for k, v in tokenized_items.items()})
                batched_results.append(batch_result)
            A = torch.vstack(batched_results)

        # track gradients for A, this will be our gradient checkpoint
        A.requires_grad = True

        # compute ELSA forward pass only for rows with values
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)
        A_negative_slicer = A[negative_slicer]
        A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)

        # ELSA step
        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        y_pred = keras.activations.relu(xAAT - x_out)

        # theoretically, this might improve performance for bigger dataset
        if self.top_k > 0:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        # compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # compute gradients for the gradient checkpoint (our ELSA A matrix)
        loss.backward()

        # sbert forward pass #2
        # now we will do the sbert forward pass again, but this time we will track gradients this time, for memory reasons in again batches
        batched_results = []
        for i in range(max_i):
            ind = i * sbert_batch_size
            ind_min = ind
            ind_max = ind + sbert_batch_size
            # actual forward pass
            temp_out = self.sbert({k: v[ind_min:ind_max] for k, v in tokenized_items.items()})
            # we need to get gradients for part of A
            temp_out.retain_grad()
            # get the slice of corresponding gradients
            partial_A_grad = A.grad[ind_min:ind_max]
            # compute gradients for sbert
            temp_out.backward(gradient=partial_A_grad)

        # get gradients for sbert
        trainable_weights = [v for v in self.sbert.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class AsymNMSEbeeformer(keras.models.Model):
    def __init__(
        self,
        tokenized_sentences,
        items_idx,
        sbert,
        device,
        top_k=0,
        sbert_batch_size=128,
        asym_params_lr_scaling=1.0,
        qvt_l2_reg=False,
        qvt_l2_weight=0.0,
    ):
        super().__init__()
        self.device = device
        self.asym_sbert = LayerAsymSBERT(sbert, device, tokenized_sentences)
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.top_k = top_k
        self.sbert_batch_size = sbert_batch_size
        self.asym_params_lr_scaling = asym_params_lr_scaling
        self.qvt_l2_reg = bool(qvt_l2_reg)
        self.qvt_l2_weight = float(qvt_l2_weight)
        self._asym_param_ids = self.asym_sbert._asym_head_param_ids

    def call(self, x):
        return self.asym_sbert(x)

    def train_step(self, data):
        a, b = data
        x, y = a
        y = torch.hstack((x, y))
        x_out = y
        tokenized_items, slicer, negative_slicer = b
        slicer = slicer.to(self.device)
        if negative_slicer is not None:
            negative_slicer = negative_slicer.to(self.device)

        self.zero_grad()
        sbert_batch_size = self.sbert_batch_size
        len_sentences = get_first_item(tokenized_items).shape[0]
        max_i = math.ceil(len_sentences / sbert_batch_size)

        with torch.no_grad():
            Q_batches, V_batches = [], []
            for i in range(max_i):
                ind_min = i * sbert_batch_size
                ind_max = ind_min + sbert_batch_size
                q_batch, v_batch = self.asym_sbert(
                    {k: v[ind_min:ind_max] for k, v in tokenized_items.items()}
                )
                Q_batches.append(q_batch)
                V_batches.append(v_batch)
            Q_all = torch.cat(Q_batches, dim=0)
            V_all = torch.cat(V_batches, dim=0)

        Q_all.requires_grad = True
        V_all.requires_grad = True

        Q_slicer = torch.nn.functional.normalize(Q_all[slicer], dim=-1)
        Q_negative_slicer = torch.nn.functional.normalize(Q_all[negative_slicer], dim=-1)
        V_negative_slicer = torch.nn.functional.normalize(V_all[negative_slicer], dim=-1)

        xQ = torch.matmul(x, Q_slicer)
        xQVT = torch.matmul(xQ, V_negative_slicer.T)

        diag_qvt = (Q_negative_slicer * V_negative_slicer).sum(dim=-1)
        y_pred = keras.activations.relu(xQVT - x_out * diag_qvt)

        if self.top_k > 0:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        loss = self.compute_loss(y=y, y_pred=y_pred)
        if self.qvt_l2_reg and self.qvt_l2_weight != 0.0:
            loss = loss + self.qvt_l2_weight * _frob_sq_q_vt(
                Q_negative_slicer, V_negative_slicer
            )
        loss.backward()

        # forward pass #2 — с градиентами, пропагируем Q_all.grad и V_all.grad
        for i in range(max_i):
            ind_min = i * sbert_batch_size
            ind_max = ind_min + sbert_batch_size
            q_batch, v_batch = self.asym_sbert(
                {k: v[ind_min:ind_max] for k, v in tokenized_items.items()}
            )
            torch.autograd.backward(
                [q_batch, v_batch],
                [Q_all.grad[ind_min:ind_max], V_all.grad[ind_min:ind_max]],
            )

        trainable_weights = [v for v in self.asym_sbert.trainable_weights]
        _scale_asym_head_gradients(
            trainable_weights,
            self._asym_param_ids,
            self.asym_params_lr_scaling,
        )
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


# ELSA model optimized for sparse data, used only for predictions
class SparseKerasELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device, top_k=0):
        super().__init__()
        self.device = device
        self.ELSA = LayerELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()
        self(np.zeros([1, n_items]))
        self.finetuning = False
        self.top_k = top_k

    def call(self, x):
        return self.ELSA(x)

    def save(self, path: str, **kwargs) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "items_idx.npy"), np.array(self.items_idx, dtype=object))
        # Save raw parameters (not normalized); torch -> numpy on CPU
        A = self.ELSA.A.detach().to("cpu").float().numpy()
        np.save(os.path.join(path, "A.npy"), A)

    @classmethod
    def load(cls, path: str, device, top_k: int = 0) -> "SparseKerasELSA":
        items_idx = np.load(os.path.join(path, "items_idx.npy"), allow_pickle=True)
        A = torch.from_numpy(np.load(os.path.join(path, "A.npy")))
        n_items, n_dims = A.shape
        model = cls(n_items=n_items, n_dims=n_dims, items_idx=items_idx, device=device, top_k=top_k)
        model.to(device)
        model.set_weights([A])
        return model

    def train_step(self, data):
        # Unpack the data
        if len(data) == 2:
            full_x = None
            a, b = data
            x, y = a
            y = torch.hstack((x, y))
            slicer, negative_slicer = b

        elif len(data) == 3:
            full_x, slicer, negative_slicer = data
        else:
            full_x, slicer = data
            negative_slicer = None

        if full_x is not None:
            if negative_slicer is not None:
                y = full_x[:, negative_slicer]
            else:
                y = full_x

            x = full_x[:, slicer]

            x = x.to(self.device)
            y = y.to(self.device)

        x = torch.nn.functional.normalize(x, p=1.0, dim=-1)
        y = torch.nn.functional.normalize(y, p=1.0, dim=-1)

        x_out = y

        self.zero_grad()

        A = self.ELSA.A
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)

        if negative_slicer is not None:
            A_negative_slicer = A[negative_slicer]
            A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        else:
            A_negative_slicer = torch.nn.functional.normalize(A, dim=-1)

        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        y_pred = xAAT - x_out

        if self.finetuning:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        # Compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_df(self, df, k=100, user_ids=None, candidates_df=None, block_reminder=True, tau_inf=None, c_time = 0):
        # create predictions from data in dataframe, returns predictions in dataframe
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(candidates_df, item_indices=self.items_idx).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)  # .to(self.device)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)

        batch_size = 1024
        n_users = len(user_ids)
        dfs = []

        for start in tqdm(range(0, n_users, batch_size), total=math.ceil(n_users / batch_size)):
            end = min(start + batch_size, n_users)
            batch_uids = user_ids[start:end]
            
            x, _ = data[start // batch_size]
            x_mask = x

            batch = torch.from_numpy(self.predict_on_batch(x))
            if block_reminder:
                mask = 1 - x_mask.astype(bool)
                batch = batch * mask

            if candidates_df is not None:
                batch *= candidates_vec
                values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame(
                {
                    "user_id": np.stack([batch_uids] * k).flatten("F"),
                    "item_id": np.array(self.items_idx)[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )
            df["user_id"] = df["user_id"].astype(str).astype("category")
            df["item_id"] = df["item_id"].astype(str).astype("category")
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype("category")
        df["item_id"] = df["item_id"].astype(str).astype("category")
        return df


class SparseKerasAsymELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device, top_k=0):
        super().__init__()
        self.device = device
        self.ELSA = LayerAsymELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()
        self(np.zeros([1, n_items]))
        self.finetuning = False
        self.top_k = top_k

    def call(self, x):
        return self.ELSA(x)

    def save(self, path: str, **kwargs) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "items_idx.npy"), np.array(self.items_idx, dtype=object))
        Q = self.ELSA.Q.detach().to("cpu").float().numpy()
        V = self.ELSA.V.detach().to("cpu").float().numpy()
        np.save(os.path.join(path, "Q.npy"), Q)
        np.save(os.path.join(path, "V.npy"), V)

    @classmethod
    def load(cls, path: str, device, top_k: int = 0) -> "SparseKerasAsymELSA":
        items_idx = np.load(os.path.join(path, "items_idx.npy"), allow_pickle=True)
        Q = torch.from_numpy(np.load(os.path.join(path, "Q.npy")))
        V = torch.from_numpy(np.load(os.path.join(path, "V.npy")))
        n_items, n_dims = Q.shape
        model = cls(n_items=n_items, n_dims=n_dims, items_idx=items_idx, device=device, top_k=top_k)
        model.to(device)
        model.set_weights([Q, V])
        return model

    def train_step(self, data):
        if len(data) == 2:
            full_x = None
            a, b = data
            x, y = a
            y = torch.hstack((x, y))
            slicer, negative_slicer = b
        elif len(data) == 3:
            full_x, slicer, negative_slicer = data
        else:
            full_x, slicer = data
            negative_slicer = None

        if full_x is not None:
            if negative_slicer is not None:
                y = full_x[:, negative_slicer]
            else:
                y = full_x
            x = full_x[:, slicer]
            x = x.to(self.device)
            y = y.to(self.device)

        x = torch.nn.functional.normalize(x, p=1.0, dim=-1)
        y = torch.nn.functional.normalize(y, p=1.0, dim=-1)
        x_out = y

        self.zero_grad()

        Q = self.ELSA.Q
        V = self.ELSA.V
        Q_slicer = torch.nn.functional.normalize(Q[slicer], dim=-1)

        if negative_slicer is not None:
            Q_negative_slicer = torch.nn.functional.normalize(Q[negative_slicer], dim=-1)
            V_negative_slicer = torch.nn.functional.normalize(V[negative_slicer], dim=-1)
        else:
            Q_negative_slicer = torch.nn.functional.normalize(Q, dim=-1)
            V_negative_slicer = torch.nn.functional.normalize(V, dim=-1)

        xQ = torch.matmul(x, Q_slicer)
        xQVT = torch.matmul(xQ, V_negative_slicer.T)
        diag_qvt = (Q_negative_slicer * V_negative_slicer).sum(dim=-1)
        y_pred = xQVT - x_out * diag_qvt

        if self.finetuning:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        loss = self.compute_loss(y=y, y_pred=y_pred)
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def predict_df(self, df, k=100, user_ids=None, candidates_df=None, block_reminder=True,
                   tau_inf=None, c_time=0):
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(candidates_df, item_indices=self.items_idx).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)

        if tau_inf is not None:
            X_weighted = get_position_weighted_matrix(df, self.items_idx, tau_inf, c_time)
            X_binary = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024).X
        else:
            data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)

        batch_size = 1024
        n_users = len(user_ids)
        dfs = []

        for start in tqdm(range(0, n_users, batch_size), total=math.ceil(n_users / batch_size)):
            end = min(start + batch_size, n_users)
            batch_uids = user_ids[start:end]

            if tau_inf is not None:
                x = X_weighted[start:end].toarray().astype("float32")
                x_mask = X_binary[start:end].toarray()
            else:
                x, _ = data[start // batch_size]
                x_mask = x

            batch = torch.from_numpy(self.predict_on_batch(x))
            if block_reminder:
                mask = 1 - x_mask.astype(bool)
                batch = batch * mask

            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            result_df = pd.DataFrame(
                {
                    "user_id": np.stack([batch_uids] * k).flatten("F"),
                    "item_id": np.array(self.items_idx)[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )
            result_df["user_id"] = result_df["user_id"].astype(str).astype("category")
            result_df["item_id"] = result_df["item_id"].astype(str).astype("category")
            dfs.append(result_df)

        out = pd.concat(dfs)
        out["user_id"] = out["user_id"].astype(str).astype("category")
        out["item_id"] = out["item_id"].astype(str).astype("category")
        return out


class SparseKerasEASE(keras.models.Model):

    def __init__(self, items_idx, device, B: torch.Tensor | None = None):
        super().__init__()
        self.items_idx = items_idx
        self.device = device
        self._B = B

    def call(self, x):
        return x @ self._B.to(x.device)

    def save(self, path: str, **kwargs) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "B.npy"), self._B.numpy())
        np.save(os.path.join(path, "items_idx.npy"), np.array(self.items_idx))

    @classmethod
    def load(cls, path: str, device) -> "SparseKerasEASE":
        items_idx = np.load(os.path.join(path, "items_idx.npy"), allow_pickle=True)
        B = torch.from_numpy(np.load(os.path.join(path, "B.npy")))
        return cls(items_idx, device, B=B)

    def predict_df(self, df, k=100, user_ids=None, candidates_df=None, block_reminder=True):
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(
                candidates_df, item_indices=self.items_idx
            ).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)
        dfs = []

        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]
            batch = torch.from_numpy(self.predict_on_batch(x))

            if block_reminder:
                batch = batch * (1 - x.astype(bool))
            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            chunk = pd.DataFrame({
                "user_id": np.stack([batch_uids] * k).flatten("F"),
                "item_id": np.array(self.items_idx)[indices_].flatten(),
                "value": values_.flatten(),
            })
            chunk["user_id"] = chunk["user_id"].astype(str).astype("category")
            chunk["item_id"] = chunk["item_id"].astype(str).astype("category")
            dfs.append(chunk)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype("category")
        df["item_id"] = df["item_id"].astype(str).astype("category")
        return df


class AsymBeeformerL3AE(keras.models.Model):

    def __init__(
        self,
        sbert,
        device,
        tokenized_sentences,
        items_idx,
        lambda_s: float = 1.0,
        lambda_b: float = 500.0,
        lambda_r: float = 10.0,
        sbert_batch_size: int = 128,
    ):
        super().__init__()
        self.device = device
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.sbert_batch_size = sbert_batch_size
        self.lambda_s = lambda_s
        self.lambda_b = lambda_b
        self.lambda_r = lambda_r

        self.asym_sbert = LayerAsymSBERT(sbert, device, tokenized_sentences)
        for param in self.asym_sbert.sbert.parameters():
            param.requires_grad_(False)

    def call(self, x):
        return self.asym_sbert(x)

    def _encode_all_items(self) -> tuple[np.ndarray, np.ndarray]:
        tokenized_items = self.tokenized_sentences
        n_total = get_first_item(tokenized_items).shape[0]
        n_batches = math.ceil(n_total / self.sbert_batch_size)

        Q_parts, V_parts = [], []
        with torch.no_grad():
            for i in range(n_batches):
                lo = i * self.sbert_batch_size
                hi = lo + self.sbert_batch_size
                batch = {k: v[lo:hi].to(self.device) for k, v in tokenized_items.items()}
                q_batch, v_batch = self.asym_sbert(batch)
                Q_parts.append(q_batch.cpu())
                V_parts.append(v_batch.cpu())

        Q = torch.nn.functional.normalize(torch.cat(Q_parts, dim=0), dim=-1).numpy()
        V = torch.nn.functional.normalize(torch.cat(V_parts, dim=0), dim=-1).numpy()
        return Q, V

    @staticmethod
    def _compute_S(Q: np.ndarray, V: np.ndarray, lambda_s: float) -> np.ndarray:
        n = Q.shape[0]
        Q = Q.astype(np.float64)
        V = V.astype(np.float64)
        G_Q = Q @ Q.T
        P_Q = np.linalg.inv(G_Q + lambda_s * np.eye(n))
        M   = P_Q @ (Q @ V.T)
        mu  = np.diag(M) / np.diag(P_Q)
        S   = M - P_Q * mu[np.newaxis, :]
        np.fill_diagonal(S, 0.0)
        return S.astype(np.float32)

    @staticmethod
    def _compute_B(
        X: sp.csr_matrix, S: np.ndarray, lambda_b: float, lambda_r: float,
    ) -> np.ndarray:
        n     = X.shape[1]
        use_S = lambda_r > 0.0
        G_X   = (X.T @ X).toarray().astype(np.float64)
        P     = np.linalg.inv(G_X + (lambda_b + (lambda_r if use_S else 0.0)) * np.eye(n))
        if use_S:
            S  = S.astype(np.float64)
            PS = P @ S
            mu = (1.0 + lambda_r * np.diag(PS)) / np.diag(P)
            B  = np.eye(n) + lambda_r * PS - P * mu[np.newaxis, :]
        else:
            B  = np.eye(n) - P / np.diag(P)[np.newaxis, :]
        np.fill_diagonal(B, 0.0)
        return B.astype(np.float32)

    def fit_closed_form(self, X_train: sp.csr_matrix) -> SparseKerasEASE:
        print("AsymBeeformerL3AE: encoding items with frozen beeformer...")
        Q, V = self._encode_all_items()

        print(f"AsymBeeformerL3AE: computing S  (λ_s={self.lambda_s})...")
        S = self._compute_S(Q, V, self.lambda_s)

        print(f"AsymBeeformerL3AE: computing B  (λ_b={self.lambda_b}, λ_r={self.lambda_r})...")
        B = self._compute_B(X_train, S, self.lambda_b, self.lambda_r)

        print("AsymBeeformerL3AE: done.")
        return SparseKerasEASE(self.items_idx, self.device, B=torch.from_numpy(B))


class L3AE(keras.models.Model):

    def __init__(
        self,
        sbert,
        device,
        tokenized_sentences,
        items_idx,
        lambda_s: float = 1.0,
        lambda_b: float = 500.0,
        lambda_r: float = 10.0,
        sbert_batch_size: int = 128,
    ):
        super().__init__()
        self.device = device
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.sbert_batch_size = sbert_batch_size
        self.lambda_s = lambda_s
        self.lambda_b = lambda_b
        self.lambda_r = lambda_r

        self.sbert = LayerSBERT(sbert, device, tokenized_sentences)
        for param in self.sbert.sbert.parameters():
            param.requires_grad_(False)

    def call(self, x):
        return self.sbert(x)

    def _encode_all_items(self) -> tuple[np.ndarray, np.ndarray]:
        tokenized_items = self.tokenized_sentences
        n_total = get_first_item(tokenized_items).shape[0]
        n_batches = math.ceil(n_total / self.sbert_batch_size)

        F_parts = []
        with torch.no_grad():
            for i in range(n_batches):
                lo = i * self.sbert_batch_size
                hi = lo + self.sbert_batch_size
                batch = {k: v[lo:hi] for k, v in tokenized_items.items()}
                f_batch = self.sbert(batch)
                F_parts.append(f_batch.cpu())

        F = torch.nn.functional.normalize(torch.cat(F_parts, dim=0), dim=-1).numpy()
        return F

    @staticmethod
    def _compute_S(F: np.ndarray, lambda_s: float) -> np.ndarray:
        n = F.shape[0]
        F = F.astype(np.float64)
        G_Q = F @ F.T
        P_Q = np.linalg.inv(G_Q + lambda_s * np.eye(n))
        M   = P_Q @ G_Q
        mu  = np.diag(M) / np.diag(P_Q)
        S   = M - P_Q * mu[np.newaxis, :]
        np.fill_diagonal(S, 0.0)
        return S.astype(np.float32)

    @staticmethod
    def _compute_B(
        X: sp.csr_matrix, S: np.ndarray, lambda_b: float, lambda_r: float,
    ) -> np.ndarray:
        n     = X.shape[1]
        use_S = lambda_r > 0.0
        G_X   = (X.T @ X).toarray().astype(np.float64)
        P     = np.linalg.inv(G_X + (lambda_b + (lambda_r if use_S else 0.0)) * np.eye(n))
        if use_S:
            S  = S.astype(np.float64)
            PS = P @ S
            mu = (1.0 + lambda_r * np.diag(PS)) / np.diag(P)
            B  = np.eye(n) + lambda_r * PS - P * mu[np.newaxis, :]
        else:
            B  = np.eye(n) - P / np.diag(P)[np.newaxis, :]
        np.fill_diagonal(B, 0.0)
        return B.astype(np.float32)

    def fit_closed_form(self, X_train: sp.csr_matrix) -> SparseKerasEASE:
        print("L3AE: encoding items with frozen encoder...")
        F = self._encode_all_items()

        print(f"L3AE: computing S  (λ_s={self.lambda_s})...")
        S = self._compute_S(F, self.lambda_s)

        print(f"L3AE: computing B  (λ_b={self.lambda_b}, λ_r={self.lambda_r})...")
        B = self._compute_B(X_train, S, self.lambda_b, self.lambda_r)

        print("L3AE: done.")
        return SparseKerasEASE(self.items_idx, self.device, B=torch.from_numpy(B))