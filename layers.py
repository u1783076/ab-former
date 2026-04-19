import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import sentence_transformers
import torch
import torch.nn as nn

from keras.layers import TorchModuleWrapper
from images import ImageModel
from add_modules import asym_head_parameter_ids


class LayerELSA(keras.layers.Layer):
    def __init__(self, n_dims, n_items, device):
        super(LayerELSA, self).__init__()
        self.device = device
        self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_dims, n_items])))

    def parameters(self, recurse=True):
        return [self.A]

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self):
        self.to(self.device)
        sample_input = torch.ones([self.A.shape[0]]).to(self.device)
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        A = torch.nn.functional.normalize(self.A, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return keras.activations.relu(xAAT - x)


class LayerAsymELSA(keras.layers.Layer):
    def __init__(self, n_items, n_dims, device):
        super(LayerAsymELSA, self).__init__()
        self.device = device
        self.Q = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))
        self.V = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))

    def parameters(self, recurse=True):
        return [self.Q, self.V]

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self):
        self.to(self.device)
        sample_input = torch.ones([self.Q.shape[0]]).to(self.device)
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        Q = torch.nn.functional.normalize(self.Q, dim=-1)
        V = torch.nn.functional.normalize(self.V, dim=-1)

        xQ = torch.matmul(x, Q)
        xQVT = torch.matmul(xQ, V.T)
        diag_qvt = (Q * V).sum(dim=-1)
        return keras.activations.relu(xQVT - x * diag_qvt)


class LayerSBERT(keras.layers.Layer):
    def __init__(self, model, device, tokenized_sentences):
        super(LayerSBERT, self).__init__()
        self.device = device
        self.sbert = TorchModuleWrapper(model.to(device))
        self.tokenize_ = self.sb().tokenize
        self.tokenized_sentences = tokenized_sentences
        self.build()

    def sb(self):
        for module in self.sbert.modules():
            if isinstance(module, sentence_transformers.SentenceTransformer) or isinstance(module, ImageModel):
                return module

    def parameters(self, recurse=True):
        return self.sbert.parameters()

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def tokenize(self, inp):
        return {k: v.to(self.device) for k, v in self.tokenize_(inp).items()}

    def build(self):
        self.to(self.device)
        sample_input = {k: v[:2].to(self.device) for k, v in self.tokenized_sentences.items()}
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        return self.sbert.forward(x)["sentence_embedding"]


class LayerAsymSBERT(keras.layers.Layer):
    """Keras wrapper around a SentenceTransformer with AsymProjectionModule, providing (Q, V) pairs."""

    def __init__(self, model: sentence_transformers.SentenceTransformer, device, tokenized_sentences):
        super().__init__()
        self.device = device
        self.sbert = TorchModuleWrapper(model.to(device))
        self.tokenize_ = self.sb().tokenize
        self.tokenized_sentences = tokenized_sentences
        self.build()

    def sb(self) -> sentence_transformers.SentenceTransformer:
        """Return the underlying SentenceTransformer (with AsymProjectionModule in its pipeline)."""
        for module in self.sbert.modules():
            if isinstance(module, sentence_transformers.SentenceTransformer):
                return module

    def parameters(self, recurse=True):
        return self.sbert.parameters()

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def tokenize(self, inp):
        return {k: v.to(self.device) for k, v in self.tokenize_(inp).items()}

    def build(self):
        self.to(self.device)
        sample_input = {k: v[:2].to(self.device) for k, v in self.tokenized_sentences.items()}
        _ = self.call(sample_input)
        self.track_module_parameters()
        self._asym_head_param_ids = asym_head_parameter_ids(self.sb())
        

    def call(self, x):
        features = self.sbert.forward(x)
        return features["q_sentence_embedding"], features["v_sentence_embedding"]
