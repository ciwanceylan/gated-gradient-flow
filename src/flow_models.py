from typing import *

import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats


class SheafFlowPlusPlus(nn.Module):

    def __init__(self,
                 num_nodes,
                 embedding_dim: int,
                 use_simple_gates: bool,
                 embedding_init: Union[str, torch.Tensor, np.ndarray] = 'zeros',
                 gates_init: Union[str, torch.Tensor, np.ndarray] = 'zeros',
                 beta: float = 1.,
                 ):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        self.node_embeddings.weight.data = initalization_switch(embedding_init, num_nodes, embedding_dim)
        self.node_embeddings.float()

        self.gates = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        self.gates.weight.data = initalization_switch(gates_init, num_nodes, embedding_dim)
        self.gates.float()
        self.beta = beta
        self.use_simple_gates = use_simple_gates

    def get_parameter_clones(self):
        return self.node_embeddings.weight.detach().cpu().clone(), self.gates.weight.detach().cpu().clone()

    def gates_forward(self, source_nodes: torch.LongTensor, target_nodes: torch.LongTensor):
        if self.use_simple_gates:
            gates = torch.sigmoid(self.beta * (self.gates(target_nodes) + self.gates(source_nodes)))
        else:
            gates = torch.sigmoid(self.beta * (self.gates(target_nodes) * self.gates(source_nodes) +
                                               self.gates(target_nodes) + self.gates(source_nodes)))
        return gates

    def gradient_forward(self, source_nodes: torch.LongTensor, target_nodes: torch.LongTensor):
        w_arriving = self.node_embeddings(target_nodes)
        w_leaving = self.node_embeddings(source_nodes)
        gradient = w_arriving - w_leaving
        return gradient

    def forward(self, source_nodes: torch.LongTensor, target_nodes: torch.LongTensor, mode=None):
        gradient = self.gradient_forward(source_nodes, target_nodes)  # [num_edges x embedding_dim]
        gates = self.gates_forward(source_nodes, target_nodes)  # [num_edges x embedding_dim]

        out = torch.sum(gates * gradient, dim=-1)  # [num_edges]
        return out


class LinearRegressionBaseline(nn.Module):

    def __init__(self, features: np.ndarray, normalize_features=False):
        super().__init__()
        num_nodes, num_features = features.shape
        self.features_mean = np.mean(features, axis=0)
        self.features_std = np.std(features, axis=0)
        if normalize_features:
            features = (features - self.features_mean) / self.features_std
        self.node_features = torch.from_numpy(features).to(torch.float)
        self.reg_model = nn.Sequential(nn.Linear(in_features=2 * num_features, out_features=1))

    def forward(self, source_nodes: torch.LongTensor, target_nodes: torch.LongTensor):
        flow_features = torch.cat((
            self.node_features[source_nodes, :],
            self.node_features[target_nodes, :]
        ), dim=1)
        return self.reg_model(flow_features).view(-1)

    def _apply(self, fn):
        super(LinearRegressionBaseline, self)._apply(fn)
        self.node_features = fn(self.node_features)
        return self


class DNNRegressionBaseline(nn.Module):

    def __init__(self, features: np.ndarray, num_layers=1, normalize_features=False):
        super().__init__()
        num_nodes, num_features = features.shape
        self.features_mean = np.mean(features, axis=0)
        self.features_std = np.std(features, axis=0)
        if normalize_features:
            features = (features - self.features_mean) / self.features_std
        # self.node_features = nn.Parameter(torch.from_numpy(features).to(torch.float), requires_grad=False)
        self.node_features = torch.from_numpy(features).to(torch.float)
        modules = []
        for _ in range(num_layers):
            modules.append(nn.Linear(in_features=2 * num_features, out_features=2 * num_features))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=2 * num_features, out_features=1))
        self.reg_model = nn.Sequential(*modules)

    def forward(self, source_nodes: torch.LongTensor, target_nodes: torch.LongTensor):
        flow_features = torch.cat((
            self.node_features[source_nodes, :],
            self.node_features[target_nodes, :]
        ), dim=1)
        return self.reg_model(flow_features).view(-1)

    def _apply(self, fn):
        super(DNNRegressionBaseline, self)._apply(fn)
        self.node_features = fn(self.node_features)
        return self


def initalization_switch(init_option, num_nodes, embedding_dim):
    if isinstance(init_option, torch.Tensor):
        return init_option.detach().clone()
    elif isinstance(init_option, np.ndarray):
        return torch.from_numpy(init_option)
    elif init_option == 'zero' or init_option == 'zeros':
        return torch.zeros(num_nodes, embedding_dim)
    elif init_option == 'one' or init_option == 'ones':
        return torch.ones(num_nodes, embedding_dim)
    elif init_option == 'normal':
        return torch.randn(num_nodes, embedding_dim)
    else:
        return torch.zeros(num_nodes, embedding_dim)


def sample_t_embeddings(num_nodes, embedding_dim, df=3):
    r = stats.t.rvs(df=df, size=(num_nodes, embedding_dim)).astype(np.float32)
    return torch.from_numpy(r)
