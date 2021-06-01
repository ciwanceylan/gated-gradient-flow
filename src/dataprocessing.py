from typing import *

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import sparse as sp
import torch


class Graph(object):

    def __init__(self, num_nodes: int, edges: np.ndarray, flow: np.ndarray = None):
        self._verify(num_nodes, edges, flow)
        self._num_nodes = num_nodes
        self.edges = edges.astype(np.int64)
        self._flow = flow
        self._degrees = np.zeros(num_nodes)
        nodes, degree = np.unique(np.concatenate((self.edges[:, 0], self.edges[:, 1])), return_counts=True)
        self._degrees[nodes] = degree
        if flow is not None:
            self.set_flow(flow)
        self._indexed_edge = pd.Series(data=range(len(self.edges)), index=self._edges2index(self.edges))

    def set_flow(self, flow: np.ndarray):
        assert len(flow) == self.num_edges
        self._flow = flow.astype(np.float32)

    @staticmethod
    def _verify(num_nodes, edges, flow):
        if len(edges) == 0:
            return
        assert edges.max() < num_nodes
        assert (edges[:, 0] <= edges[:, 1]).all()

    @classmethod
    def read_csv(cls, path):
        with open(path, 'r') as fp:
            num_nodes_line = fp.readline()
            num_nodes = int(num_nodes_line.strip("#\n"))
            df = pd.read_csv(fp, names=["src", "dst", "flow"],
                             dtype={"src": np.int64, "dst": np.int64, "flow": np.float64})
        flow = None if df.loc[:, "flow"].isnull().all() else df.loc[:, "flow"].to_numpy()
        return cls(num_nodes, edges=df.loc[:, ["src", "dst"]].to_numpy(), flow=flow)

    def to_csv(self, path):
        with open(path, 'w') as fp:
            fp.write("#" + str(self.num_nodes) + "\n")
            pd.DataFrame(
                {"src": self.edges[:, 0], "dst": self.edges[:, 1], "flow": self.flow}
            ).to_csv(fp, mode='a', header=False, index=False)

    @property
    def degrees(self):
        return self._degrees

    @property
    def src_nodes(self):
        return self.edges[:, 0]

    @property
    def dst_nodes(self):
        return self.edges[:, 1]

    @property
    def flow(self):
        return self._flow

    @property
    def num_nodes(self):
        return self._num_nodes

    def num_vertices(self):
        return self.num_nodes

    @property
    def num_edges(self):
        return self.edges.shape[0]

    def _edges2index(self, edges):
        return edges[:, 0] * self.num_nodes + edges[:, 1]

    def edges2index(self, edges):
        return self._indexed_edge[self._edges2index(edges)].to_numpy()

    def grad_matrix(self):
        columns = np.concatenate((self.dst_nodes, self.src_nodes), axis=0)
        values = np.concatenate((np.ones(self.num_edges), -np.ones(self.num_edges)), axis=0)

        rows = np.concatenate((np.arange(self.num_edges), np.arange(self.num_edges)))
        grad_matrix = sp.coo_matrix((values, (rows, columns)), shape=(self.num_edges, self.num_nodes)).tocsr()
        return grad_matrix

    def split_train_val_test_filters(self,
                                     desired_split=(0.7, 0.2, 0.1),
                                     required_train: Optional[np.ndarray] = None,
                                     required_val: Optional[np.ndarray] = None,
                                     required_test: Optional[np.ndarray] = None
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_train, num_val, num_test = self.num_edges * np.concatenate(
            (np.asarray(desired_split)[:3] / np.sum(desired_split),
             np.zeros(max(0, 3 - len(desired_split)))))

        num_val = int(num_val)
        num_test = int(num_test)
        num_train = self.num_edges - num_val - num_test

        train_filter = np.zeros(self.num_edges, dtype=bool) if required_train is None else required_train
        val_filter = np.zeros(self.num_edges, dtype=bool) if required_val is None else required_val
        test_filter = np.zeros(self.num_edges, dtype=bool) if required_test is None else required_test

        tree_edges = self.random_min_spanning_tree()
        tree_indices = self.edges2index(tree_edges)
        train_filter[tree_indices] = True

        remaining_edge_indices = np.random.permutation(
            np.logical_not(train_filter | val_filter | test_filter).nonzero()[0]
        )

        if len(remaining_edge_indices) > 0:
            current_num_train = int(train_filter.sum().item())
            num_additional_train_edges = max(num_train - current_num_train, 0)
            train_filter[remaining_edge_indices[:num_additional_train_edges]] = True
            remaining_edge_indices = remaining_edge_indices[num_additional_train_edges:]

            num_val = int((float(num_val) / (num_val + num_test)) * len(remaining_edge_indices))
            num_additional_val = max(num_val - int(val_filter.sum().item()), 0)
            val_filter[remaining_edge_indices[:num_additional_val]] = True
            remaining_edge_indices = remaining_edge_indices[num_additional_val:]

            test_filter[remaining_edge_indices] = True

        return train_filter, val_filter, test_filter

    def split_train_val_test_edges(self,
                                   desired_split=(0.7, 0.2, 0.1),
                                   required_train: Optional[np.ndarray] = None,
                                   required_val: Optional[np.ndarray] = None,
                                   required_test: Optional[np.ndarray] = None
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        train_filter, val_filter, test_filter = self.split_train_val_test_filters(
            desired_split=desired_split, required_train=required_train,
            required_val=required_val, required_test=required_test
        )

        train_edges = self.edges[train_filter]
        val_edges = self.edges[val_filter]
        test_edges = self.edges[test_filter]
        return train_edges, val_edges, test_edges

    def split_train_val_test_graphs(self,
                                    desired_split=(0.7, 0.2, 0.1),
                                    required_train: Optional[np.ndarray] = None,
                                    required_val: Optional[np.ndarray] = None,
                                    required_test: Optional[np.ndarray] = None
                                    ) -> Tuple['Graph', 'Graph', 'Graph']:
        train_filter, val_filter, test_filter = self.split_train_val_test_filters(
            desired_split=desired_split, required_train=required_train,
            required_val=required_val, required_test=required_test
        )

        train_edges = self.edges[train_filter]
        val_edges = self.edges[val_filter]
        test_edges = self.edges[test_filter]

        if self.flow is not None:
            train_graph = Graph(self.num_nodes, train_edges, flow=self.flow[train_filter])
            val_graph = Graph(self.num_nodes, val_edges, flow=self.flow[val_filter])
            test_graph = Graph(self.num_nodes, test_edges, flow=self.flow[test_filter])
        else:
            train_graph = Graph(self.num_nodes, train_edges)
            val_graph = Graph(self.num_nodes, val_edges)
            test_graph = Graph(self.num_nodes, test_edges)

        return train_graph, val_graph, test_graph

    def random_min_spanning_tree(self):
        random_weights = np.random.rand(self.num_edges)
        adj = sp.coo_matrix(
            (random_weights, (self.src_nodes, self.dst_nodes)), shape=(self.num_nodes, self.num_nodes)
        ).tocsr()
        min_tree = sp.csgraph.minimum_spanning_tree(adj).tocoo()
        return np.stack((min_tree.row, min_tree.col), axis=1).astype(np.int64)

    def __repr__(self):
        return '{}({} nodes, {} edges)'.format(self.__class__.__name__, self.num_nodes, self.num_edges)


# ========================================================================================
# ===========================  Generate random networks ==================================
# ========================================================================================

def complete_graph(num_nodes: int):
    lower_triu_max = sp.triu(np.ones((num_nodes, num_nodes)), k=1, format='coo')
    edges = np.stack((lower_triu_max.row, lower_triu_max.col), axis=1)
    return Graph(num_nodes=num_nodes, edges=edges)


def sample_noise(num_samples: int, dim: int, df: Union[str, float]):
    if df == 'normal' or df > 10:
        r = stats.norm.rvs(size=(num_samples, dim)).astype(np.float32)
    else:
        r = stats.t.rvs(df=df, size=(num_samples, dim)).astype(np.float32)
    return r


def add_sheaf_flow(graph: Graph, sheaf_flow_model):
    sources, targets = graph.src_nodes, graph.dst_nodes
    sources_torch = torch.from_numpy(sources).to(sheaf_flow_model.node_embeddings.weight.device)
    targets_torch = torch.from_numpy(targets).to(sheaf_flow_model.node_embeddings.weight.device)
    with torch.no_grad():
        gt_flow = sheaf_flow_model(sources_torch, targets_torch)
    graph.set_flow(gt_flow.detach().cpu().numpy())

    grad_flow, harmonic_flow = decompose_flow(sources, targets, graph.flow, graph.num_nodes)
    grad_norm = np.linalg.norm(grad_flow)
    harmonic_norm = np.linalg.norm(harmonic_flow)
    summary = flow_summary(graph.flow)
    summary.update({'grad_norm': grad_norm, 'harmonic_norm': harmonic_norm})
    return graph, summary


def flow_summary(flow, cutoff: float = 1e-10):
    ratio_positive = (np.sign(flow) > 0).sum() / len(flow)
    mean_magnitude = np.mean(np.log10(np.maximum(np.abs(flow), cutoff)))
    mean_value = np.mean(np.abs(flow))
    max_value = np.max(np.abs(flow))
    flow_std = np.std(flow)
    flow_norm = np.linalg.norm(flow)
    return {'num_edges': len(flow), 'flow_ratio_p': ratio_positive, 'flow_mean_mag': mean_magnitude,
            'flow_mean_value': mean_value, 'flow_max_value': max_value, 'flow_std': flow_std,
            'flow_norm': flow_norm}


def compute_indicence_matrix(source_nodes, target_nodes, num_nodes, num_edges):
    leaving = source_nodes
    arriving = target_nodes
    columns = np.concatenate((arriving, leaving), axis=0)
    values = np.concatenate((np.ones(num_edges), -np.ones(num_edges)), axis=0)

    rows = np.concatenate((np.arange(num_edges), np.arange(num_edges)))
    incidence_matrix = sp.coo_matrix((values, (rows, columns)), shape=(num_edges, num_nodes)).tocsr()
    return incidence_matrix


def compute_flow_potentials(source_nodes, target_nodes, flow, num_nodes, return_indicence=False):
    num_edges = len(flow)
    incidence_matrix = compute_indicence_matrix(source_nodes, target_nodes, num_nodes, num_edges)
    potentials, istop, itn, normr = sp.linalg.lsqr(incidence_matrix, flow)[:4]
    if return_indicence:
        return potentials, incidence_matrix
    return potentials


def decompose_flow(source_nodes, target_nodes, flow, num_nodes):
    potentials, incidence_matrix = compute_flow_potentials(source_nodes, target_nodes, flow, num_nodes,
                                                           return_indicence=True)

    grad_flow = incidence_matrix.dot(potentials)
    harmonic_flow = flow - grad_flow

    return grad_flow, harmonic_flow


def decompose_flow_normalized(source_nodes, target_nodes, flow, num_nodes):
    grad_flow, harmonic_flow = decompose_flow(source_nodes, target_nodes, flow, num_nodes)

    grad_norm = np.linalg.norm(grad_flow)
    normalized_grad_flow = grad_flow / grad_norm
    harmonic_norm = np.linalg.norm(harmonic_flow)
    normalized_harmonic_flow = harmonic_flow / harmonic_norm

    return normalized_grad_flow, grad_norm, normalized_harmonic_flow, harmonic_norm


def sample_gmm_from_mu(n, mu, std):
    k, dim = mu.shape
    num_repeats = int(np.ceil(n / k))

    samples = np.repeat(mu, num_repeats, 0)
    samples += std * np.random.randn(*samples.shape)

    component_ids = np.repeat(np.arange(k), num_repeats)
    permutation = np.random.permutation(samples.shape[0])[:n]

    return samples[permutation, :], component_ids[permutation]


def sample_t_from_mu(n, mu, std):
    k, dim = mu.shape
    num_repeats = int(np.ceil(n / k))

    samples = np.repeat(mu, num_repeats, 0)
    samples += std * stats.t(df=2).rvs(size=samples.shape)

    component_ids = np.repeat(np.arange(k), num_repeats)
    permutation = np.random.permutation(samples.shape[0])[:n]

    return samples[permutation, :], component_ids[permutation]


def sample_t_from_mu_and_ids(mu, std, component_ids):
    comps, inverse = np.unique(component_ids, return_inverse=True)
    samples = mu[inverse, :]
    samples += std * stats.t(df=2).rvs(size=samples.shape)

    return samples, component_ids


def sample_gmm_from_mu_and_ids(mu, std, component_ids):
    comps, inverse = np.unique(component_ids, return_inverse=True)
    samples = mu[inverse, :]
    samples += std * np.random.randn(*samples.shape)

    return samples, component_ids


def sample_2modes_2d(n, std, num_sigma=6, num_cross_dims=2, component_ids=None):
    radius = num_sigma * std / 2
    if num_cross_dims > 1:
        mu = np.array([[radius, radius], [-radius, -radius]])
    elif num_cross_dims == 1:
        mu = np.array([[radius, 0], [-radius, 0]])
    else:
        mu = np.zeros((2, 2))
    if component_ids is None:
        samples, component_ids = sample_gmm_from_mu(n, mu, std)
    else:
        samples, _ = sample_gmm_from_mu_and_ids(mu, std, component_ids)
    return samples, component_ids


def sample_trade_scenario(scenario: str, num_nodes):
    scenarios = {'0:0', '0:1', '1:0', '1:1', '2:2'}
    if scenario not in scenarios:
        raise RuntimeError(f"Invalid scenario {scenario}")
    emb_scenario, gates_scenario = scenario.split(':')
    emb_std = 1.
    if emb_scenario == '0':
        emb_mu = np.array([[-100., -20.], [100., 20.]])
    elif emb_scenario == '1':
        emb_mu = np.array([[-100., 20.], [100., -20.]])
    elif emb_scenario == '2':
        emb_mu = np.array([[100., 100.], [0., 0.], [0., -50]])
    else:
        raise RuntimeError(f"Invalid scenario {scenario}")

    emb_samples, emb_comp_ids = sample_t_from_mu(num_nodes, emb_mu, emb_std)

    gates_std = 0.01

    if gates_scenario == '0':
        gates_mu = np.array([[-1., -1.]])
        gates_samples, gates_comp_ids = sample_gmm_from_mu(num_nodes, gates_mu, gates_std)
        num_traders = int(0.5 * num_nodes)
        traders = np.random.choice(num_nodes, size=num_traders, replace=False)
        traders1 = traders[:num_traders // 2]
        traders2 = traders[num_traders // 2:]
        gates_samples[traders1, 0] = 4. * (np.ones((len(traders1),)) + gates_std * np.random.randn(len(traders1)))
        gates_samples[traders2, 1] = 4. * (np.ones((len(traders2),)) + gates_std * np.random.randn(len(traders2)))
    elif gates_scenario == '2':
        gates_mu = np.array([[4., 4.], [0., -5.], [-5., 1.]])
        gates_samples, gates_comp_ids = sample_t_from_mu_and_ids(gates_mu, gates_std, emb_comp_ids)

    return emb_samples, gates_samples, emb_comp_ids, gates_comp_ids


def sample_scenario(n, scenario,
                    num_emb_cross_dim,
                    emb_std, gates_std,
                    emb_num_sigma=16, gates_num_sigma=16):
    node_embeddings, node_comp_ids = sample_2modes_2d(n, std=emb_std, num_sigma=emb_num_sigma,
                                                      num_cross_dims=num_emb_cross_dim)
    if scenario == "inter":
        gates, gate_ids = sample_2modes_2d(n, std=gates_std, num_sigma=gates_num_sigma,
                                           num_cross_dims=0,
                                           component_ids=node_comp_ids)
    elif scenario == "intra":
        gates, gate_ids = sample_2modes_2d(n, std=gates_std, num_sigma=gates_num_sigma,
                                           num_cross_dims=num_emb_cross_dim,
                                           component_ids=None)
    else:
        raise ValueError(f"Unknown scenario {scenario}")
    return node_embeddings, node_comp_ids, gates, gate_ids
