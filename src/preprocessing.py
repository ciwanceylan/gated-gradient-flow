from typing import *

import graph_tool.all as gt
import numpy as np
from scipy import sparse as sp
import src.dataprocessing as dataproc

GT_TYPES_WITH_ARRAY = {'bool', 'int16_t', 'int32_t', 'int64_t', 'double', 'long double'}


class UntrimableGraph(Exception):
    pass


def preprocess_flow_graph(graph, trim_size_threshold=0, resolve_methods=None, remove_loop=True, trim=True,
                          verbosity=0):
    if verbosity > 0:
        print("Extracting lcc...")
    graph = gt.extract_largest_component(graph, directed=False, prune=True)

    if verbosity > 0:
        print("Making edges canonical...")
    graph = to_canonical_directed_edges(graph, resolve_methods=resolve_methods,
                                        remove_self_loops=remove_loop)
    if trim:
        graph = trim_graph(graph, trim_size_threshold, verbosity=verbosity)

    graph = add_edgeindex_pmap(graph)
    graph = add_min_deg_pmap(graph)
    return graph


def trim_graph(graph, size_threshold, verbosity=0):
    degmap = graph.degree_property_map('total')
    if verbosity > 0:
        print("trimming graph... this may take some time")
    i = 0
    while degmap.a.min() < 2:
        filter_ = graph.new_vertex_property('bool')
        filter_.a = degmap.a > 1
        num_left = filter_.a.astype(int).sum()
        if num_left < size_threshold:
            raise UntrimableGraph(f"The graph was trimmed below threshold {size_threshold}")
        graph = gt.extract_largest_component(gt.GraphView(graph, vfilt=filter_), directed=False, prune=True)
        degmap = graph.degree_property_map('total')
        i += 1
        if verbosity > 1 and i % 9 == 0:
            print(f"Made {i + 1} trims. {graph.num_vertices()} nodes left in graph")
    return graph


def to_canonical_directed_edges(graph: gt.Graph,
                                resolve_methods: Optional[Dict[str, Union[str, Callable]]] = None,
                                remove_self_loops=True):
    if resolve_methods is None:
        resolve_methods = dict()
    canonical_graph = gt.Graph(gt.GraphView(graph, efilt=graph.new_ep('bool', False)), directed=True, prune=True)
    canonical_graph.set_directed(True)

    adj_coo = gt.adjacency(graph).tocoo().T
    canonical_edges = __get_canonical_edges(adj_coo)
    anti_canonical_edges = __get_anti_canonical_edges(adj_coo)

    new_canonical_edges = np.unique(
        np.concatenate(
            (canonical_edges, np.flip(anti_canonical_edges, axis=1)),
            axis=0
        ), axis=0
    )
    canonical_graph.add_edge_list(new_canonical_edges)

    graph_edge2index = create_edge2index(graph)
    canonical_graph_edge2index = create_edge2index(canonical_graph)

    can_index_can_edges = edges2indices(canonical_edges, canonical_graph_edge2index)
    graph_index_can_edges = edges2indices(canonical_edges, graph_edge2index)
    can_index_anti_can_edges = edges2indices(np.flip(anti_canonical_edges, axis=1), canonical_graph_edge2index)
    graph_index_anti_can_edges = edges2indices(anti_canonical_edges, graph_edge2index)

    for key in graph.ep:
        if canonical_graph.ep[key].value_type() not in GT_TYPES_WITH_ARRAY:
            continue
        canonical_graph.ep[key].a[can_index_can_edges] = graph.ep[key].a[graph_index_can_edges]
        resolve_method = resolve_methods.get(key, 'plus')
        if resolve_method == 'minus' or resolve_method == '-' or resolve_method == 'subtraction':
            canonical_graph.ep[key].a[can_index_anti_can_edges] -= graph.ep[key].a[graph_index_anti_can_edges]
        else:
            canonical_graph.ep[key].a[can_index_anti_can_edges] += graph.ep[key].a[graph_index_anti_can_edges]

    return canonical_graph


def create_edge2index(graph):
    return dict(zip(((int(s), int(t)) for s, t in graph.edges()), range(graph.num_edges())))


def edges2indices(edges, edge2index):
    return list(filter(lambda x: x is not None, map(lambda st: edge2index.get(tuple(st), None), edges)))


def __basic_resolve_funs(value1, value2, resolve_method='plus'):
    if isinstance(resolve_method, Callable):
        return resolve_method(value1, value2)

    if resolve_method == 'minus' or resolve_method == 'subtraction' or resolve_method == '-':
        value1 = 0 if value1 is None else value1
        value2 = 0 if value2 is None else value2
        return value1 - value2
    elif resolve_method == 'plus' or resolve_method == 'addition' or resolve_method == '+':
        value1 = 0 if value1 is None else value1
        value2 = 0 if value2 is None else value2
        return value1 + value2
    elif resolve_method == 'times' or resolve_method == 'multiplication' or resolve_method == '*':
        value1 = 1 if value1 is None else value1
        value2 = 1 if value2 is None else value2
        return value1 * value2
    elif resolve_method == 'max':
        value1 = -float('inf') if value1 is None else value1
        value2 = -float('inf') if value2 is None else value2
        return max(value1, value2)
    elif resolve_method == 'min':
        value1 = float('inf') if value1 is None else value1
        value2 = float('inf') if value2 is None else value2
        return min(value1, value2)
    else:
        raise ValueError(f"Unknown resolve_choice {resolve_method}")


def __get_canonical_edges(adj_coo):
    adj_upper = sp.triu(adj_coo, k=1)
    assert np.all(adj_upper.row < adj_upper.col)
    edges = np.stack((adj_upper.row, adj_upper.col), axis=1)
    return edges


def __get_anti_canonical_edges(adj_coo):
    adj_lower = sp.tril(adj_coo, k=-1)
    assert np.all(adj_lower.row > adj_lower.col)
    edges = np.stack((adj_lower.row, adj_lower.col), axis=1)
    return edges


def __get_nodes_with_self_loops(adj_coo):
    nodes = adj_coo.diagonal().nonzero()[0]
    return nodes


def add_edgeindex_pmap(graph: gt.Graph, add_sources_and_targets=True, use64bit=False):
    if use64bit:
        index_ep = graph.new_edge_property('int64_t')
        source_ep = graph.new_edge_property('int64_t')
        target_ep = graph.new_edge_property('int64_t')
    else:
        index_ep = graph.new_edge_property('int32_t')
        source_ep = graph.new_edge_property('int32_t')
        target_ep = graph.new_edge_property('int32_t')
    if add_sources_and_targets:
        for i, e in enumerate(graph.edges()):
            index_ep[e] = i
            source_ep[e] = int(e.source())
            target_ep[e] = int(e.target())
        graph.ep.index = index_ep
        graph.ep.source = source_ep
        graph.ep.target = target_ep
    else:
        for i, e in enumerate(graph.edges()):
            index_ep[e] = i
        graph.ep.index = index_ep
    return graph


def add_min_deg_pmap(graph: gt.Graph):
    min_deg_ep = graph.new_edge_property('int')
    min_deg_ep.a = np.minimum(
        graph.degree_property_map('total').a[graph.ep.source.a],
        graph.degree_property_map('total').a[graph.ep.target.a])
    graph.ep.min_deg = min_deg_ep
    return graph


def gt2dataprocgraph(graph: gt.Graph, flow_ep_name=None) -> dataproc.Graph:
    edges = np.asarray(np.concatenate((graph.ep.source.a, graph.ep.target.a)))
    num_nodes = graph.num_vertices()
    if flow_ep_name is not None:
        flow = np.asarray(graph.ep[flow_ep_name].a)
    else:
        flow = None
    return dataproc.Graph(num_nodes=num_nodes, edges=edges, flow=flow)

# def split_train_val_test(graph: gt.Graph,
#                          desired_split=(0.7, 0.2, 0.1),
#                          required_train: Optional[gt.EdgePropertyMap] = None,
#                          required_val: Optional[gt.EdgePropertyMap] = None,
#                          required_test: Optional[gt.EdgePropertyMap] = None,
#                          return_graphs=False
#                          ) -> Union[Tuple[gt.Graph, gt.Graph, gt.Graph], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     num_edges = graph.num_edges()
#
#     num_train, num_val, num_test = num_edges * np.concatenate(
#         (np.asarray(desired_split)[:3] / np.sum(desired_split),
#          np.zeros(max(0, 3 - len(desired_split)))))
#
#     num_val = int(num_val)
#     num_test = int(num_test)
#     num_train = num_edges - num_val - num_test
#
#     train_filter = graph.new_edge_property('bool', False) if required_train is None else required_train
#     val_filter = graph.new_edge_property('bool', False) if required_val is None else required_val
#     test_filter = graph.new_edge_property('bool', False) if required_test is None else required_test
#
#     rand_min_tree = gtutils.random_min_spanning_tree(graph)
#     train_filter.a = np.logical_or(train_filter.a, rand_min_tree.a)
#
#     remaining_edge_indices = np.random.permutation(np.logical_not(
#         train_filter.a | val_filter.a | test_filter.a).nonzero()[0])
#
#     if len(remaining_edge_indices) > 0:
#         current_num_train = train_filter.a.sum().item()
#         num_additional_train_edges = max(num_train - current_num_train, 0)
#         train_filter.a[remaining_edge_indices[:num_additional_train_edges]] = True
#         remaining_edge_indices = remaining_edge_indices[num_additional_train_edges:]
#
#         num_val = int((float(num_val) / (num_val + num_test)) * len(remaining_edge_indices))
#         num_additional_val = max(num_val - val_filter.a.sum().item(), 0)
#         val_filter.a[remaining_edge_indices[:num_additional_val]] = True
#         remaining_edge_indices = remaining_edge_indices[num_additional_val:]
#
#         test_filter.a[remaining_edge_indices] = True
#
#     if return_graphs:
#
#         train_graph = gt.Graph(gt.GraphView(graph, efilt=train_filter), prune=True)
#         val_graph = gt.Graph(gt.GraphView(graph, efilt=val_filter), prune=True)
#         test_graph = gt.Graph(gt.GraphView(graph, efilt=test_filter), prune=True)
#         return train_graph, val_graph, test_graph
#     else:
#         train_edges = train_filter.a.nonzero()[0]
#         val_edges = val_filter.a.nonzero()[0]
#         test_edges = test_filter.a.nonzero()[0]
#         return train_edges, val_edges, test_edges
