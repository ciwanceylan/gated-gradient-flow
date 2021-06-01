import graph_tool.all as gt
from src import preprocessing as preproc


def prepare_ethereum():
    print("Preprocessing ethereum data...")
    ethereum_raw_data_path = 'data/unprocessed_ethereum_2018_2020'

    graph = gt.load_graph_from_csv(ethereum_raw_data_path, directed=True,
                                   eprop_types=['float', 'float', 'int'],
                                   eprop_names=['value', 'gas', 'count'], skip_first=True)

    graph = preproc.preprocess_flow_graph(graph=graph, trim_size_threshold=0, resolve_methods={'value': 'minus'},
                                          remove_loop=True, trim=True, verbosity=2)
    graph.save("data/preprocessed_ethereum_2018_2020.gt")
    preproc.gt2dataprocgraph(graph, flow_ep_name='value').to_csv("data/preprocessed_ethereum_2018_2020.csv")


def prepare_bitcoin():
    print("Preprocessing bitcoin data...")
    g = gt.collection.ns["bitcoin"]
    e_filter = g.new_edge_property('bool', False)
    e_filter.a = g.ep.count.a > 10
    reduced_bitcoin_graph = gt.Graph(gt.GraphView(g, efilt=e_filter), prune=True)
    graph = preproc.preprocess_flow_graph(reduced_bitcoin_graph, verbosity=2)
    graph.save("data/preprocessed_bitcoin.gt")
    preproc.gt2dataprocgraph(graph).to_csv("data/preprocessed_bitcoin.csv")


def prepare_cora():
    print("Preprocessing cora data...")
    g = gt.collection.ns["cora"]
    graph = preproc.preprocess_flow_graph(g, verbosity=2)
    graph.save("data/preprocessed_cora.gt")
    preproc.gt2dataprocgraph(graph).to_csv("data/preprocessed_cora.csv")


if __name__ == "__main__":
    prepare_ethereum()
    prepare_cora()
    prepare_bitcoin()
