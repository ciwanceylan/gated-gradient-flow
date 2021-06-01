'''
Code for the paper:
Edge Weight Prediction in Weighted Signed Networks.
Conference: ICDM 2016
Authors: Srijan Kumar, Francesca Spezzano, VS Subrahmanian and Christos Faloutsos

Author of code: Srijan Kumar
Email of code author: srijan@cs.stanford.edu

Update to python 3, and add train and eval functions: Ciwan Ceylan 2021-03-23
'''

import numpy as np
import pandas as pd
import networkx as nx
import math

import src.dataprocessing as dataproc
from src.evaluation import calc_flow_prediction_evaluation


class FGModel(object):

    def __init__(self, fairness, goodness, scale_factor, powerfactor):
        self.fg = pd.DataFrame({"fairness": fairness, "goodness": goodness}).sort_index(ascending=True)
        self.scale_factor = scale_factor
        self.powerfactor = powerfactor

    def predict(self, test_graph: dataproc.Graph):
        pred = (self.fg.loc[test_graph.src_nodes, "fairness"].to_numpy() *
                self.fg.loc[test_graph.dst_nodes, "goodness"].to_numpy())
        pred *= self.scale_factor
        pred = np.sign(pred) * (np.power(10, np.abs(pred)) - 1)
        pred /= self.powerfactor
        return pred

    @classmethod
    def read_csv(cls, path):
        with open(path, 'r') as fp:
            num_nodes_line = fp.readline()
            scale_factor, powerfactor = tuple(float(x) for x in num_nodes_line.strip("#\n").split("|"))
            fg = pd.read_csv(fp, dtype=[np.float64, np.float64])
        return cls(fg["fairness"], fg["goodness"], scale_factor=scale_factor, powerfactor=powerfactor)

    def to_csv(self, path):
        with open(path, 'w') as fp:
            fp.write("#" + str(self.scale_factor) + "|" + str(self.powerfactor) + "\n")
            self.fg.to_csv(fp, mode='a', header=True, index=True)


def single_train(train_graph, powerfactor, max_iter=100):
    train_source_nodes, train_target_nodes = train_graph.src_nodes, train_graph.dst_nodes
    train_flow = powerfactor * train_graph.flow
    train_flow = np.sign(train_flow) * np.log10(1 + np.abs(train_flow))
    scale_factor = np.abs(train_flow).max()
    weights = train_flow / scale_factor

    df = pd.DataFrame(data={
        'src': train_source_nodes,
        'dst': train_target_nodes,
        'weights': weights
    })
    edgelist = list(df.itertuples(index=False, name=None))

    G = nx.DiGraph()
    G.add_weighted_edges_from(edgelist)
    fairness, goodness = compute_fairness_goodness(G, max_iter=max_iter)
    return FGModel(fairness=fairness, goodness=goodness, scale_factor=scale_factor, powerfactor=powerfactor)


def train(train_graph: dataproc.Graph, val_graph: dataproc.Graph, powerfactors=(1, 10, 100, 1000), max_iter=100):
    errors = []
    for powerfactor in powerfactors:
        fg_model = single_train(train_graph, powerfactor, max_iter=max_iter)
        val_pred = fg_model.predict(val_graph)
        res = calc_flow_prediction_evaluation(val_pred, val_graph.flow)
        errors.append(res["MeAE"])

    best_pf = powerfactors[np.argmax(np.asarray(errors)).item()]
    edges = np.concatenate((train_graph.edges, val_graph.edges), axis=0)
    flow = np.concatenate((train_graph.flow, val_graph.flow), axis=0)
    joined_graph = dataproc.Graph(num_nodes=train_graph.num_nodes, edges=edges, flow=flow)

    fg_model = single_train(joined_graph, best_pf, max_iter=max_iter)
    return fg_model


def initiliaze_scores(G):
    fairness = {}
    goodness = {}

    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight') * 1.0 / G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness


def compute_fairness_goodness(G: nx.DiGraph, max_iter=100):
    fairness, goodness = initiliaze_scores(G)

    nodes = G.nodes()
    iter = 0
    while iter < max_iter:
        df = 0
        dg = 0

        print('-----------------')
        print("Iteration number", iter)

        print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]] * edge[2]

            try:
                dg += abs(g / len(inedges) - goodness[node])
                goodness[node] = g / len(inedges)
            except:
                pass

        print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]]) / 2.0
            try:
                df += abs(f / len(outedges) - fairness[node])
                fairness[node] = f / len(outedges)
            except:
                pass

        print('Differences in fairness score and goodness score = %.2f, %.2f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter += 1

    return fairness, goodness
