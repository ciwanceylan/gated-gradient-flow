import os
import warnings
import pandas as pd
import numpy as np
import torch

import src.dataprocessing as dataproc
import src.flow_models as flow_models
import src.evaluation as evaluation
import src.experimentutils as experutils
import src.fairness_goodness as fairness_goodness

HYPERPARAMS_COLUMNS = {
    "fairness_goodness": ["powerfactor"],
    "dnn2_engi_feat": ['feature_set', 'q_inx', 'reg_inx', 'reg'],
    "dnn2_node2vec": ['feature_set', 'q_inx', 'reg_inx', 'reg'],
    "dnn2_both": ['feature_set', 'q_inx', 'reg_inx', 'reg'],
    "grad": ['emb_reg_inx', 'gates_reg_inx', 'emb_reg', 'gate_reg'],
    "gated": ['dim', 'emb_reg_inx', 'gates_reg_inx', 'emb_reg', 'gate_reg'],
}

MODEL_FILE_TMPLS = {
    "dnn2": {
        "features": "features_{}_{:d}.pth",
        "model": "model_parameters_baseline_dnn2_{}_{:d}_{:d}.pth"
    },
    "grad": "model_parameters_grad_baseline_{:d}_{:d}.pth",
    "gated": "model_parameters_joint_{:d}_{:d}_{:d}.pth"
}


def get_all_best_model_paths(best_hps, basepath="results/ethereum/chpt"):
    filenames = dict()
    for model_name, best_hp in best_hps.items():
        filename_or_dict = get_model_path(best_hp, model_name)
        if isinstance(filename_or_dict, dict):
            filenames[model_name] = dict()
            for key, filename in filename_or_dict.items():
                filenames[model_name][key] = os.path.join(basepath, filename)
        elif filename_or_dict is None:
            continue
        else:
            filenames[model_name] = os.path.join(basepath, filename_or_dict)
    return filenames


def get_model_path(best_hp, model_name):
    if model_name == "gated":
        filename = MODEL_FILE_TMPLS[model_name].format(int(best_hp["dim"]), int(best_hp["emb_reg_inx"]),
                                                       int(best_hp["gates_reg_inx"]))
    elif model_name == "grad":
        filename = MODEL_FILE_TMPLS[model_name].format(int(best_hp["emb_reg_inx"]), int(best_hp["gates_reg_inx"]))
    elif model_name.startswith("dnn2"):
        filename = {
            "features": MODEL_FILE_TMPLS["dnn2"]["features"].format(best_hp['feature_set'], int(best_hp['q_inx'])),
            "model": MODEL_FILE_TMPLS["dnn2"]["model"].format(best_hp['feature_set'], int(best_hp['q_inx']),
                                                              int(best_hp['reg_inx']))
        }
    else:
        filename = None
    return filename


def get_all_best_hp(results, eval_col=("val_MeAE*", "val_MeAE"), ascending=True):
    top_indices = dict()
    best_hps = dict()
    for model_name, df in results.items():
        top_index, hyper_param = get_best_hp(df, HYPERPARAMS_COLUMNS[model_name],
                                             eval_col=eval_col, ascending=ascending)
        top_indices[model_name] = top_index
        best_hps[model_name] = hyper_param

    return best_hps, top_indices


def get_best_hp(df, hyper_parameter_columns, eval_col=("val_MeAE*", "val_MeAE"), ascending=True):
    hp_columns = set(hyper_parameter_columns).intersection(df.columns)

    top = get_top_k_results(df, cols=hp_columns, eval_col=eval_col, k=1, ascending=ascending)
    top_index = top.index[0]
    hyper_param = top.loc[top_index, :].to_dict()

    return top_index, hyper_param


def get_top_k_results(df, cols=None, eval_col=("val_MeAE*", "val_MeAE"), k=1, ascending=True) -> pd.DataFrame:
    if isinstance(eval_col, tuple):
        eval_col = list(eval_col)
    out = df.sort_values(by=eval_col, ascending=ascending).head(k)
    if cols:
        out = out.loc[:, cols]
    return out


def evaluate_all_models(model_order, model_paths, train_graph: dataproc.Graph,
                        test_graph: dataproc.Graph, common_configs, fg_powerfactor=None):
    init_config = common_configs['init_config']
    all_results = {}
    all_flow_predictions = {}
    all_representations = {}
    all_flow_predictions["gt_flow"] = test_graph.flow
    for model_name in model_order:
        if model_name in model_paths:
            test_res, pred, representations = evaluate_from_paths(model_paths[model_name], model_name,
                                                                  test_graph, init_config)
        elif model_name == "fairness_goodness" and fg_powerfactor is not None:
            test_res, pred, representations = eval_fairness_goodness(train_graph, test_graph,
                                                                     powerfactor=fg_powerfactor)
        elif model_name == "zeros":
            pred = np.zeros_like(test_graph.flow)
            test_res = evaluation.calc_flow_prediction_evaluation(pred, test_graph.flow, prefix="test")
            representations = None
        elif model_name == "init":
            test_res, pred, representations = eval_grad_init_model(train_graph=train_graph, test_graph=test_graph,
                                                                   init_config=init_config)
        else:
            warnings.warn(f"Missing test evaluation implementation for {model_name}")
            continue

        test_res["model"] = model_name
        all_results[model_name] = test_res
        all_flow_predictions[model_name] = pred
        all_representations[model_name] = representations

    return pd.DataFrame(all_results).T, all_flow_predictions, all_representations


def evaluate_from_paths(model_path, model_name, test_graph: dataproc.Graph, init_config):
    if model_name == "gated" or model_name == "grad":
        test_res, pred, representations = eval_gradient_model(model_path, test_graph=test_graph,
                                                              init_config=init_config)

    elif model_name.startswith("dnn2"):
        test_res, pred, representations = eval_dnn2_baseline_model(feature_path=model_path["features"],
                                                                   model_path=model_path["model"],
                                                                   test_graph=test_graph)
    else:
        raise ValueError(f"Model {model_name} cannot be loaded from path {model_path}")

    return test_res, pred, representations


def eval_gradient_model(path, test_graph: dataproc.Graph, init_config):
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    representations = {
        'emb': state_dict['node_embeddings.weight'].numpy(),
        'gates': state_dict['gates.weight'].numpy()
    }
    num_nodes, emb_dim = representations['emb'].shape

    model = flow_models.SheafFlowPlusPlus(num_nodes=num_nodes,
                                          embedding_dim=emb_dim,
                                          use_simple_gates=init_config.use_simple_gates,
                                          beta=init_config.beta)

    model.load_state_dict(state_dict)

    pred = evaluation.get_flow_prediction(test_graph=test_graph, model=model)
    test_res = evaluation.calc_flow_prediction_evaluation(pred, test_graph.flow, prefix="test")

    return test_res, pred, representations


def eval_dnn2_baseline_model(feature_path, model_path, test_graph: dataproc.Graph):
    features = np.loadtxt(feature_path)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model = flow_models.DNNRegressionBaseline(features, num_layers=2)
    model.load_state_dict(state_dict)

    pred = evaluation.get_flow_prediction(test_graph=test_graph, model=model)
    test_res = evaluation.calc_flow_prediction_evaluation(pred, test_graph.flow, prefix="test")
    representations = features

    return test_res, pred, representations


def eval_fairness_goodness(train_graph: dataproc.Graph, test_graph: dataproc.Graph, powerfactor):
    fg_model = fairness_goodness.single_train(train_graph=train_graph, powerfactor=powerfactor)
    pred = fg_model.predict(test_graph)
    test_res = evaluation.calc_flow_prediction_evaluation(pred, test_graph.flow, prefix="test")
    representations = None
    return test_res, pred, representations


def eval_grad_init_model(train_graph: dataproc.Graph, test_graph: dataproc.Graph, init_config):
    use_simple_gates = init_config.use_simple_gates
    num_nodes = train_graph.num_vertices()
    src_nodes = torch.from_numpy(train_graph.src_nodes)
    dst_nodes = torch.from_numpy(train_graph.dst_nodes)
    train_flow = torch.from_numpy(train_graph.flow)

    embeddings_init, normr = experutils.grad_embeddings_init(src_nodes, dst_nodes, train_flow,
                                                             num_nodes, embedding_dim=1)

    grad_init_model = flow_models.SheafFlowPlusPlus(num_nodes,
                                                    embedding_dim=1,
                                                    use_simple_gates=use_simple_gates,
                                                    embedding_init=2 * embeddings_init,
                                                    gates_init='zeros',
                                                    beta=1)

    pred = evaluation.get_flow_prediction(test_graph=test_graph, model=grad_init_model)
    test_res = evaluation.calc_flow_prediction_evaluation(pred, test_graph.flow, prefix="test")
    representations = {
        'emb': grad_init_model.node_embeddings.weight.detach().numpy(),
        'gates': grad_init_model.gates.weight.detach().numpy()
    }

    return test_res, pred, representations


def split_baselines(baseline_df):
    results = dict()
    unique_baselines = baseline_df.baseline_name.unique()
    for bl in unique_baselines:
        results[bl] = baseline_df.loc[baseline_df.baseline_name == bl, :]
    return results