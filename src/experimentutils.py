from __future__ import annotations
from typing import *
import datetime
import random
import time
import os
import copy

import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse as sp

import src.utils
from src import training as training, dataprocessing as dataproc, flow_models, fairness_goodness
from src import evaluation
from src.utils import TrainingConfig, HyperParamConfig, InitConfig, EvalConfig, LossConfig, BaselineHyperParamConfig, \
    GradientNoise, SheafFlowReg


class ModelFactory:
    named_aggs = {
        "avg": pd.NamedAgg(column="flow", aggfunc="mean"),
        "sum": pd.NamedAgg(column="flow", aggfunc="sum"),
        "degree": pd.NamedAgg(column="flow", aggfunc="count"),
        "avg_abs": pd.NamedAgg(column="flow_abs", aggfunc="mean"),
        "std": pd.NamedAgg(column="flow", aggfunc="std"),
        "avg_sign": pd.NamedAgg(column="flow_sign", aggfunc="mean")
    }

    def __init__(self, base: training.TrainerBase):

        self.base = base

    @classmethod
    def create(cls, graph: dataproc.Graph, device, loss_config: LossConfig = LossConfig()):
        train_graph, val_graph, _ = graph.split_train_val_test_graphs(desired_split=(0.8, 0.2, 0.))

        base = training.TrainerBase(train_graph=train_graph, val_graph=val_graph,
                                    device=device, loss_config=loss_config)
        return cls(base=base)

    def build_sheaf_flow_model(self, config: InitConfig, eval_model=False):
        model = flow_models.SheafFlowPlusPlus(self.base.num_nodes, config.embedding_dim,
                                              use_simple_gates=config.use_simple_gates,
                                              embedding_init=config.embedding_init,
                                              gates_init=config.gates_init,
                                              beta=config.beta)
        results = []
        if eval_model:
            model.to(self.base.device)
            results_ = eval_model_basic(self.base, model)
            results_['init_id'] = 0
            results.append(results_)
            model.to(torch.device('cpu'))

        if config.embedding_init == 'auto':
            model = self.initialize_embeddings(model)
        elif config.embedding_init == 'gradient':
            model = self.initialize_embeddings(model)
        elif config.embedding_init == 'auto-noise':
            model = self.initialize_embeddings(model, config.auto_noise_std)

        model.to(self.base.device)
        if eval_model:
            results_ = eval_model_basic(self.base, model)
            results_['init_id'] = 1
            results.append(results_)

        if config.gates_init == 'activation':
            model = self.initialize_gates_via_activation_fit(model, config=config)
        elif config.gates_init == 'auto':
            model = self.initialize_gates_via_activation_fit(model, config=config)

        if eval_model:
            results_ = eval_model_basic(self.base, model)
            results_['init_id'] = 2
            results.append(results_)
            return model, results

        return model

    def initialize_embeddings(self, model, noise_std=0.):
        embeddings_init, normr = self.get_embeddings_gradient_init_np(model.gates.weight.shape[1])

        embeddings_init = torch.from_numpy(embeddings_init)
        if torch.sum(torch.abs(model.gates.weight.detach())) < 1e-4:  # Correct for zero initalization of gates
            embeddings_init = 2 * embeddings_init
        model.node_embeddings.weight.data = embeddings_init
        if noise_std > 0.:
            model.node_embeddings.weight.data += noise_std * torch.sqrt(
                torch.abs(model.node_embeddings.weight.data)) * torch.randn_like(model.node_embeddings.weight.data)
        return model

    def get_embeddings_gradient_init_np(self, embedding_dim):
        embeddings_init, normr = grad_embeddings_init(self.base.train_graph.src_nodes,
                                                      self.base.train_graph.dst_nodes,
                                                      self.base.train_graph.flow,
                                                      self.base.num_nodes, embedding_dim)
        return embeddings_init, normr

    def initialize_gates_via_activation_fit(self, model: flow_models.SheafFlowPlusPlus,
                                            config: InitConfig):
        train_config = TrainingConfig()
        train_config.max_steps = config.max_steps
        train_config.tol = config.tol
        model0, _, _ = fit_gates_init(base=self.base, model=model, train_config=train_config, init_config=config)
        return model0

    def build_baseline_models(self, feature_set: Literal["engi_feat", "node2vec", "both"],
                              model_type: Literal["linear", "dnn1", "dnn2"],
                              node2vec_path,
                              normalize_features=True) -> nn.Module:
        features = self.make_features(feature_set=feature_set, node2vec_path=node2vec_path,
                                      normalize_features=normalize_features)

        return self.build_nn_model(features, model_type)

    def build_nn_model(self, features, model_type):
        if model_type == 'linear':
            model = flow_models.LinearRegressionBaseline(features)
        elif model_type == 'dnn1':
            model = flow_models.DNNRegressionBaseline(features, num_layers=1)
        elif model_type == 'dnn2':
            model = flow_models.DNNRegressionBaseline(features, num_layers=2)
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        model.to(self.base.device)
        return model

    def make_features(self, feature_set: Literal["engi_feat", "node2vec", "both"],
                      node2vec_path, normalize_features=True):
        if feature_set == 'engi_feat':
            features = self.build_flow_features(normalize_features=normalize_features)
        elif feature_set == 'node2vec':
            features = self.load_node2vec_embeddings(node2vec_path)
        elif feature_set == 'both':
            features = np.concatenate((
                self.build_flow_features(normalize_features=normalize_features),
                self.load_node2vec_embeddings(node2vec_path)
            ), axis=1)
        else:
            raise ValueError(f"Unknown feature_set {feature_set}")
        return features

    def build_flow_features(self, normalize_features=True):

        data = pd.DataFrame({
            "nodes": np.concatenate((self.base.train_graph.edges[:, 0], self.base.train_graph.edges[:, 1]), axis=0),
            "flow": np.concatenate((-self.base.train_graph.flow, self.base.train_graph.flow), axis=0),
        })
        data["flow_sign"] = np.sign(data["flow"])
        data["flow_abs"] = np.abs(data["flow"])
        features = data.groupby("nodes").agg(**self.named_aggs)
        features.fillna(0, inplace=True)
        features = features.sort_index(ascending=True).to_numpy()

        if normalize_features:
            features_mean = np.mean(features, axis=0)
            features_std = np.std(features, axis=0)
            features = (features - features_mean) / features_std
        return features

    @staticmethod
    def load_node2vec_embeddings(path: str):
        embeddings = torch.load(path, map_location=torch.device('cpu'))['embedding.weight'].numpy()
        return embeddings


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_model_basic(base: training.TrainerBase, model):
    results = training.Trainer(base=base, model=model, optimizer=None).eval_model()
    return results


def fit_baseline(base: training.TrainerBase,
                 model: nn.Module,
                 model_type: Literal["linear", "dnn1", "dnn2"],
                 feature_set: Literal["engi_feat", "node2vec", "both"],
                 train_config: TrainingConfig = TrainingConfig(),
                 eval_config: EvalConfig = EvalConfig(),
                 baseline_hyperpara_config: BaselineHyperParamConfig = BaselineHyperParamConfig(),
                 verbosity=0, return_history=False):
    optimizer = optim.AdamW(model.reg_model.parameters(), lr=train_config.lr, amsgrad=True)
    trainer = training.BaselineTrainer(base=base, model=model, optimizer=optimizer, train_config=train_config,
                                       eval_config=eval_config, baseline_hyperpara_config=baseline_hyperpara_config)
    start_time = time.time()
    results, *history = trainer.train(tol=train_config.tol, max_steps=train_config.max_steps, verbosity=verbosity,
                                      return_history=return_history)
    train_time = time.time() - start_time
    results.update({"train_time": train_time,
                    "baseline_name": model_type + "_" + feature_set,
                    'model_type': model_type,
                    'feature_set': feature_set})
    return model, results, history


def fit_gates_init(base: training.TrainerBase, model: flow_models.SheafFlowPlusPlus,
                   gt_embeddings=None, gt_gates=None,
                   train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                   hyperpara_config: HyperParamConfig = HyperParamConfig(), init_config: InitConfig = InitConfig(),
                   verbosity=0, return_history=False):
    gates_optimizer = optim.AdamW(model.gates.parameters(), lr=0.01, amsgrad=True)
    trainer = training.GatesInitTrainer(base=base, model=model, optimizer=gates_optimizer,
                                        gt_embeddings=gt_embeddings, gt_gates=gt_gates,
                                        train_config=train_config, eval_config=eval_config,
                                        hyperpara_config=hyperpara_config, init_config=init_config)
    start_time = time.time()
    results, *history = trainer.train(tol=train_config.tol, max_steps=train_config.max_steps, verbosity=verbosity,
                                      return_history=return_history)
    train_time = time.time() - start_time
    results.update({"train_time": train_time})
    return model, results, history


def fit_joint(base: training.TrainerBase, model: flow_models.SheafFlowPlusPlus,
              gt_embeddings=None, gt_gates=None,
              train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
              hyperpara_config: HyperParamConfig = HyperParamConfig(),
              verbosity=0, return_history=False
              ):
    if train_config.use_lbfgs:
        optimizer = optim.LBFGS([
            {'params': model.node_embeddings.parameters()},
            {'params': model.gates.parameters(), 'lr': train_config.gates_lr}], lr=train_config.lr,
            max_iter=train_config.max_lbfgs_step)
    else:
        optimizer = optim.AdamW([
            {'params': model.node_embeddings.parameters()},
            {'params': model.gates.parameters(), 'lr': train_config.gates_lr}], lr=train_config.lr, amsgrad=True)

    trainer = training.SheafTrainer(base=base, model=model, optimizer=optimizer, gt_embeddings=gt_embeddings,
                                    gt_gates=gt_gates, train_config=train_config, eval_config=eval_config,
                                    hyperpara_config=hyperpara_config)

    start_time = time.time()
    results, *history = trainer.train(tol=train_config.tol, max_steps=train_config.max_steps, verbosity=verbosity,
                                      return_history=return_history)
    train_time = time.time() - start_time
    results.update({"train_time": train_time})
    return model, results, history


def fit_gradient_baseline_mf(model_factory: ModelFactory, gt_embeddings=None, gt_gates=None,
                             train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                             hyperpara_config: HyperParamConfig = HyperParamConfig(),
                             init_config: InitConfig = InitConfig()):
    init_config = copy.deepcopy(init_config)
    init_config.gates_init = 'auto'
    init_config.gates_init = 'zeros'
    model = model_factory.build_sheaf_flow_model(config=init_config)
    _, grad_baseline_results, _ = fit_gradient_baseline(base=model_factory.base, model=model,
                                                        gt_embeddings=gt_embeddings,
                                                        gt_gates=gt_gates, train_config=train_config,
                                                        eval_config=eval_config,
                                                        hyperpara_config=hyperpara_config)
    return grad_baseline_results


def fit_gradient_baseline(base: training.TrainerBase, model: flow_models.SheafFlowPlusPlus,
                          gt_embeddings=None, gt_gates=None,
                          train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                          hyperpara_config: HyperParamConfig = HyperParamConfig(),
                          verbosity=0, return_history=False
                          ):
    model.gates.weight.data = torch.zeros_like(model.gates.weight.data)
    optimizer = optim.AdamW(model.node_embeddings.parameters(), lr=train_config.lr, amsgrad=True)

    # Ensure that gates are not affected by adding noise or reguralisation
    hyperpara_config = copy.deepcopy(hyperpara_config)
    hyperpara_config.gates_grad_noise = GradientNoise(False, np.inf, 0.)
    hyperpara_config.gates_reg = SheafFlowReg(nn.L1Loss(), 0.0)

    trainer = training.SheafTrainer(base=base, model=model, optimizer=optimizer, gt_embeddings=gt_embeddings,
                                    gt_gates=gt_gates, train_config=train_config, eval_config=eval_config,
                                    hyperpara_config=hyperpara_config)

    start_time = time.time()
    results, *history = trainer.train(tol=train_config.tol, max_steps=train_config.max_steps, verbosity=verbosity,
                                      return_history=return_history)
    train_time = time.time() - start_time
    results.update({"train_time": train_time})
    return model, results, history


def fit_init_joint(model_factory: ModelFactory, gt_node_embeddings, gt_gates,
                   train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                   hyperpara_config: HyperParamConfig = HyperParamConfig(), init_config: InitConfig = InitConfig()
                   ):
    model = model_factory.build_sheaf_flow_model(config=init_config)

    _, joint_results, _ = fit_joint(base=model_factory.base, model=model,
                                    gt_embeddings=gt_node_embeddings,
                                    gt_gates=gt_gates, train_config=train_config, eval_config=eval_config,
                                    hyperpara_config=hyperpara_config)

    return joint_results


def fit_feature_baselines(model_factory: ModelFactory, ablation_idx, gt_node_embeddings, gt_gates,
                          features: np.ndarray, feature_set, model_type,
                          train_config: TrainingConfig, eval_config: EvalConfig,
                          hyperpara_config: HyperParamConfig, init_config: InitConfig, loss_config=LossConfig(),
                          baseline_hyperpara_config=BaselineHyperParamConfig()):
    model = model_factory.build_nn_model(features, model_type=model_type)
    features_eval = evaluation.baseline_features_eval(model.node_features, gt_node_embeddings,
                                                      eval_config.gt_num_emb_modes)
    features_eval_gates = evaluation.baseline_features_eval(model.node_features, gt_gates,
                                                            eval_config.gt_num_gate_modes)
    features_eval_gates = {key + "_gates": val for key, val in features_eval_gates.items()}

    _, results, _ = fit_baseline(model_factory.base, model=model,
                                 model_type=model_type,
                                 feature_set=feature_set,
                                 train_config=train_config, eval_config=eval_config,
                                 baseline_hyperpara_config=baseline_hyperpara_config)
    if ablation_idx is not None:
        results.update({"ablation_idx": ablation_idx})
        results.update(features_eval)
        results.update(features_eval_gates)
    return results


def fit_fg_hp_search(base: training.TrainerBase, powerfactors=(1, 10, 100, 1000), max_iter=100):
    results = []
    for i, powerfactor in enumerate(powerfactors):
        start_time = time.time()
        fg_model = fairness_goodness.single_train(base.train_graph, powerfactor, max_iter=max_iter)
        train_time = time.time() - start_time
        train_pred = fg_model.predict(base.train_graph)
        train_res = evaluation.calc_flow_prediction_evaluation(train_pred, base.train_graph.flow, prefix="train")
        val_pred = fg_model.predict(base.val_graph)
        val_res = evaluation.calc_flow_prediction_evaluation(val_pred, base.val_graph.flow, prefix="val")
        result = train_res
        result.update(val_res)
        result["train_time"] = train_time
        result["baseline_name"] = "fairness_goodness"
        result["powerfactor"] = powerfactor
        result["ablation_idx"] = i
        results.append(result)
    return results


# def fit_baselines(model_factory: ModelFactory, ablation_idx, gt_node_embeddings, gt_gates,
#                   feature_set
#                   train_config: TrainingConfig, eval_config: EvalConfig, hyperpara_config: HyperParamConfig,
#                   init_config: InitConfig, loss_config=LossConfig(),
#                   baseline_hyperpara_config=BaselineHyperParamConfig(), node2vec_path=None):
#     feature_set = baseline_hyperpara_config.feature_set
#     if node2vec_path is None and (feature_set == "both" or feature_set == "engi_feat"):
#         raise ValueError("node2vec path is None")
#     baseline_model, dnn_baseline_model1, dnn_baseline_model2 = \
#         model_factory.build_baseline_models(feature_set=baseline_hyperpara_config.feature_set,
#                                             node2vec_path=node2vec_path)
#     features_eval = evaluation.baseline_features_eval(baseline_model.node_features, gt_node_embeddings,
#                                                       eval_config.gt_num_emb_modes)
#     features_eval_gates = evaluation.baseline_features_eval(baseline_model.node_features, gt_gates,
#                                                             eval_config.gt_num_gate_modes)
#     features_eval_gates = {key + "_gates": val for key, val in features_eval_gates.items()}
#     _, baseline_result, _ = fit_baseline(model_factory.base, model=baseline_model, baseline_name='linear',
#                                          train_config=train_config, eval_config=eval_config,
#                                          baseline_hyperpara_config=baseline_hyperpara_config)
#     _, dnn_baseline1_result, _ = fit_baseline(model_factory.base, model=dnn_baseline_model1, baseline_name='dnn1',
#                                               train_config=train_config, eval_config=eval_config,
#                                               baseline_hyperpara_config=baseline_hyperpara_config)
#     _, dnn_baseline2_result, _ = fit_baseline(model_factory.base, model=dnn_baseline_model2, baseline_name='dnn2',
#                                               train_config=train_config, eval_config=eval_config,
#                                               baseline_hyperpara_config=baseline_hyperpara_config)
#     if ablation_idx is not None:
#         baseline_result.update({"ablation_idx": ablation_idx})
#         baseline_result.update(features_eval)
#         baseline_result.update(features_eval_gates)
#         dnn_baseline1_result.update({"ablation_idx": ablation_idx})
#         dnn_baseline2_result.update({"ablation_idx": ablation_idx})
#     return baseline_result, dnn_baseline1_result, dnn_baseline2_result


def run_ablation(graph: dataproc.Graph,
                 device, gt_node_embeddings, gt_gates, config_files, flow_id,
                 feature_sets, node2vec_path=None,
                 model_types=("dnn2",), powerfactors=(1, 10, 100, 1000),
                 verbosity=1, config_save_dir=None):
    verboseprint = print if verbosity > 0 else lambda *a, **k: None
    configs = src.utils.load_configs(config_files[0])

    model_factory = ModelFactory.create(graph, device, loss_config=configs['loss_config'])
    auto_init_config = copy.deepcopy(configs['init_config'])  # type: InitConfig
    auto_init_config.embedding_init = 'auto'
    auto_init_config.gates_init = 'auto'
    _, init_results = model_factory.build_sheaf_flow_model(auto_init_config, eval_model=True)

    baseline_results = []
    joint_results = []
    grad_baseline_results = []

    fg_results = fit_fg_hp_search(base=model_factory.base, powerfactors=powerfactors)

    verboseprint(datetime.datetime.now(), "Training baseline")

    features = {}
    for feature_set in feature_sets:
        features[feature_set] = model_factory.make_features(feature_set=feature_set, node2vec_path=node2vec_path)

    for feature_set in feature_sets:
        for model_type in model_types:
            baseline_result = fit_feature_baselines(model_factory, ablation_idx=0,
                                                    gt_node_embeddings=gt_node_embeddings, gt_gates=gt_gates,
                                                    feature_set=feature_set, model_type=model_type,
                                                    features=features[feature_set], **configs)
            baseline_results.append(baseline_result)

    verboseprint(datetime.datetime.now(), "Training base model")
    joint_result, grad_baseline_result = _ablation_step(model_factory, gt_node_embeddings, gt_gates,
                                                        ablation_idx=0, **configs)
    joint_results.append(joint_result)
    grad_baseline_results.append(grad_baseline_result)

    for ablation_idx in trange(1, len(config_files)):
        new_configs = src.utils.load_configs(config_files[ablation_idx])
        configs.update(new_configs)  # Update to new hyperparameters
        if config_save_dir:
            src.utils.save_configs(os.path.join(config_save_dir, f"ablation_config_{ablation_idx}.json"), **configs)

        joint_result, grad_baseline_result = _ablation_step(model_factory, gt_node_embeddings, gt_gates,
                                                            ablation_idx=ablation_idx, **configs)
        joint_results.append(joint_result)
        grad_baseline_results.append(grad_baseline_result)
        if 'baseline_hyperpara_config' in new_configs:
            for feature_set in feature_sets:
                for model_type in model_types:
                    baseline_result = fit_feature_baselines(model_factory, ablation_idx=ablation_idx,
                                                            gt_node_embeddings=gt_node_embeddings, gt_gates=gt_gates,
                                                            feature_set=feature_set, model_type=model_type,
                                                            features=features[feature_set], **configs)
                    baseline_results.append(baseline_result)

    for results in [joint_results, grad_baseline_results, baseline_results, fg_results]:
        for sub_res in results:
            sub_res.update({'flow_id': flow_id})

    return joint_results, baseline_results, init_results, grad_baseline_results, fg_results


def _ablation_step(model_factory: ModelFactory, gt_node_embeddings, gt_gates, ablation_idx: int,
                   train_config: TrainingConfig, eval_config: EvalConfig, hyperpara_config: HyperParamConfig,
                   init_config: InitConfig, loss_config=LossConfig(),
                   baseline_hyperpara_config=BaselineHyperParamConfig()):
    joint_results = fit_init_joint(model_factory, gt_node_embeddings, gt_gates,
                                   train_config=train_config, eval_config=eval_config,
                                   hyperpara_config=hyperpara_config, init_config=init_config)
    grad_baseline_results = fit_gradient_baseline_mf(model_factory, gt_node_embeddings, gt_gates,
                                                     train_config=train_config, eval_config=eval_config,
                                                     hyperpara_config=hyperpara_config, init_config=init_config)

    joint_results.update({"ablation_idx": ablation_idx})
    grad_baseline_results.update({"ablation_idx": ablation_idx})
    return joint_results, grad_baseline_results


def fit_synthetic_models(model_factory: ModelFactory, feature_sets, model_types, node2vec_path,
                         powerfactor, train_config: TrainingConfig, eval_config: EvalConfig,
                         hyperpara_config: HyperParamConfig,
                         init_config: InitConfig,
                         baseline_hyperpara_config=BaselineHyperParamConfig()):
    all_res = {}
    all_models = {}

    for feature_set in feature_sets:
        for model_type in model_types:
            features = model_factory.make_features(feature_set=feature_set, node2vec_path=node2vec_path)
            model = model_factory.build_nn_model(features, model_type=model_type)
            model, results, _ = fit_baseline(model_factory.base, model=model,
                                             model_type=model_type,
                                             feature_set=feature_set,
                                             train_config=train_config, eval_config=eval_config,
                                             baseline_hyperpara_config=baseline_hyperpara_config)
            baseline_name = model_type + "_" + feature_set
            all_models[baseline_name] = model
            all_res[baseline_name] = results

    all_models["fairness_goodness"] = fairness_goodness.single_train(model_factory.base.train_graph,
                                                                     powerfactor=powerfactor,
                                                                     max_iter=100)

    model = model_factory.build_sheaf_flow_model(config=init_config)
    gated_init_model = copy.deepcopy(model)

    joint_model, joint_results, _ = fit_joint(base=model_factory.base, model=model, gt_embeddings=None,
                                              gt_gates=None, train_config=train_config, eval_config=eval_config,
                                              hyperpara_config=hyperpara_config)
    all_res['gated'] = joint_results
    all_models['gated'] = joint_model
    all_models['gated_init'] = gated_init_model

    init_config_grad = copy.deepcopy(init_config)
    init_config_grad.gates_init = 'auto'
    init_config_grad.gates_init = 'zeros'
    model = model_factory.build_sheaf_flow_model(config=init_config_grad)
    grad_init_model = copy.deepcopy(model)
    grad_model, grad_baseline_results, _ = fit_gradient_baseline(base=model_factory.base, model=model,
                                                                 gt_embeddings=None,
                                                                 gt_gates=None, train_config=train_config,
                                                                 eval_config=eval_config,
                                                                 hyperpara_config=hyperpara_config)
    all_res['grad'] = grad_baseline_results
    all_models['grad'] = grad_model
    all_models['grad_init'] = grad_init_model

    return all_models, all_res


def _choose_better(options: Tuple[Any, ...], scores: Iterable[float, ...]):
    scores = np.asarray(scores)
    best_idx = int(np.argmax(scores))
    return options[best_idx]


def add_order_index(list_of_dict: List[Dict]):
    for inx, d in enumerate(list_of_dict):
        d.update({'experiment_order_inx': inx})


def grad_embeddings_init(train_source_nodes, train_target_nodes, train_flow, num_nodes, embedding_dim):
    num_edges = len(train_flow)
    leaving = train_source_nodes
    arriving = train_target_nodes
    columns = np.concatenate((arriving, leaving), axis=0)
    values = np.concatenate((np.ones(num_edges), -np.ones(num_edges)), axis=0)

    rows = np.concatenate((np.arange(num_edges), np.arange(num_edges)))
    incidence_matrix = sp.coo_matrix((values, (rows, columns)), shape=(num_edges, num_nodes)).tocsr()
    potentials, istop, itn, normr = sp.linalg.lsqr(incidence_matrix, train_flow)[:4]
    embeddings_init = potentials.astype(np.float32) / embedding_dim
    embeddings_init = np.repeat(embeddings_init[..., np.newaxis], embedding_dim, 1)
    return embeddings_init, normr
