from typing import Literal
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from src import dataprocessing as dataproc, experimentutils as experutils, flow_models
import src.utils


def run_eth_hp_search_nn_baselines(model_factory, emb_reg_weights, q_indices, fg_max_iter, args, configs):
    num_emb_reg = len(emb_reg_weights)
    node2vec_path = os.path.join(args.node2vec_path, "node2vec_emb_{}.pth")
    chp_folder = os.path.join(args.path, "chpt")

    train_config = configs['train_config']
    eval_config = configs['eval_config']
    init_config = configs['init_config']
    hyperpara_config = configs['hyperpara_config']
    baseline_hyperpara_config = configs['baseline_hyperpara_config']
    loss_config = configs['loss_config']

    print("======> Running baseline")
    all_baseline_results = []
    if fg_max_iter is not None:
        print("... Running fairness-goodness baseline")
        fg_results = experutils.fit_fg_hp_search(model_factory.base, powerfactors=(1, 10, 100, 1000),
                                                 max_iter=fg_max_iter)
        all_baseline_results += fg_results

    for feature_set in tqdm(["engi_feat", "node2vec", "both"], total=3):
        _q_indices = [0] if feature_set == 'engi_feat' else q_indices
        for q_inx in tqdm(_q_indices, total=len(_q_indices)):

            node2vec_pth = node2vec_path.format(q_inx)
            features = model_factory.make_features(feature_set=feature_set, node2vec_path=node2vec_pth)
            np.savetxt(os.path.join(chp_folder, f"features_{feature_set}_{q_inx}.pth"), features)
            for reg_inx, reg in tqdm(zip(range(num_emb_reg), emb_reg_weights), total=num_emb_reg):

                baseline_hyperpara_config.reg = src.utils.SheafFlowReg(nn.L1Loss(), reg)
                current_parameters = {'feature_set': feature_set, 'q_inx': q_inx, 'reg_inx': reg_inx, 'reg': reg}

                src.utils.save_configs(
                    os.path.join(args.path, f"baseline_configs_{feature_set}_{q_inx}_{reg_inx}.json"), **configs)

                for model_type in ('dnn2',):
                    eval_config.model_chp_path = \
                        os.path.join(chp_folder,
                                     f"model_parameters_baseline_{model_type}_{feature_set}_{q_inx}_{reg_inx}.pth")
                    model = model_factory.build_nn_model(features=features, model_type=model_type)
                    _, baseline_result, baseline_history = experutils.fit_baseline(
                        model_factory.base, model=model, model_type=model_type, feature_set=feature_set,
                        train_config=train_config, eval_config=eval_config,
                        baseline_hyperpara_config=baseline_hyperpara_config)
                    baseline_result.update(current_parameters)
                    all_baseline_results.append(baseline_result)
                pd.DataFrame(all_baseline_results).to_csv(os.path.join(args.path, "ethereum_baseline_results.csv"),
                                                          header=True, index=False)


def run_eth_hp_search_gradient_models(model_factory, emb_reg_weights, gates_reg_weights, args, configs):
    num_emb_reg = len(emb_reg_weights)
    num_gates_reg = len(gates_reg_weights)
    chp_folder = os.path.join(args.path, "chpt")

    train_config = configs['train_config']
    eval_config = configs['eval_config']
    init_config = configs['init_config']
    hyperpara_config = configs['hyperpara_config']
    baseline_hyperpara_config = configs['baseline_hyperpara_config']
    loss_config = configs['loss_config']

    all_joint_results = []
    all_init_results = []
    all_grad_baseline_results = []

    for dim in trange(args.min_dim, args.max_dim + 1):
        init_config.embedding_dim = dim

        for emb_reg_inx, emb_reg in tqdm(zip(range(num_emb_reg), emb_reg_weights), total=num_emb_reg):

            for gates_reg_inx, gate_reg in tqdm(zip(range(num_gates_reg), gates_reg_weights), total=num_gates_reg):
                hyperpara_config.embeddings_reg = src.utils.SheafFlowReg(nn.L1Loss(), emb_reg)
                hyperpara_config.gates_reg = src.utils.SheafFlowReg(nn.L1Loss(), gate_reg)
                current_parameters = {'dim': dim, 'emb_reg_inx': emb_reg_inx, 'emb_reg': emb_reg,
                                      'gates_reg_inx': gates_reg_inx, 'gate_reg': gate_reg}

                src.utils.save_configs(os.path.join(args.path, f"configs_{dim}_{emb_reg_inx}_{gates_reg_inx}.json"),
                                       **configs)

                model, init_results = model_factory.build_sheaf_flow_model(config=init_config, eval_model=True)
                for res in init_results:
                    res.update(current_parameters)

                eval_config.model_chp_path = os.path.join(chp_folder,
                                                          f"model_parameters_joint_{dim}_{emb_reg_inx}_{gates_reg_inx}.pth")

                _, joint_results, _ = experutils.fit_joint(base=model_factory.base, model=model,
                                                           train_config=train_config,
                                                           eval_config=eval_config, hyperpara_config=hyperpara_config)
                if dim == 1:
                    eval_config.model_chp_path = os.path.join(chp_folder,
                                                              f"model_parameters_grad_baseline_{emb_reg_inx}_{gates_reg_inx}.pth")
                    gradient_baseline_results = experutils.fit_gradient_baseline_mf(model_factory=model_factory,
                                                                                    train_config=train_config,
                                                                                    eval_config=eval_config,
                                                                                    hyperpara_config=hyperpara_config,
                                                                                    init_config=init_config)

                    gradient_baseline_results.update(current_parameters)
                    all_grad_baseline_results.append(gradient_baseline_results)

                joint_results.update(current_parameters)
                all_joint_results.append(joint_results)
                all_init_results += init_results

                pd.DataFrame(all_joint_results).to_csv(os.path.join(args.path, "ethereum_joint_results.csv"),
                                                       header=True,
                                                       index=False)
                pd.DataFrame(all_init_results).to_csv(os.path.join(args.path, "ethereum_init_results.csv"),
                                                      header=True,
                                                      index=False)
                pd.DataFrame(all_grad_baseline_results).to_csv(
                    os.path.join(args.path, "ethereum_grad_baseline_results.csv"),
                    header=True,
                    index=False)


def run_synth_ablations(mode: Literal["unimodal", "multimodal"], ablation_config_files, args):
    if args.graph == 'cora':
        graph = dataproc.Graph.read_csv(os.path.join(args.data_path, "preprocessed_cora.csv"))
        node2vec_path = os.path.join(args.data_path, "trained_node2vec/node2vec_cora/node2vec_emb_0.pth")
    elif args.graph == 'bitcoin':
        graph = dataproc.Graph.read_csv(os.path.join(args.data_path, "preprocessed_bitcoin.csv"))
        node2vec_path = os.path.join(args.data_path, "trained_node2vec/node2vec_bitcoin/node2vec_emb_0.pth")
    elif args.graph == 'complete':
        graph = dataproc.complete_graph(num_nodes=40)
        node2vec_path = os.path.join(args.data_path, "trained_node2vec/node2vec_complete/node2vec_emb_0.pth")
    else:
        raise ValueError(f"Unknown graph {args.graph}")

    feature_sets = ("engi_feat", "node2vec", "both")
    model_types = ("dnn2",)

    if mode == "unimodal":
        emb_scale = 1e2
        gates_scale = 1.
        run_synth_ablation_unimodal(graph, ablation_config_files, emb_scale, gates_scale,
                                    feature_sets=feature_sets, model_types=model_types, node2vec_path=node2vec_path,
                                    args=args)
    elif mode == "multimodal":
        scenarios = ["2:2"]
        run_synth_ablation_multimodal(graph, ablation_config_files, feature_sets=feature_sets, model_types=model_types,
                                      node2vec_path=node2vec_path, args=args, scenarios=scenarios)


def run_synth_ablation_unimodal(graph, ablation_config_files, emb_scale, gates_scale, feature_sets,
                                model_types, node2vec_path, args):
    num_nodes = graph.num_vertices()
    init_config = src.utils.load_configs(ablation_config_files[0])['init_config']  # type: src.utils.InitConfig
    dev = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    device = torch.device(dev)

    flow_summaries = []

    all_joint_results = []
    all_baseline_results = []
    all_init_results = []
    all_grad_baseline_results = []
    all_fg_results = []

    for i in trange(args.num_flows):
        gt_node_embeddings = src.utils.sample_student_t(num_nodes, init_config.embedding_dim, df=2, scale=emb_scale)
        gt_gates = src.utils.sample_student_t(num_nodes, init_config.embedding_dim, df=4,
                                              scale=gates_scale) - 2 * gates_scale

        gt_model = flow_models.SheafFlowPlusPlus(num_nodes=num_nodes, embedding_dim=init_config.embedding_dim,
                                                 use_simple_gates=init_config.use_simple_gates,
                                                 embedding_init=gt_node_embeddings, gates_init=gt_gates, beta=1.)

        graph, flow_summary = dataproc.add_sheaf_flow(graph, gt_model)
        flow_summary.update({'flow_id': i})
        flow_summaries.append(flow_summary)

        joint_results, baseline_results, init_results, grad_baseline_results, fg_results = experutils.run_ablation(
            graph=graph, device=device, gt_node_embeddings=gt_node_embeddings, gt_gates=gt_gates,
            config_files=ablation_config_files, flow_id=i,
            feature_sets=feature_sets, model_types=model_types, node2vec_path=node2vec_path,
            powerfactors=(1, 10, 100, 1000), verbosity=args.verbosity, config_save_dir=args.path)

        all_joint_results += joint_results
        all_baseline_results += baseline_results
        all_init_results += init_results
        all_grad_baseline_results += grad_baseline_results
        all_fg_results += fg_results

        pd.DataFrame(all_joint_results).to_csv(os.path.join(args.path, f"{args.graph}_ablation_joint_results.csv"),
                                               header=True,
                                               index=False)
        pd.DataFrame(all_baseline_results).to_csv(
            os.path.join(args.path, f"{args.graph}_ablation_baseline_results.csv"),
            header=True,
            index=False)
        pd.DataFrame(all_init_results).to_csv(os.path.join(args.path, f"{args.graph}_ablation_init_results.csv"),
                                              header=True,
                                              index=False)
        pd.DataFrame(all_grad_baseline_results).to_csv(
            os.path.join(args.path, f"{args.graph}_ablation_grad_baseline_results.csv"),
            header=True,
            index=False)
        pd.DataFrame(all_fg_results).to_csv(
            os.path.join(args.path, f"{args.graph}_ablation_fairness_goodness.csv"),
            header=True,
            index=False)
        pd.DataFrame(flow_summaries).to_csv(os.path.join(args.path, "flow_info.csv"), header=True, index=False)


def run_synth_ablation_multimodal(graph, ablation_config_files, feature_sets, model_types, node2vec_path,
                                  args, scenarios):
    dev = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    device = torch.device(dev)

    init_config = src.utils.load_configs(ablation_config_files[0])['init_config']  # type: src.utils.InitConfig

    num_nodes = graph.num_vertices()

    flow_summaries = []
    # scenarios = ['2:2']

    all_joint_results = []
    all_baseline_results = []
    all_init_results = []
    all_grad_baseline_results = []
    all_fg_results = []

    for scenario in tqdm(scenarios):
        for i in trange(args.num_flows):
            gt_node_embeddings, gt_gates, emb_comp_ids, gates_comp_ids = dataproc.sample_trade_scenario(scenario,
                                                                                                        num_nodes=num_nodes)
            gt_node_embeddings = torch.from_numpy(gt_node_embeddings.astype(np.float32))
            gt_gates = torch.from_numpy(gt_gates.astype(np.float32))

            gt_model = flow_models.SheafFlowPlusPlus(num_nodes=num_nodes, embedding_dim=init_config.embedding_dim,
                                                     use_simple_gates=init_config.use_simple_gates,
                                                     embedding_init=gt_node_embeddings, gates_init=gt_gates,
                                                     beta=1.)

            graph, flow_summary = dataproc.add_sheaf_flow(graph, gt_model)
            flow_summary.update({'flow_id': i, 'scenario': scenario})
            flow_summaries.append(flow_summary)

            joint_results, baseline_results, init_results, grad_baseline_results, fg_results = experutils.run_ablation(
                graph=graph, device=device, gt_node_embeddings=gt_node_embeddings, gt_gates=gt_gates,
                config_files=ablation_config_files, flow_id=i,
                feature_sets=feature_sets, model_types=model_types, node2vec_path=node2vec_path,
                verbosity=args.verbosity,
                config_save_dir=args.path)
            for results in (joint_results, baseline_results, init_results, grad_baseline_results):
                for res in results:
                    res.update({'scenario': scenario})

            all_joint_results += joint_results
            all_baseline_results += baseline_results
            all_init_results += init_results
            all_grad_baseline_results += grad_baseline_results
            all_fg_results += fg_results

            pd.DataFrame(all_joint_results).to_csv(os.path.join(args.path, f"{args.graph}_ablation_joint_results.csv"),
                                                   header=True,
                                                   index=False)
            pd.DataFrame(all_baseline_results).to_csv(
                os.path.join(args.path, f"{args.graph}_ablation_baseline_results.csv"),
                header=True,
                index=False)
            pd.DataFrame(all_init_results).to_csv(os.path.join(args.path, f"{args.graph}_ablation_init_results.csv"),
                                                  header=True,
                                                  index=False)
            pd.DataFrame(all_grad_baseline_results).to_csv(
                os.path.join(args.path, f"{args.graph}_ablation_grad_baseline_results.csv"),
                header=True,
                index=False)
            pd.DataFrame(all_fg_results).to_csv(
                os.path.join(args.path, f"{args.graph}_ablation_fairness_goodness.csv"),
                header=True,
                index=False)
            pd.DataFrame(flow_summaries).to_csv(os.path.join(args.path, "flow_info.csv"), header=True, index=False)
