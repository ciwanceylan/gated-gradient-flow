import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

import src.dataprocessing as dataproc
import src.training as train_n2f
import src.experimentutils as experutils
import src.runutils as runutils
import src.utils

parser = argparse.ArgumentParser(description="Run the training on ethereum graph")
parser.add_argument("path", type=str, help="path for saving submission_results")
parser.add_argument("--data_path", type=str, help="the number of dims to search",
                    default="data/preprocessed_ethereum_2018_2020.csv")
parser.add_argument("--split_path", type=str, help="Path to train, val, test split folder",
                    default=None)
parser.add_argument("--node2vec_path", type=str, help="Path to pretrained node2vec embeddings",
                    default=None)
parser.add_argument("--config_path", type=str, help="path were the configs can be read",
                    default="configs_eth/eth_config_auto_noise.json")
parser.add_argument("--max_dim", type=int, help="the number of dims to search", default=3)
parser.add_argument("--min_dim", type=int, help="the number of dims to search", default=1)
parser.add_argument("--seed", type=int, help="random seed", default=1234)
parser.add_argument("--use_gpu", action='store_true', help="Use gpu")
parser.add_argument("--gates_init", type=str, help="Which gates init to use", default='zeros')
parser.add_argument("--only_baselines", action='store_true', help="Only run  the baselines")
parser.add_argument("--skip_baselines", action='store_true', help="Only run  the baselines")
parser.add_argument("--verbosity", type=int, help="verbosity level", default=1)
parser.add_argument("--debug", action='store_true', help="Run in debug mode, only one epoch per model")

args = parser.parse_args()
chp_folder = os.path.join(args.path, "chpt")
os.makedirs(args.path, exist_ok=True)
os.makedirs(chp_folder, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
device = torch.device(dev)
if device.type == 'cuda' and args.verbosity > 0:
    print(f"Using {torch.cuda.get_device_name(0)}")
experutils.set_seeds(args.seed)

configs = src.utils.load_configs(args.config_path)

configs['hyperpara_config'].emb_grad_noise = src.utils.GradientNoise(False, np.inf, 0.)
configs['hyperpara_config'].gates_grad_noise = src.utils.GradientNoise(False, np.inf, 0.)
configs['baseline_hyperpara_config'].grad_noise = src.utils.GradientNoise(False, np.inf, 0.)

configs['init_config'].gates_init = args.gates_init

configs['train_config'].max_steps = 1e5
configs['train_config'].tol = 1e-5
configs['train_config'].substep_tol = 1e-5
configs['init_config'].max_steps = 1e4
fg_max_iter = 100

if args.debug:
    warnings.warn("RUNNING WITH DEBUG CONFIGURATIONS. MODELS WONT BE TRAINED.")
    configs['train_config'].max_steps = 0
    configs['init_config'].embedding_init = 'zeros'
    configs['init_config'].gates_init = 'zeros'
    configs['init_config'].max_steps = 0
    fg_max_iter = 1
    # fg_max_iter = None

emb_reg_weights = [3., 1, 0.3, 0.1]
gates_reg_weights = [3., 1., 0.3, 0.1]
q_indices = [0, 1, 2]

graph = dataproc.Graph.read_csv(args.data_path)
num_nodes = graph.num_vertices()

flow_summary = dataproc.flow_summary(graph.flow)
pd.DataFrame([flow_summary]).to_csv(os.path.join(args.path, f"ethereum_flow_info.csv"), header=True, index=False)

if args.split_path is None:
    train_graph, val_graph, test_graph = graph.split_train_val_test_graphs((0.7, 0.15, 0.15))
    train_graph.to_csv(os.path.join(args.path, f"preprocessed_ethereum_train.csv"))
    val_graph.to_csv(os.path.join(args.path, f"preprocessed_ethereum_val.csv"))
    test_graph.to_csv(os.path.join(args.path, f"preprocessed_ethereum_test.csv"))
else:
    train_graph = dataproc.Graph.read_csv(os.path.join(args.split_path, f"preprocessed_ethereum_train.csv"))
    val_graph = dataproc.Graph.read_csv(os.path.join(args.split_path, f"preprocessed_ethereum_val.csv"))
    test_graph = dataproc.Graph.read_csv(os.path.join(args.split_path, f"preprocessed_ethereum_test.csv"))

base = train_n2f.TrainerBase(train_graph=train_graph, val_graph=val_graph, device=device,
                             loss_config=configs['loss_config'])
model_factory = experutils.ModelFactory(base=base)

if not args.skip_baselines:
    runutils.run_eth_hp_search_nn_baselines(model_factory, emb_reg_weights, q_indices, fg_max_iter, args, configs)
else:
    print("====> Skipping baselines")

if args.only_baselines:
    sys.exit("Finished baselines with option 'only_baselines', exiting.")

runutils.run_eth_hp_search_gradient_models(model_factory, emb_reg_weights, gates_reg_weights, args, configs)
