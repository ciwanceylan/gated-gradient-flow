import argparse
import os

import torch
import src.experimentutils as experutils
import src.runutils as runutils

parser = argparse.ArgumentParser(description="Run the ablation study on a graph using synthetic flow")
parser.add_argument("path", type=str, help="path for saving submission_results")
parser.add_argument("--data_path", type=str, help="Path to data parent folder", default="data/")
parser.add_argument("--dist", type=str, help="unimodal or multimodal")
parser.add_argument("--configs_path", type=str, help="path were the configs can be read")
parser.add_argument("--graph", type=str, help="which graph", default='cora')
parser.add_argument("--num_flows", type=int, help="number of random flows per sampled embedding set", default=10)
parser.add_argument("--seed", type=int, help="random seed", default=1234)
parser.add_argument("--use_gpu", action='store_true', help="Use gpu")
parser.add_argument("--verbosity", type=int, help="verbosity level", default=1)

args = parser.parse_args()

os.makedirs(args.path, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
device = torch.device(dev)
if device.type == 'cuda' and args.verbosity > 0:
    print(f"Using {torch.cuda.get_device_name(0)}")
experutils.set_seeds(args.seed)

ablation_config_files = [
    "ablation_base0.json",
    "ablation_emb_auto1.json",
    "ablation_emb_auto-noise2.json",
    "ablation_gates_reg6.json",
    "ablation_emb_reg7.json",
    "ablation_both_reg8.json"
]
ablation_config_files = [os.path.join(args.configs_path, p) for p in ablation_config_files]

runutils.run_synth_ablations(mode=args.dist, ablation_config_files=ablation_config_files, args=args)
