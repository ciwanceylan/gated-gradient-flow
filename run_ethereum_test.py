import os

import pandas as pd
import torch

import src.testutils as testutils
import src.dataprocessing as dataproc
import src.utils
from src.testutils import split_baselines


def main():
    folder = "results/ethereum/"
    data_folder = "data/"
    test_cache_folder = 'results/ethereum/test_cache'
    MODEL_ORDER = ["gated", "grad", "dnn2_engi_feat", "dnn2_node2vec", "fairness_goodness", "zeros", "init"]

    os.makedirs(test_cache_folder, exist_ok=True)

    results = dict()

    results["gated"] = pd.read_csv(folder + "ethereum_joint_results.csv")
    results["grad"] = pd.read_csv(folder + "ethereum_grad_baseline_results.csv")
    results.update(split_baselines(pd.read_csv(folder + "ethereum_baseline_results.csv")))
    # results["init"] = pd.read_csv(folder + "ethereum_init_results.csv")

    best_hyperparameters, _ = testutils.get_all_best_hp(results, eval_col=["val_MeAE*", "val_MeAE"])
    print("The best hyperparameters")
    for model_name, best_hp in best_hyperparameters.items():
        print(model_name, ": ", best_hp)

    model_paths = testutils.get_all_best_model_paths(best_hyperparameters, os.path.join(folder, "chpt"))
    print("loading models from")
    for model_name, path in model_paths.items():
        print(model_name, ":")
        print(path)

    if os.path.exists(os.path.join(folder, "preprocessed_ethereum_train.csv")):
        train_graph = dataproc.Graph.read_csv(os.path.join(folder, "preprocessed_ethereum_train.csv"))
        test_graph = dataproc.Graph.read_csv(os.path.join(folder, "preprocessed_ethereum_test.csv"))
    else:
        train_graph = dataproc.Graph.read_csv(os.path.join(data_folder, "eth_split", "preprocessed_ethereum_train.csv"))
        test_graph = dataproc.Graph.read_csv(os.path.join(data_folder, "eth_split", "preprocessed_ethereum_test.csv"))

    # Load any config file saved during training. Only values common to all config files are used
    common_configs = src.utils.load_configs(folder + f"configs_1_0_0.json")

    results, all_flows, representations = testutils.evaluate_all_models(MODEL_ORDER, model_paths=model_paths,
                                                                        train_graph=train_graph, test_graph=test_graph,
                                                                        common_configs=common_configs,
                                                                        fg_powerfactor=
                                                                        best_hyperparameters["fairness_goodness"][
                                                                            "powerfactor"])
    results.to_csv(os.path.join(test_cache_folder, "results_table.csv"))
    torch.save(all_flows, os.path.join(test_cache_folder, "flows.pth"))
    torch.save(representations, os.path.join(test_cache_folder, "representations.pth"))


if __name__ == "__main__":
    main()
