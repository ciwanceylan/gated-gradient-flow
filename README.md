# Gated gradient model reproducibility code

## Run instructions
The repository contains the data and code needed to reproduce the experimental results of the paper **"Learning node representations using stationary flow prediction on large payment and cash transaction networks"**.
The bash script `run_eth_experiment.sh` will run the hyperparameter search for the gated gradient model and all the baselines using the ethereum data, which will have to be downloaded separately, see LINK. 
`run_abl_unimodal.sh` and `run_abl_mulimodal.sh` will reproduce the synthetic flow experimental results.

**NB:** Running all the experiments may take a long time. Using a GPU `run_abl_unimodal.sh` and `run_abl_mulimodal.sh` may take a few hours each and  `run_eth_experiment.sh` may take up to 36 hours.
To avoid having to rerun all experiments, the result files have been included in the results folder.
The figures and tables of the paper can be reproduced from these results using the three notebooks.

## Installation instructions

To install the minimal requirements run
```bash
pip install -r requirements_min.txt
```
This will install everything necessary to run `run_eth_experiment.sh`, `run_abl_unimodal.sh` and `run_abl_mulimodal.sh`.
- To run `train_node2vec_emb.sh` also install [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/).
- To run `preprocess_network_data.py`  also install [graph-tool](https://graph-tool.skewed.de/).
- To run the notebooks also install jupyter, matplotlib, seaborn and tikzplotlib.
 