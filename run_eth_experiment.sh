#!/bin/bash

# Remove debug flag to train model.
python run_ethereum_experiment.py \
  --config_path=data/configs_eth/eth_config_auto_noise.json \
  --split_path=data/eth_split/ \
  --node2vec_path=data/trained_node2vec/node2vec_eth/ \
  --max_dim=3 \
  --min_dim=1 \
  --use_gpu \
  --seed=42 \
  --gates_init=zeros \
  --data_path=data/preprocessed_ethereum_2018_2020.csv \
  --only_baselines \
  --debug \
  debug_results/ethereum_experiment_results
