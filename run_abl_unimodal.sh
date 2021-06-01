#!/bin/bash

python run_ablation.py \
  --configs_path=data/configs_abl_unimodal/ \
  --data_path=data/ \
  --dist=unimodal \
  --graph=complete \
  --num_flows=10 \
  --use_gpu \
  --seed=56 \
  results/complete_unimodal

python run_ablation.py \
  --configs_path=data/configs_abl_unimodal/\
  --data_path=data/ \
  --dist=unimodal \
  --graph=cora \
  --num_flows=10 \
  --use_gpu \
  --seed=1234 \
  results/cora_unimodal

python run_ablation.py \
  --configs_path=data/configs_abl_unimodal/ \
  --data_path=data/ \
  --dist=unimodal \
  --graph=bitcoin \
  --num_flows=10 \
  --use_gpu \
  --seed=779 \
  results/bitcoin_unimodal