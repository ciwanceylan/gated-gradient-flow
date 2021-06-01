#!/bin/bash

python run_ablation.py \
  --configs_path=data/configs_abl_multimodal/ \
  --data_path=data/ \
  --dist=multimodal \
  --graph=complete \
  --num_flows=10 \
  --use_gpu \
  --seed=56 \
  results/complete_multimodal

python run_ablation.py \
  --configs_path=data/configs_abl_multimodal/ \
  --data_path=data/ \
  --dist=multimodal \
  --graph=cora \
  --num_flows=10 \
  --use_gpu \
  --seed=1234 \
  results/cora_multimodal

python run_ablation.py \
  --configs_path=data/configs_abl_multimodal/ \
  --data_path=data/ \
  --dist=multimodal \
  --graph=bitcoin \
  --num_flows=10 \
  --use_gpu \
  --seed=779 \
  results/bitcoin_multimodal