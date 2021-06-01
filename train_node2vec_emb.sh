#!/bin/bash

python run_node2vec_training.py \
  --data_path=data/preprocessed_ethereum_2018_2020.csv \
  --num_epochs=10 \
  --batch_size=128 \
  --seed=1234 \
  --use_gpu \
  results/node2vec_eth

python run_node2vec_training.py \
  --data_path=completegraph \
  --num_epochs=40 \
  --batch_size=400 \
  --seed=1234 \
  --use_gpu \
  results/node2vec_complete

python run_node2vec_training.py \
  --data_path=data/preprocessed_cora.csv \
  --num_epochs=10 \
  --batch_size=256 \
  --seed=1234 \
  --use_gpu \
  results/node2vec_cora

python run_node2vec_training.py \
  --data_path=data/preprocessed_bitcoin.csv \
  --num_epochs=10 \
  --batch_size=256 \
  --seed=1234 \
  --use_gpu \
  results/node2vec_bitcoin