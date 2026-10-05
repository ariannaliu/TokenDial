#!/usr/bin/env bash
# Thin launcher for motion-slider training. Edit hyperparameters in configs/motion.yaml.
set -e
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 accelerate launch -m tokendial.train \
  --config configs/motion.yaml
