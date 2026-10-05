#!/usr/bin/env bash
# Thin launcher for appearance-slider training. Edit hyperparameters in
# configs/appearance.yaml; swap the slider by changing --direction.
set -e
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 accelerate launch -m tokendial.train \
  --config configs/appearance.yaml \
  --direction configs/directions/person_older.json
