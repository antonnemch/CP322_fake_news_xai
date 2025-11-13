#!/usr/bin/env bash
# File: run_eval.sh
#
# Responsibilities:
# - Thin wrapper to launch evaluation of a trained model.
# - Can be extended to take a checkpoint path as an argument.
#
# Contributors:
# - <Name 1>
# - <Name 2>
# - <Name 3>

python -m src.evaluate --config config/default.yaml