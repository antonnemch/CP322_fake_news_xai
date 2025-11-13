#!/usr/bin/env bash
# File: run_xai.sh
#
# Responsibilities:
# - Orchestrate running LIME, SHAP, and IG on a subset of the test data.
# - Store explanations and XAI metrics under artifacts/explanations and artifacts/metrics.
#
# Contributors:
# - <Name 1>
# - <Name 2>
# - <Name 3>

python -m src.explain.eval_xai --config config/default.yaml
