"""
File: explain/shap_utils.py

Responsibilities:
- Wrap SHAP (KernelSHAP or other variants) for generating explanations on text inputs.
- Provide utilities to compute token- or word-level SHAP values for a subset of samples.
- Serialize SHAP outputs for later metric calculations and visualization.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- make_shap_explainer(model, tokenizer, cfg)
- explain_sample_shap(text: str, explainer, cfg) -> dict
- run_shap_batch(dataset, model, tokenizer, cfg) -> list[dict]
"""