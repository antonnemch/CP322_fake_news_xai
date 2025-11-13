"""
File: explain/lime_utils.py

Responsibilities:
- Wrap LIME's text explainer for use with our trained DistilBERT model.
- Provide functions to generate local explanations for individual samples.
- Optionally, batch-run LIME on a subset of test examples and serialize outputs.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- make_lime_explainer(cfg, class_names: list[str]) -> LimeTextExplainer
- explain_sample_lime(text: str, model, tokenizer, explainer, cfg) -> dict
- run_lime_batch(dataset, model, tokenizer, cfg) -> list[dict]
"""