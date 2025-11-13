"""
File: models.py

Responsibilities:
- Define and construct baseline models (e.g., TF-IDF + Logistic Regression).
- Load and configure DistilBERT-based sequence classification model from Hugging Face.
- Provide utility functions to initialize tokenizers and models in a consistent way.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- build_baseline_model(cfg) -> sklearn model
- get_tokenizer(cfg) -> PreTrainedTokenizerFast
- get_distilbert_model(cfg) -> PreTrainedModel
"""