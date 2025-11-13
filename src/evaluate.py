"""
File: evaluate.py

Responsibilities:
- Load trained model checkpoints and evaluate on validation/test splits.
- Compute classification metrics (Accuracy, F1, Precision, Recall, etc.).
- Save metrics to CSV files under artifacts/metrics/ for later analysis.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- main(config_path: str, checkpoint_path: str | None = None) -> None
- evaluate_model(model, tokenizer, dataset, cfg) -> dict
- save_metrics(metrics: dict, out_path: str) -> None
"""