"""
File: layers/probe.py

Responsibilities:
- Perform layer-wise probing on DistilBERT hidden states.
- Train simple classifiers (e.g., logistic regression) on each layer's representation to predict fake vs. real.
- Save probe accuracies and plots to understand where task information emerges.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- extract_layer_representations(model, tokenizer, dataset, cfg) -> dict[layer_index -> np.array]
- train_probes(layer_reps: dict, labels: np.array, cfg) -> dict[layer_index -> float]
- save_probe_results(results: dict, out_path: str) -> None
"""