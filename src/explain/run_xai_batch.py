"""
Unified XAI batch runner: IG, LIME, SHAP on Kaggle or LIAR.

Usage examples:

    # IG only on Kaggle
    PYTHONPATH=. python -m src.explain.run_xai_batch --dataset kaggle --methods ig

    # IG + LIME + SHAP on LIAR
    PYTHONPATH=. python -m src.explain.run_xai_batch --dataset liar --methods ig lime shap
"""
import numpy as np

import argparse
import json
from pathlib import Path
from typing import Dict, Callable, List

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import load_config
from src.explain.ig_utils import run_ig_batch
from src.explain.lime_utils import run_lime_batch
from src.explain.shap_utils import run_shap_batch


# Map method name → batch runner
METHOD_RUNNERS: Dict[str, Callable] = {
    "ig": run_ig_batch,
    "lime": run_lime_batch,
    "shap": run_shap_batch,
}

def _json_default(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def load_processed_test_split(dataset_name: str) -> Dataset:
    """
    Load processed test split from:
        data/processed/<dataset_name>_test.csv

    Expect columns: 'text', 'label'
    """
    path = Path("data/processed") / f"{dataset_name}_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed test CSV: {path}")

    df = pd.read_csv(path)
    expected_cols = {"text", "label"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {expected_cols}, got {set(df.columns)}")

    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kaggle", "liar"], required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ig", "lime", "shap"],
        help="Any subset of: ig lime shap",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="YAML config path (for max_length, XAI params, etc.)",
    )
    args = parser.parse_args()

    # Validate methods
    for m in args.methods:
        if m not in METHOD_RUNNERS:
            raise ValueError(f"Unknown method '{m}'. Available: {list(METHOD_RUNNERS.keys())}")

    # Load config
    cfg = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[xai] Device: {device}")

    # Load fine-tuned model and tokenizer
    model_dir = Path("artifacts/distilbert") / args.dataset / "final_model"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {model_dir}\n"
            f"Make sure Zaid's training script has saved the final model there."
        )

    print(f"[xai] Loading tokenizer + model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load processed test split as a HF Dataset
    test_ds = load_processed_test_split(args.dataset)

    out_dir = Path("artifacts/explanations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run each method on the same test dataset (each method will choose its own subset size)
    for method in args.methods:
        runner = METHOD_RUNNERS[method]
        print(f"[xai] Running {method.upper()} on dataset '{args.dataset}'")

        results: List[Dict] = runner(test_ds, model, tokenizer, cfg)

        # Attach metadata
        for r in results:
            r["dataset"] = args.dataset
            r["method"] = method

        out_path = out_dir / f"{args.dataset}_{method}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, default=_json_default) + "\n")

        print(f"[xai] Saved {len(results)} {method.upper()} explanations to {out_path}")

    print("[xai] All requested XAI methods completed.")


if __name__ == "__main__":
    main()
