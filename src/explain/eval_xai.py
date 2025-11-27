"""
File: explain/eval_xai.py

Responsibilities:
- Evaluate explanation quality across methods (LIME, SHAP, IG).
- Implement faithfulness (deletion/insertion tests), stability, and plausibility scoring.
- Aggregate metrics into tables/CSVs for comparison across explainer types.

Contributors:
- <Name 1> Vidya Puliadi
- <Name 2>
- <Name 3>

Key functions to implement:
- compute_faithfulness(explanations, model, tokenizer, cfg) -> dict
- compute_stability(explanations_original, explanations_perturbed, cfg) -> dict
- compute_plausibility(explanations, human_labels) -> dict
- summarize_xai_metrics(all_metrics: dict, out_path: str) -> None
"""

# src/explain/eval_xai.py
#
# Super simple evaluation code for LIME / SHAP / IG.
# The goal is just to get basic numbers for:
#   - faithfulness
#   - stability
#   - plausibility

import json
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def top_k_indices(scores, k):
    if not scores:
        return []
    k = min(k, len(scores))
    return sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)[:k]


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def remove_tokens(tokens, bad_idx):
    return " ".join(t for i, t in enumerate(tokens) if i not in bad_idx)


def predict_proba(text: str, model, tokenizer, device, label_idx: int = 1) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits  # shape (1, num_labels)
        probs = F.softmax(logits, dim=-1)[0]
    return probs[label_idx].item()


def compute_faithfulness(explanations, model, tokenizer, cfg=None) -> dict:
    top_k = 3 if cfg is None else cfg.get("top_k", 3)
    max_examples = 200 if cfg is None else cfg.get("max_examples", 200)
    label_idx = 1 if cfg is None else cfg.get("label_idx", 1)

    device = next(model.parameters()).device
    drops = []

    for ex in explanations[:max_examples]:
        text = ex["text"]
        tokens = ex.get("tokens") or text.split()
        imps = ex["importances"]  # list of floats

        p_orig = float(ex["prob_pred"]) if "prob_pred" in ex else \
            predict_proba(text, model, tokenizer, device, label_idx)

        bad_idx = top_k_indices(imps, top_k)
        masked_text = remove_tokens(tokens, bad_idx)
        p_masked = predict_proba(masked_text, model, tokenizer, device, label_idx)

        drops.append(p_orig - p_masked)

    if not drops:
        return {"avg_drop": 0.0, "n": 0}

    avg = sum(drops) / len(drops)
    return {"avg_drop": avg, "n": len(drops)}



def compute_stability(explanations_original, explanations_perturbed, cfg=None) -> dict:
    max_pairs = 200 if cfg is None else cfg.get("max_pairs", 200)

    orig_by_id = {ex["id"]: ex for ex in explanations_original if "id" in ex}
    pert_by_id = {ex["id"]: ex for ex in explanations_perturbed if "id" in ex}

    sims = []
    for ex_id, ex_orig in orig_by_id.items():
        if ex_id not in pert_by_id:
            continue
        ex_pert = pert_by_id[ex_id]

        imp1 = ex_orig["importances"]
        imp2 = ex_pert["importances"]
        L = min(len(imp1), len(imp2))
        if L == 0:
            continue
        sims.append(cosine_similarity(imp1[:L], imp2[:L]))
        if len(sims) >= max_pairs:
            break

    if not sims:
        return {"avg_sim": 0.0, "n": 0}

    avg = sum(sims) / len(sims)
    return {"avg_sim": avg, "n": len(sims)}


def compute_plausibility(explanations, human_labels) -> dict:
    sums = {}
    counts = {}
    for row in human_labels:
        m = row["method"]
        r = float(row["rating"])
        sums[m] = sums.get(m, 0.0) + r
        counts[m] = counts.get(m, 0) + 1

    avg = {m: sums[m] / counts[m] for m in sums}
    return avg


def summarize_xai_metrics(all_metrics: dict, out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved XAI metrics to {out_path}")


if __name__ == "__main__":

    from src.models import get_tokenizer, get_distilbert_model  # adjust if needed

    tokenizer = get_tokenizer()
    model = get_distilbert_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    datasets = ["kaggle", "liar"]
    methods = ["lime", "shap", "ig"]

    all_metrics = {}

    # Faithfulness: basic deletion test
    for ds in datasets:
        for m in methods:
            key = f"{ds}_{m}"
            path = Path(f"artifacts/explanations/{key}.jsonl")
            if not path.exists():
                print(f"[faithfulness] Skipping {key} (no {path})")
                continue

            exs = load_jsonl(str(path))
            faith = compute_faithfulness(
                exs,
                model,
                tokenizer,
                cfg={"top_k": 3, "max_examples": 100, "label_idx": 1},
            )
            all_metrics.setdefault(key, {})["faithfulness"] = faith
            print(f"[faithfulness] {key}: {faith}")

    # Stability: compare original vs. perturbed explanations if available
    for ds in datasets:
        for m in methods:
            key = f"{ds}_{m}"
            orig_path = Path(f"artifacts/explanations/{key}.jsonl")
            pert_path = Path(f"artifacts/explanations/{key}_perturbed.jsonl")

            if not (orig_path.exists() and pert_path.exists()):
                print(f"[stability] Skipping {key} (need {orig_path} AND {pert_path})")
                continue

            ex_orig = load_jsonl(str(orig_path))
            ex_pert = load_jsonl(str(pert_path))

            stab = compute_stability(ex_orig, ex_pert, cfg={"max_pairs": 200})
            all_metrics.setdefault(key, {})["stability"] = stab
            print(f"[stability] {key}: {stab}")

    # Plausibility: average human ratings per method
    ratings_path = Path("artifacts/explanations/human_ratings.jsonl")
    if ratings_path.exists():
        human_labels = load_jsonl(str(ratings_path))
        plaus = compute_plausibility([], human_labels)
        all_metrics["plausibility"] = plaus
        print(f"[plausibility] {plaus}")
    else:
        print("[plausibility] No human_ratings.jsonl found, skipping plausibility.")

    summarize_xai_metrics(all_metrics, "artifacts/metrics/xai_metrics.json")

