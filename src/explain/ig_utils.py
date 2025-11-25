"""
File: explain/ig_utils.py

Responsibilities:
- Implement Integrated Gradients (IG) for DistilBERT
- Optionally provide LayerIntegratedGradients for layer-aware attributions.
- Convert raw attributions to human-readable token importance scores.

Contributors:
- Anton Nemchinski

Key functions to implement:
- compute_ig_attributions(text_batch: list[str], model, tokenizer, cfg) -> list[dict]
- compute_layer_ig_attributions(text_batch: list[str], model, tokenizer, layer, cfg) -> list[dict]
- format_attributions(tokens: list[str], attributions) -> dict

Output format (per explanation dict):
{
    "sample_id": int,
    "text": str,
    "tokens": [str],
    "importances": [float],
    "pred_label": int,
    "true_label": int
}

The unified runner will later add:
    - "method": "ig"
    - "dataset": "kaggle" | "liar"
"""

import torch
from torch.nn.functional import softmax
from tqdm import tqdm


def explain_sample_ig(
    text,
    model,
    tokenizer,
    cfg,
    sample_id=None,
    true_label=None,
):
    device = next(model.parameters()).device
    model.eval()

    ig_cfg = cfg.get("ig", {})
    n_steps = ig_cfg.get("n_steps", 32)
    max_len = ig_cfg.get("max_seq_length", cfg["data"]["max_length"])

    # ---- 1. Tokenize ----
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # ---- 2. Get embeddings ----
    embedding_layer = model.get_input_embeddings()
    x = embedding_layer(input_ids)           # [1, seq_len, hidden_dim]
    x0 = torch.zeros_like(x)                 # baseline

    # ---- 3. Predicted label ----
    with torch.no_grad():
        out = model(**enc)
        probs = softmax(out.logits, dim=-1)
        pred_label = int(torch.argmax(probs, dim=-1).item())

    # ---- 4. Integrated Gradients (manual safe version) ----
    total_grad = torch.zeros_like(x)

    for k in range(1, n_steps + 1):
        alpha = float(k) / n_steps

        # Non-leaf tensor: we must ask PyTorch to keep grad for it
        x_step = x0 + alpha * (x - x0)      # [1, seq_len, hidden_dim]
        x_step.requires_grad_(True)
        x_step.retain_grad()               # <<--- IMPORTANT

        logits = model(
            inputs_embeds=x_step,
            attention_mask=attention_mask,
        ).logits

        logit_for_label = logits[0, pred_label]
        logit_for_label.backward(retain_graph=True)

        # Now x_step.grad is populated because of retain_grad()
        total_grad += x_step.grad.detach()

        # Clear grads on model params so they don't accumulate
        model.zero_grad(set_to_none=True)

    # ---- 5. Final IG attribution ----
    ig = (x - x0) * total_grad / n_steps     # [1, seq_len, hidden_dim]

    # Collapse hidden_dim → one score per token
    token_importances = ig.sum(dim=-1).squeeze(0)  # [seq_len]

    # Normalize (L2)
    denom = torch.norm(token_importances) + 1e-8
    token_importances = (token_importances / denom).cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    return {
        "sample_id": sample_id,
        "text": text,
        "tokens": tokens,
        "importances": token_importances,
        "pred_label": pred_label,
        "true_label": true_label,
    }


def run_ig_batch(dataset, model, tokenizer, cfg):
    n_total = len(dataset)
    ig_cfg = cfg.get("ig", {})
    num_samples = ig_cfg.get("num_explain_samples", min(50, n_total))

    print(f"Running IG explanations on {num_samples} samples...")

    results = []
    for i in tqdm(range(num_samples), desc="IG Batch"):
        try:
            sample = dataset[i]
            text = sample["text"]
            label = int(sample.get("label", -1))

            exp = explain_sample_ig(
                text=text,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                sample_id=i,
                true_label=label,
            )
            results.append(exp)
        except Exception as e:
            print(f"Error generating IG at index {i}: {e}")
            continue

    return results