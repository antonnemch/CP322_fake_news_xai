"""
File: explain/lime_utils.py

Responsibilities:
- Wrap LIME's text explainer for use with our trained DistilBERT model.
- Provide functions to generate local explanations for individual samples.
- Optionally, batch-run LIME on a subset of test examples and serialize outputs.

Contributors:
- Ryan Wilson
- <Name 2>
- <Name 3>

Key functions to implement:
- make_lime_explainer(cfg, class_names: list[str]) -> LimeTextExplainer
- explain_sample_lime(text: str, model, tokenizer, explainer, cfg) -> dict
- run_lime_batch(dataset, model, tokenizer, cfg) -> list[dict]
"""
import numpy as np
import torch 
from tqdm import tqdm
from typing import List, Dict, Any 
from lime.lime_text import LimeTextExplainer

def make_lime_explainer(cfg, class_names: List[str]):
    """
    Create a LIME text explainer with parameters from config object.
    Uses cfg.lime.* hyperparameters if available.

    Args:
        cfg: config dictionary
        class_names: names for each prediction class

    Returns:
        LimeTextExplainer
    """
    lime_cfg = cfg.get("lime", {})

    #Build LIME explainer
    return LimeTextExplainer(
        class_names=class_names,
        bow=lime_cfg.get("bow", True),
        split_expression=lime_cfg.get("split_expression", r'\W+'),
        random_state=cfg.get("seed", 42)
    )

def explain_sample_lime(
        text: str,
        model,
        tokenizer,
        explainer: LimeTextExplainer,
        cfg,
        sample_id: int | str = None, # type: ignore
        true_label: int | None = None
):
  """
  Generate LIME explanation for a single text sample.
  Returns a dict formatted according to project standards.

  Args:
      text: input string
      model: Transformer model (DistilBERT fine-tuned)
      tokenizer: HuggingFace tokenizer
      explainer: LimeTextExplainer instance
      cfg: config dictionary
      sample_id: numeric/string ID
      true_label: optional class label

  Returns:
      dict with explanation info
  """
  model.eval()
  lime_cfg = cfg.get("lime", {})
  device = next(model.parameters()).device

  #Prediction function for LIME
  def lime_predict(texts: List[str]):
      """
      Prediction function for LIME.
      Takes a list/array of text strings and returns class probabilities.

      Args:
          texts: List or array of text strings
      Returns:
          numpy array of shape (n_samples, n_classes)
      """
      inputs = tokenizer(
          texts,
          return_tensors='pt',
          padding=True,
          truncation=True,
          max_length=lime_cfg.get("max_seq_length", cfg["data"]["max_length"])
      )

      #Move inputs to device
      inputs = {k: v.to(device) for k, v in inputs.items()}

      with torch.no_grad():
          preds = model(**inputs).logits
          probs = torch.softmax(preds, dim=-1).cpu().numpy()
      return probs
  
  #Get model prediction
  inputs = tokenizer(
      text,
      return_tensors='pt',
      padding=True,
      truncation=True,
      max_length=lime_cfg.get("max_seq_length", cfg["data"]["max_length"])
  )
  inputs = {k: v.to(device) for k, v in inputs.items()}
  with torch.no_grad():
    logits = model(**inputs).logits
  pred_label = int(logits.argmax(dim=-1).item())

  # Run LIME
  explanation = explainer.explain_instance(
      text_instance=text,
      classifier_fn=lime_predict,
      num_features=lime_cfg.get("num_features", 10),
      num_samples=lime_cfg.get("num_samples", 500),
      labels=(pred_label,) 
  )

  #Extract tokens and importance weights
  token_importances = dict(explanation.as_list(label=pred_label))
  tokens = list(token_importances.keys())
  importances = list(token_importances.values())

  result_dict = {
      "sample_id": sample_id if sample_id is not None else 0,
      "text": text,
      "tokens": tokens,
      "importances": importances,
      "pred_label": pred_label,
      "true_label": true_label
  }

  return result_dict
      

def run_lime_batch(dataset, model, tokenizer, cfg):
  """
  Run LIME explanations over a subset of dataset

  Args:
      dataset: torch Dataset or list-like
      model: DistilBERT model
      tokenizer: HuggingFace tokenizer
      cfg: config dictionary

  Returns:
      List[dict] formatted explanations
  """
  lime_cfg = cfg.get("lime", {})

  # Check for empty dataset
  if len(dataset) == 0:
    print("Warning: Empty dataset provided")
    return []

  num_samples = lime_cfg.get("num_explain_samples", min(100, len(dataset)))
  class_names = cfg.get("class_names", ["fake", "real"])
  explainer = make_lime_explainer(cfg, class_names)
  results: List[Dict[str, Any]] = []

  print(f"Running LIME explanations on {num_samples} samples...")

  #Run through sample batch
  for i in tqdm(range(num_samples), desc="LIME Batch"):
      try:
        sample = dataset[i]
        text = sample["text"]
        true_label = int(sample.get("label", -1))

        explanation = explain_sample_lime(
            text=text,
            model=model,
            tokenizer=tokenizer,
            explainer=explainer,
            cfg=cfg,
            sample_id=i,
            true_label=true_label
        )
        results.append(explanation)
      except Exception as e:
        print(f"Error accessing dataset at index {i}: {e}")
        continue
  
  return results 
   
  



