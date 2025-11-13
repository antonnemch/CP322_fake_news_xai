# Explainable Fake News Detection with DistilBERT

This project fine-tunes **DistilBERT** to classify news as *fake* or *real* on two benchmark datasets (Kaggle Fake News and LIAR), and then applies three explainability techniques (**LIME, SHAP, Integrated Gradients**) to understand *why* the model makes its predictions. We quantitatively evaluate the explanations (faithfulness, stability, plausibility) and perform a simple **layer-wise analysis** of DistilBERT to see where in the network the fake/real signal emerges.

---

## Repository Layout

```text
fake-news-xai/
├─ README.md                 # Project description, instructions, and repo overview
├─ LICENSE                   # License for code and documentation (TBD)
├─ .gitignore                # Ignore venvs, caches, data, and generated artifacts
├─ requirements.txt          # Pinned Python dependencies for reproducible setups
├─ Makefile                  # Convenience commands (setup, data, train, eval, xai, layers)
├─ config/
│  ├─ default.yaml           # Main configuration (dataset, model, training, XAI settings)
│  └─ paths.yaml             # Optional machine-specific paths (e.g., HF cache directory)
├─ data/
│  ├─ raw/                   # Raw downloaded datasets (not committed to git)
│  └─ processed/             # Preprocessed/split datasets ready for training
├─ notebooks/
│  ├─ 00_eda.ipynb           # Exploratory data analysis of the datasets
│  └─ 10_figures.ipynb       # Notebook for generating final report/presentation figures
├─ src/
│  ├─ __init__.py            # Marks src as a Python package
│  ├─ data.py                # Data loading, cleaning, preprocessing, and splitting logic
│  ├─ models.py              # Baseline and DistilBERT model construction helpers
│  ├─ train.py               # Training entry point (reads config, trains, saves checkpoints)
│  ├─ evaluate.py            # Evaluation entry point (loads model, computes metrics)
│  ├─ utils.py               # Shared utilities (seeding, logging, path helpers, config loading)
│  ├─ explain/
│  │  ├─ lime_utils.py       # LIME wrapper functions for text explanations
│  │  ├─ shap_utils.py       # SHAP wrapper functions for text explanations
│  │  ├─ ig_utils.py         # Integrated Gradients + LayerIG helpers via Captum
│  │  └─ eval_xai.py         # Faithfulness, stability, plausibility scoring for explanations
│  └─ layers/
│     ├─ probe.py            # Layer-wise probing to see which layers encode fake/real best
│     └─ attention_viz.py    # Helpers to visualize BERT attention for selected examples
├─ scripts/
│  ├─ prepare_data.sh        # Download/unpack datasets and trigger preprocessing
│  ├─ run_train.sh           # Wrapper to launch training with src/train.py
│  ├─ run_eval.sh            # Wrapper to evaluate models with src/evaluate.py
│  ├─ run_xai.sh             # Wrapper to run LIME/SHAP/IG and XAI metrics
│  └─ run_layers.sh          # Wrapper for layer-wise probing and attention visualization
└─ artifacts/
   ├─ checkpoints/           # Saved model checkpoints per run (gitignored)
   ├─ metrics/               # Evaluation and XAI metric CSVs (gitignored)
   ├─ explanations/          # Stored explanations (e.g., JSON/CSV, heatmaps) (gitignored)
   ├─ probes/                # Layer probing results and plots (gitignored)
   └─ figures/               # Final figures for report/slides (gitignored)
```

---

## How to Reproduce Our Results (for the Instructor)

> **Note:** Exact commands may be adjusted as the project evolves, but the overall flow will remain the same.

1. **Clone the repository**
   ```bash
   git clone <REPO_URL> fake-news-xai
   cd fake-news-xai
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download and prepare data**
   ```bash
   make data
   # or
   bash scripts/prepare_data.sh
   ```

5. **Train DistilBERT and baseline models**
   ```bash
   make train
   # trains DistilBERT according to config/default.yaml
   ```

6. **Evaluate classification performance**
   ```bash
   make eval
   # writes metrics to artifacts/metrics/
   ```

7. **Run explainability experiments (LIME, SHAP, IG)**
   ```bash
   make xai
   # runs explainability methods and saves explanations and XAI metrics
   ```

8. **Run model layer analysis**
   ```bash
   make layers
   # generates probing results and attention visualizations
   ```

All key results (metrics, explanations, probing outputs, and plots) will appear under the `artifacts/` directory, organized by run ID, and can be inspected or reused in the report.

---

## Developer Setup (for Contributors)

1. **Clone and create venv**
   ```bash
   git clone <REPO_URL> fake-news-xai
   cd fake-news-xai
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure (optional)**
   - Edit `config/default.yaml` to adjust:
     - dataset (`kaggle_fake_news` vs `liar`)
     - maximum sequence length
     - training hyperparameters (epochs, batch sizes, learning rate)
     - XAI settings (number of samples, top-k values for deletion tests, etc.).
   - Machine-specific paths can go in `config/paths.yaml` if needed.

4. **Run typical workflow during development**
   ```bash
   # 1. Prepare data once
   make data

   # 2. Train model (quick experiment)
   make train

   # 3. Evaluate and inspect metrics
   make eval

   # 4. Run a small XAI subset when testing code
   make xai

   # 5. Run layer analysis when stable
   make layers
   ```

5. **Notebooks**
   - Use `notebooks/00_eda.ipynb` for exploring datasets and sanity-checking preprocessing.
   - Use `notebooks/10_figures.ipynb` to regenerate final plots from saved artifacts (metrics, explanations, probing results).

All experiments should be run through the config files (`config/*.yaml`) and the provided scripts/Makefile so that they remain **reproducible** and easy for others (including the instructor) to re-run on their own machine.
