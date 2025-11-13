"""
File: data.py

Responsibilities:
- Load raw datasets (Kaggle Fake News, LIAR) from data/raw.
- Clean and preprocess text (e.g., remove boilerplate, handle missing values).
- Create train/validation/test splits and save processed versions.
- Provide helper functions to return dataset objects or DataLoaders for training and evaluation.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- load_raw_datasets(cfg) -> dict
- preprocess_examples(examples, cfg) -> dict
- build_splits(cfg) -> (train_dataset, val_dataset, test_dataset)
- get_hf_datasets(cfg) -> DatasetDict
"""