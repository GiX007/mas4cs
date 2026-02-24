"""
Dataset Loader: HuggingFace MultiWOZ 2.2 interface.

Loads raw MultiWOZ 2.2 dataset from HuggingFace Hub with automatic caching to avoid repeated downloads. 
Also provides access to the loaded dataset per split.
Entry point for all dataset operations.
"""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from src.utils import print_separator


def load_multiwoz(verbose: bool = False) -> Any:
    """
    Load MultiWOZ 2.2 dataset from HuggingFace.
    Dataset is always cached under <project_root>/hf_cache/datasets.

    Args:
        verbose: If True, print loading messages (Default: False)

    Returns:
        HuggingFace DatasetDict with 'train', 'validation', 'test' splits
    """

    if verbose:
        print_separator("LOADING MULTIWOZ 2.2 DATASET")

    project_root = Path(__file__).resolve().parents[1]  # data_loader.py is in src/, parents[1] = project root
    cache_dir = project_root / "hf_cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("pfb30/multi_woz_v22", trust_remote_code=True, cache_dir=str(cache_dir))

    if verbose:
        print("\nDataset loaded successfully!")

    return dataset


def load_split_data(filepath: str, split: str) -> list[dict[str, Any]]:
    """
    Load a specific split from the processed MultiWOZ dataset.

    Args:
        filepath: Path to filtered_processed_multiwoz22.json
        split: Split name ('train', 'validation', or 'test')

    Returns:
        List of dialogue dictionaries from the specified split

    Raises:
        KeyError: If split name not found in dataset
        FileNotFoundError: If filepath does not exist
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(f"Split '{split}' not found. Available: {list(data.keys())}")

    return data[split]

