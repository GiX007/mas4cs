"""Test for data loading functionality."""

from src.data import load_multiwoz
from src.utils import search_multiwoz_datasets


def test_load_dataset() -> None:
    """
    Quick test to verify dataset loads and has expected splits.
    Run: python -m tests.test_loader
    """
    dataset = load_multiwoz(verbose=True)

    print("\n", dataset)
    print("\nDataset splits:", dataset.keys())
    print("Train size:", len(dataset['train']))
    print("Validation size:", len(dataset['validation']))
    print("Test size:", len(dataset['test']))

    # Show one example
    print("\nFirst sample (dialogue):", dataset['train'][0])


if __name__ == "__main__":

    # Search for MultiWOZ datasets in HuggingFace
    search_multiwoz_datasets()

    # Get the dataset
    test_load_dataset()

