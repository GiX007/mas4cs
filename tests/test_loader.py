"""Test for data loading functionality."""
from src.data import load_multiwoz, load_split


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


def test_load_dataset_v2() -> None:
    """
    Quick test to verify official MultiWOZ 2.2 loads and filters correctly.
    Run: python -m tests.test_loader
    """
    for split in ("train", "dev", "test"):
        load_split(split, verbose=True)


if __name__ == "__main__":

    # Search for MultiWOZ datasets in HuggingFace
    # from src.utils import search_multiwoz_datasets
    # search_multiwoz_datasets()

    # Test the loader on the MultiWOZ 2.2 from HuggingFace
    # test_load_dataset()

    # Test the v2 loader on the official MultiWOZ 2.2 from GitHub
    test_load_dataset_v2()
