import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


RANDOM_SEED = 42
LOCAL_DATASET_PATH = Path("data/raw/aclImdb")
TEST_POS_PATH = LOCAL_DATASET_PATH / "test" / "pos"
TEST_NEG_PATH = LOCAL_DATASET_PATH / "test" / "neg"
MODEL_PATH = Path("models/sentiment_model.joblib")


class LocalDatasetError(Exception):
    """Raised when the expected local IMDb test folders are unavailable."""


def get_sample_size() -> int:
    """Read TEST_SAMPLE_SIZE from the environment."""
    value = os.getenv("TEST_SAMPLE_SIZE")
    if value is None:
        return 500

    try:
        sample_size = int(value)
    except ValueError:
        print("TEST_SAMPLE_SIZE must be a number. Using default: 500")
        return 500

    if sample_size <= 0:
        print("TEST_SAMPLE_SIZE must be greater than 0. Using default: 500")
        return 500

    return sample_size


def validate_local_dataset() -> None:
    """Confirm the local Stanford IMDb test folders exist."""
    missing_folders = [
        folder for folder in (TEST_POS_PATH, TEST_NEG_PATH) if not folder.exists()
    ]

    if missing_folders:
        missing = ", ".join(str(folder) for folder in missing_folders)
        raise LocalDatasetError(
            "Local IMDb test folders not found. Download and extract "
            f"the Stanford IMDb dataset into data/raw/. Missing: {missing}"
        )


def read_reviews(folder: Path, label: int) -> pd.DataFrame:
    """Read review text files from one IMDb sentiment folder."""
    rows = [
        {"text": review_path.read_text(encoding="utf-8"), "label": label}
        for review_path in sorted(folder.glob("*.txt"))
    ]
    return pd.DataFrame(rows, columns=["text", "label"])


def load_test_dataset() -> pd.DataFrame:
    """Load labeled IMDb test data from local files."""
    validate_local_dataset()

    pos_df = read_reviews(TEST_POS_PATH, label=1)
    neg_df = read_reviews(TEST_NEG_PATH, label=0)
    return pd.concat([pos_df, neg_df], ignore_index=True)


def prepare_sample(test_df: pd.DataFrame, test_sample_size: int) -> pd.DataFrame:
    """Shuffle and sample test data with a fixed random seed."""
    sample_count = min(test_sample_size, len(test_df))
    shuffled = test_df.sample(frac=1, random_state=RANDOM_SEED)
    return shuffled.head(sample_count).reset_index(drop=True)


def main() -> None:
    """Evaluate the saved sentiment model on local IMDb test reviews."""
    if not MODEL_PATH.exists():
        print("Model not found. Run python -m sentiment_analyzer.train first.")
        return

    try:
        test_df = load_test_dataset()
    except LocalDatasetError as exc:
        print(exc)
        return

    test_sample_size = get_sample_size()
    test_sample = prepare_sample(test_df, test_sample_size)

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test_sample["text"])
    labels = test_sample["label"]

    print(f"accuracy: {accuracy_score(labels, predictions):.4f}")
    print(f"precision: {precision_score(labels, predictions):.4f}")
    print(f"recall: {recall_score(labels, predictions):.4f}")
    print(f"F1 score: {f1_score(labels, predictions):.4f}")


if __name__ == "__main__":
    main()
