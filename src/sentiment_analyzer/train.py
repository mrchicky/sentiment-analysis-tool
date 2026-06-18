import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
LOCAL_DATASET_PATH = Path("data/raw/aclImdb")
TRAIN_POS_PATH = LOCAL_DATASET_PATH / "train" / "pos"
TRAIN_NEG_PATH = LOCAL_DATASET_PATH / "train" / "neg"
MODEL_PATH = Path("models/sentiment_model.joblib")


class LocalDatasetError(Exception):
    """Raised when the expected local IMDb dataset folders are unavailable."""


def get_sample_size(name: str, default: int) -> int:
    """Read a positive integer sample size from the environment."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        sample_size = int(value)
    except ValueError:
        print(f"{name} must be a number. Using default: {default}")
        return default

    if sample_size <= 0:
        print(f"{name} must be greater than 0. Using default: {default}")
        return default

    return sample_size


def validate_local_dataset() -> None:
    """Confirm the local Stanford IMDb training folders exist."""
    if not LOCAL_DATASET_PATH.exists():
        raise LocalDatasetError(
            "Local IMDb dataset not found. Download and extract "
            "aclImdb_v1.tar.gz into data/raw/ first."
        )

    missing_folders = [
        folder for folder in (TRAIN_POS_PATH, TRAIN_NEG_PATH) if not folder.exists()
    ]
    if missing_folders:
        missing = ", ".join(str(folder) for folder in missing_folders)
        raise LocalDatasetError(f"Missing training data folder(s): {missing}")


def read_reviews(folder: Path, label: int) -> pd.DataFrame:
    """Read review text files from one IMDb sentiment folder."""
    rows = [
        {"text": review_path.read_text(encoding="utf-8"), "label": label}
        for review_path in sorted(folder.glob("*.txt"))
    ]
    return pd.DataFrame(rows, columns=["text", "label"])


def load_dataset() -> pd.DataFrame:
    """Load labeled IMDb training data from local files."""
    print("loading local IMDb training dataset")
    validate_local_dataset()

    pos_df = read_reviews(TRAIN_POS_PATH, label=1)
    neg_df = read_reviews(TRAIN_NEG_PATH, label=0)
    return pd.concat([pos_df, neg_df], ignore_index=True)


def prepare_sample(train_df: pd.DataFrame, train_sample_size: int) -> pd.DataFrame:
    """Shuffle and sample training data for quick local model training."""
    print("preparing training sample")
    train_count = min(train_sample_size, len(train_df))
    shuffled = train_df.sample(frac=1, random_state=RANDOM_SEED)
    return shuffled.head(train_count).reset_index(drop=True)


def build_model() -> Pipeline:
    """Create the sentiment analysis model pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    """Train and save a lightweight sentiment analysis model."""
    train_sample_size = get_sample_size("TRAIN_SAMPLE_SIZE", 2000)

    try:
        train_df = load_dataset()
    except LocalDatasetError as exc:
        print(exc)
        return

    train_sample = prepare_sample(train_df, train_sample_size)

    print("training model")
    model = build_model()
    model.fit(train_sample["text"], train_sample["label"])

    print("saving model")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("done")


if __name__ == "__main__":
    main()
