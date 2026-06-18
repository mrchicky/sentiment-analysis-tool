import argparse
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset as load_hf_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
DATASET_ID = "cardiffnlp/tweet_eval"
CACHE_DIR = Path("data/raw/tweeteval")
BUNDLE_PATH = Path("models/tweeteval_bundle.joblib")

TASKS = ("sentiment", "irony")
TASK_CHOICES = (*TASKS, "all")
SPLITS = ("train", "test")
LABEL_MAPS = {
    "sentiment": {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    },
    "irony": {
        0: "Not Ironic",
        1: "Ironic",
    },
}


class TweetEvalError(Exception):
    """Raised when TweetEval task, data, or model bundle usage is invalid."""


def validate_task(task: str) -> str:
    """Return a normalized task name if it is supported."""
    normalized = task.strip().lower()
    if normalized not in TASKS:
        supported = ", ".join(TASKS)
        raise TweetEvalError(f"Unsupported task '{task}'. Use one of: {supported}.")
    return normalized


def cache_path(task: str, split: str) -> Path:
    """Return the local CSV cache path for a TweetEval task split."""
    return CACHE_DIR / f"{task}_{split}.csv"


def dataset_to_frame(dataset) -> pd.DataFrame:
    """Convert a Hugging Face Dataset split into the local text/label schema."""
    frame = dataset.to_pandas()
    return frame.loc[:, ["text", "label"]]


def load_split(task: str, split: str) -> pd.DataFrame:
    """Load a TweetEval split from CSV cache, downloading it when absent."""
    task = validate_task(task)
    if split not in SPLITS:
        supported = ", ".join(SPLITS)
        raise TweetEvalError(f"Unsupported split '{split}'. Use one of: {supported}.")

    path = cache_path(task, split)
    if path.exists():
        return pd.read_csv(path)

    print(f"downloading TweetEval {task} {split} split")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_hf_dataset(DATASET_ID, task, split=split)
    frame = dataset_to_frame(dataset)
    frame.to_csv(path, index=False)
    return frame


def load_task_data(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data for one TweetEval task."""
    return load_split(task, "train"), load_split(task, "test")


def build_model() -> Pipeline:
    """Create a scikit-learn text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_task(task: str) -> Pipeline:
    """Train one TweetEval task model."""
    train_df, _ = load_task_data(task)
    model = build_model()

    print(f"training TweetEval {task} model")
    model.fit(train_df["text"].astype(str), train_df["label"].astype(int))
    return model


def train_bundle() -> dict:
    """Train and save the TweetEval-only model bundle."""
    bundle = {
        "dataset_id": DATASET_ID,
        "tasks": {},
    }

    for task in TASKS:
        bundle["tasks"][task] = {
            "model": train_task(task),
            "label_map": LABEL_MAPS[task],
        }

    print("saving TweetEval bundle")
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


def load_bundle() -> dict:
    """Load the TweetEval model bundle from disk."""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "TweetEval bundle not found at models/tweeteval_bundle.joblib.\n"
            "Train it first by running:\n"
            "python -m sentiment_analyzer.tweeteval_bundle train"
        )
    return joblib.load(BUNDLE_PATH)


def evaluate_task(task: str, bundle: dict | None = None) -> dict:
    """Evaluate one model from the TweetEval bundle on its test split."""
    task = validate_task(task)
    bundle = load_bundle() if bundle is None else bundle

    _, test_df = load_task_data(task)
    task_bundle = bundle["tasks"][task]
    model = task_bundle["model"]
    label_map = task_bundle["label_map"]

    labels = test_df["label"].astype(int)
    predictions = model.predict(test_df["text"].astype(str))
    target_names = [label_map[label] for label in sorted(label_map)]

    return {
        "task": task,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=sorted(label_map),
            target_names=target_names,
            zero_division=0,
        ),
    }


def evaluate_bundle() -> list[dict]:
    """Evaluate every model in the TweetEval bundle."""
    bundle = load_bundle()
    return [evaluate_task(task, bundle) for task in TASKS]


def predict(task: str, text: str, bundle: dict | None = None) -> dict:
    """Predict one TweetEval task label for text."""
    task = validate_task(task)
    cleaned_text = text.strip() if isinstance(text, str) else ""
    if not cleaned_text:
        raise ValueError("Please enter some text to predict.")

    bundle = load_bundle() if bundle is None else bundle
    task_bundle = bundle["tasks"][task]
    model = task_bundle["model"]
    label_map = task_bundle["label_map"]

    prediction = int(model.predict([cleaned_text])[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([cleaned_text])[0]))

    return {
        "task": task,
        "label": label_map[prediction],
        "label_id": prediction,
        "confidence": confidence,
    }


def predict_selected_tasks(task: str, text: str) -> list[dict]:
    """Predict one or all bundled TweetEval tasks for text."""
    normalized_task = task.strip().lower()
    if normalized_task == "all":
        selected_tasks = TASKS
    else:
        selected_tasks = (validate_task(normalized_task),)

    bundle = load_bundle()
    return [predict(selected_task, text, bundle) for selected_task in selected_tasks]


def print_evaluation(results: list[dict]) -> None:
    """Print evaluation metrics for CLI usage."""
    for result in results:
        print(f"\nTask: {result['task']}")
        print(f"accuracy: {result['accuracy']:.4f}")
        print(f"macro F1: {result['macro_f1']:.4f}")
        print(result["classification_report"])


def main() -> None:
    """Train, evaluate, or predict with the TweetEval-only bundle."""
    parser = argparse.ArgumentParser(description="TweetEval-only model bundle tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train sentiment and irony models.")
    subparsers.add_parser("evaluate", help="Evaluate sentiment and irony models.")

    predict_parser = subparsers.add_parser("predict", help="Predict a TweetEval task.")
    predict_parser.add_argument(
        "text",
        help="Text to classify.",
    )
    predict_parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="all",
        help="TweetEval task to use. Defaults to all.",
    )

    args = parser.parse_args()

    try:
        if args.command == "train":
            train_bundle()
            print("done")
        elif args.command == "evaluate":
            print_evaluation(evaluate_bundle())
        elif args.command == "predict":
            results = predict_selected_tasks(args.task, args.text)
            for result in results:
                if result["task"] == "sentiment":
                    print("Casual sentiment")
                elif result["task"] == "irony":
                    print("Irony/sarcasm")
                else:
                    print(f"Task: {result['task']}")

                print(f"Label: {result['label']}")
                if result["confidence"] is not None:
                    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    except (FileNotFoundError, TweetEvalError, ValueError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
