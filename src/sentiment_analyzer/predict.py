import argparse
from pathlib import Path

import joblib


MODEL_PATH = Path("models/sentiment_model.joblib")


def load_model():
    """Load the trained sentiment model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found at models/sentiment_model.joblib.\n"
            "Please train the model first by running:\n"
            "python -m sentiment_analyzer.train"
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str) -> dict:
    """Predict whether the provided text is positive or negative."""
    cleaned_text = text.strip() if isinstance(text, str) else ""

    if not cleaned_text:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "error": "Input text is empty.",
        }

    model = load_model()
    prediction = model.predict([cleaned_text])[0]

    label = "Positive" if int(prediction) == 1 else "Negative"
    confidence = 1.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probabilities))

    return {
        "label": label,
        "confidence": confidence,
    }


def main() -> None:
    """Run sentiment prediction from the command line."""
    parser = argparse.ArgumentParser(description="Predict sentiment for text.")
    parser.add_argument("text", help="Text to analyze")
    args = parser.parse_args()

    try:
        result = predict_sentiment(args.text)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return

    confidence_percent = result["confidence"] * 100
    print(f"Label: {result['label']}")
    print(f"Confidence: {confidence_percent:.2f}%")


if __name__ == "__main__":
    main()
