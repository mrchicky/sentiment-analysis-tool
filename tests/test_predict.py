from sentiment_analyzer.predict import predict_sentiment


def test_predict_sentiment_empty_input_returns_safe_structure() -> None:
    result = predict_sentiment("")

    assert isinstance(result, dict)
    assert result["label"] == "Unknown"
    assert result["confidence"] == 0.0
    assert result["error"] == "Input text is empty."
