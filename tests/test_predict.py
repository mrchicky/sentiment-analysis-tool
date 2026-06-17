from sentiment_analyzer.predict import predict_sentiment


def test_predict_sentiment_empty_input_returns_safe_structure() -> None:
    result = predict_sentiment("")

    assert isinstance(result, dict)
    assert result["label"] == "unknown"
    assert result["confidence"] is None
    assert "message" in result
