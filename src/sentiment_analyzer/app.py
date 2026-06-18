import streamlit as st

from sentiment_analyzer.predict import predict_sentiment


EXAMPLES = [
    "This movie was amazing and emotional.",
    "This was boring, slow, and terrible.",
]


def show_prediction(label: str, confidence: float) -> None:
    """Render the prediction in a compact, friendly layout."""
    confidence_percent = confidence * 100

    st.subheader("Result")
    left_column, right_column = st.columns(2)

    with left_column:
        st.metric("Predicted label", label)

    with right_column:
        st.metric("Confidence", f"{confidence_percent:.2f}%")

    message = f"{label} sentiment detected with {confidence_percent:.2f}% confidence."
    if label == "Positive":
        st.success(message)
    else:
        st.warning(message)


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis Tool", page_icon=":bar_chart:")

    st.title("Sentiment Analysis Tool")
    st.write(
        "Analyze whether a movie review or short English text sounds positive or negative."
    )

    text = st.text_area(
        "Enter text",
        placeholder="Type or paste a short English movie review here.",
        height=160,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text before analyzing sentiment.")
        else:
            try:
                result = predict_sentiment(text)
            except FileNotFoundError:
                st.error("Model not found. Run python -m sentiment_analyzer.train first.")
            else:
                show_prediction(result["label"], result["confidence"])

    st.divider()
    st.subheader("Examples")
    for example in EXAMPLES:
        st.code(example, language="text")


if __name__ == "__main__":
    main()
