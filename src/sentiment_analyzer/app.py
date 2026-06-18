import streamlit as st

from sentiment_analyzer.predict import predict_sentiment
from sentiment_analyzer.tweeteval_bundle import predict_selected_tasks

try:
    from sentiment_analyzer.tweeteval_bundle import analyze_text
except ImportError:
    analyze_text = None


EXAMPLES = [
    "This movie was emotional and beautifully made.",
    "Yeah great, another error. Love that.",
    "I had a normal day and finished my work.",
]

MOVIE_MODE = "Movie Review Sentiment"
CASUAL_MODE = "Casual Text + Irony"


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


def show_metric_section(title: str, label: str, confidence: float | None) -> None:
    """Render one labeled result section."""
    st.subheader(title)
    left_column, right_column = st.columns(2)

    with left_column:
        st.metric("Label", label)

    with right_column:
        if confidence is None:
            st.metric("Confidence", "Not available")
        else:
            st.metric("Confidence", f"{confidence * 100:.2f}%")


def get_tweeteval_results(text: str) -> dict:
    """Analyze casual sentiment and irony using the TweetEval bundle."""
    if analyze_text is not None:
        results = analyze_text(text)
    else:
        results = predict_selected_tasks("all", text)

    if isinstance(results, list):
        return {result["task"]: result for result in results}

    return results


def get_task_result(results: dict, task: str) -> dict:
    """Return one TweetEval task result from common result shapes."""
    if task in results:
        return results[task]

    for key in (f"{task}_result", f"casual_{task}"):
        if key in results:
            return results[key]

    return {}


def show_irony_interpretation(label: str, confidence: float | None) -> None:
    """Explain the irony result in plain language."""
    if label == "Ironic" and confidence is not None and confidence >= 0.70:
        st.info("Likely ironic/sarcastic")
    elif label == "Ironic":
        st.info("Possibly ironic, but confidence is low")
    elif label == "Not Ironic":
        st.info("No strong irony detected")


def show_tweeteval_prediction(text: str) -> None:
    """Render casual sentiment and irony results."""
    results = get_tweeteval_results(text)
    sentiment = get_task_result(results, "sentiment")
    irony = get_task_result(results, "irony")

    show_metric_section(
        "Casual sentiment",
        sentiment.get("label", "Unknown"),
        sentiment.get("confidence"),
    )

    show_metric_section(
        "Irony/sarcasm",
        irony.get("label", "Unknown"),
        irony.get("confidence"),
    )
    show_irony_interpretation(irony.get("label", ""), irony.get("confidence"))


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis Tool", page_icon=":bar_chart:")

    st.title("Sentiment Analysis Tool")
    st.write("Analyze movie reviews, casual text sentiment, and possible irony/sarcasm.")

    analysis_mode = st.radio(
        "Analysis mode",
        [MOVIE_MODE, CASUAL_MODE],
        horizontal=True,
    )

    text = st.text_area(
        "Enter text",
        placeholder="Type or paste text to analyze.",
        height=160,
    )

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter some text before analyzing sentiment.")
        elif analysis_mode == MOVIE_MODE:
            try:
                result = predict_sentiment(text)
            except FileNotFoundError:
                st.error("IMDb model not found. Run python -m sentiment_analyzer.train first.")
            else:
                show_prediction(result["label"], result["confidence"])
        else:
            try:
                show_tweeteval_prediction(text)
            except FileNotFoundError:
                st.error(
                    "TweetEval bundle not found. Run python -m sentiment_analyzer.tweeteval_bundle train first."
                )

    st.divider()
    st.subheader("Examples")
    for example in EXAMPLES:
        st.code(example, language="text")


if __name__ == "__main__":
    main()
