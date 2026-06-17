import streamlit as st

from sentiment_analyzer.predict import predict_sentiment


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis Tool", page_icon=":bar_chart:")
    st.title("Sentiment Analysis Tool")

    user_text = st.text_area("Text to analyze", placeholder="Enter a sentence or short review.")

    if st.button("Analyze"):
        result = predict_sentiment(user_text)
        st.write("Label:", result["label"])
        st.write("Confidence:", result["confidence"])
        st.info(result["message"])


if __name__ == "__main__":
    main()
