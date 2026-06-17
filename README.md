# Sentiment Analysis Tool

A lightweight Python project for a future sentiment analysis web app using scikit-learn and Streamlit.

## Planned Features

- Train a simple sentiment analysis model with scikit-learn.
- Save and load a trained model with joblib.
- Evaluate model performance on a test sample.
- Provide a small Streamlit interface for entering text and viewing predictions.

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- datasets
- joblib
- Streamlit
- matplotlib
- pytest

## Project Structure

```text
.
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   └── sentiment_analyzer/
│       ├── __init__.py
│       ├── app.py
│       ├── evaluate.py
│       ├── predict.py
│       └── train.py
├── tests/
│   └── test_predict.py
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run the Placeholder App

```bash
streamlit run src/sentiment_analyzer/app.py
```

No real API keys, secrets, data files, or trained model files are included.
