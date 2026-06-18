# Sentiment Analysis Tool

## Project Overview

Sentiment Analysis Tool is a beginner-friendly machine learning project that classifies movie review text as positive or negative. It trains a lightweight model on the Stanford IMDb dataset using a TF-IDF vectorizer and Logistic Regression classifier.

The project uses local dataset files stored in `data/raw/aclImdb`, saves the trained model locally to `models/sentiment_model.joblib`, and includes a Streamlit app for interactive predictions.

Dataset files and trained model files are not committed to GitHub.

## Features

- Train a sentiment analysis model from local IMDb review files.
- Use TF-IDF text features with Logistic Regression.
- Save and load the trained model with `joblib`.
- Predict sentiment from the command line.
- Evaluate the model with common classification metrics.
- Run an interactive Streamlit web app for text input and predictions.

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- joblib
- Streamlit
- pytest

## Project Structure

```text
.
|-- data/
|   |-- .gitkeep
|   `-- raw/
|       `-- aclImdb/              # Local dataset files, ignored by Git
|-- models/
|   |-- .gitkeep
|   `-- sentiment_model.joblib    # Generated locally, ignored by Git
|-- src/
|   `-- sentiment_analyzer/
|       |-- __init__.py
|       |-- app.py
|       |-- evaluate.py
|       |-- predict.py
|       `-- train.py
|-- tests/
|   `-- test_predict.py
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Dataset Setup

This project uses the Stanford IMDb dataset. The dataset is not included in the repository because it is a large external data file.

1. Download the Stanford IMDb dataset.
2. Extract the dataset into this path:

```text
data/raw/aclImdb
```

After extraction, the project should contain folders such as:

```text
data/raw/aclImdb/train/pos
data/raw/aclImdb/train/neg
data/raw/aclImdb/test/pos
data/raw/aclImdb/test/neg
```

The `data/raw/` directory is ignored by Git, so downloaded dataset files stay local and are not committed to GitHub.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies and the local package:

```powershell
pip install -r requirements.txt
pip install -e .
```

## Training the Model

Train the TF-IDF + Logistic Regression model:

```powershell
python -m sentiment_analyzer.train
```

The trained model is saved locally to:

```text
models/sentiment_model.joblib
```

The `models/` directory is used for generated model artifacts. The trained model file is not committed to GitHub.

## Running Predictions

Run a prediction from the command line:

```powershell
python -m sentiment_analyzer.predict "This movie was amazing"
```

The command loads the locally trained model and prints the predicted sentiment.

## Evaluating the Model

Evaluate the trained model:

```powershell
python -m sentiment_analyzer.evaluate
```

This reports accuracy, precision, recall, and F1 score on the evaluation data.

## Running the Streamlit App

Start the Streamlit app:

```powershell
streamlit run src/sentiment_analyzer/app.py
```

The app provides a simple interface for entering review text and viewing the predicted sentiment.

## Current Results

Current evaluation metrics:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8160 |
| Precision | 0.8129 |
| Recall | 0.8496 |
| F1 Score | 0.8309 |

## Limitations

- The model is trained on movie reviews, so it may not understand sarcasm or casual social media text perfectly.
- The model uses classical machine learning, not deep learning.
- The trained model file is generated locally and not included in the repository.

## Future Improvements

- Add a casual text sentiment dataset.
- Add a sarcasm/irony detector.
- Add emotion classification.
- Improve the UI with charts.
- Add screenshots.
