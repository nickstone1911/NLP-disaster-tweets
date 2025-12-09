NLP Disaster Tweets — RNN Classification

This repository contains my end-to-end project for the Kaggle competition
“Real or Not? Disaster Tweets”
https://www.kaggle.com/competitions/nlp-getting-started

The goal is to classify whether a tweet refers to a real disaster (1) or not (0) using recurrent neural networks (LSTM / Bidirectional LSTM / GRU).

Project Overview

This project was built as a course assignment and includes:

Problem & Data Description

Binary text classification on tweets.

Training data: ~7,600 labeled tweets.

Test data: ~3,200 unlabeled tweets.

Columns:

id: tweet ID

keyword: optional keyword (often missing)

location: noisy user location text

text: the tweet content

target: 1 = disaster, 0 = not disaster (train only)

Exploratory Data Analysis (EDA)

Class balance visualization (disaster vs non-disaster).

Tweet length distribution (character-based).

Missing keyword/location inspection.

Example tweets for each class.

Preprocessing

Lowercasing.

Removing URLs, mentions, hashtags, punctuation, digits.

Tokenization of cleaned text.

Padding sequences to a fixed length.

Vocabulary size: 20,000 most frequent words.

Sequence length: 120 tokens.

Models

Baseline LSTM.

Bidirectional LSTM with dropout.

Bidirectional GRU variant.

Evaluation & Kaggle Submission

Validation accuracy comparison across models.

Final submission CSV for Kaggle leaderboard.

Repository Structure
nlp-disaster-tweets/
│
├── data/
│   ├── raw/               # Original train.csv, test.csv
│   └── processed/         # submission.csv, any cleaned/derived data
│
├── notebooks/
│   └── 01_disaster_tweets_rnn.ipynb   # Main Jupyter notebook
│
├── src/
│   ├── preprocessing.py   # (optional) text cleaning & tokenization helpers
│   ├── models.py          # (optional) model definitions
│   └── utils.py           # (optional) utilities
│
├── reports/
│   ├── figures/           # EDA and training plots
│   └── final-report.md    # Written summary / report (if used)
│
├── requirements.txt
└── README.md


Note: data/ and venv/ should not be pushed to GitHub. Use .gitignore to exclude them.

Setup
1. Clone the repository
git clone https://github.com/<your-username>/nlp-disaster-tweets.git
cd nlp-disaster-tweets

2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

3. Install dependencies

If requirements.txt is present:

pip install -r requirements.txt


If not, the main libraries used are:

pandas

numpy

matplotlib

seaborn

scikit-learn

tensorflow (Keras)

You can install them with:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

How to Run the Notebook

Make sure your virtual environment is activated.

Start Jupyter:

jupyter notebook


Open:

notebooks/01_disaster_tweets_rnn.ipynb


Run the cells in order:

Load data (train.csv, test.csv from data/raw/).

EDA.

Text cleaning & tokenization.

Model training (LSTM / BiLSTM / GRU).

Kaggle submission generation.

Model Overview
Baseline Model — Simple LSTM

Architecture:
Embedding → LSTM(64) → Dropout → Dense(32) → Dense(1, sigmoid)

Purpose: establish a simple sequential baseline.

Result: validation accuracy around 0.5–0.6 (close to random).

Improved Model — Bidirectional LSTM

Architecture (example):
Embedding → SpatialDropout1D → Bidirectional LSTM(64) → Dropout → Dense(64) → Dense(1, sigmoid)

Motivation:

Capture context from both left-to-right and right-to-left in the tweet.

Regularize embedding space with dropout.

Increase model capacity.

Result: validation accuracy around 0.75–0.78 (much better performance).

GRU Variant

Swaps LSTM units for GRU units in a Bidirectional layer.

GRUs are a simpler gating mechanism with fewer parameters.

Achieves similar performance to BiLSTM in many cases.

Kaggle Submission

The final model generates predictions on test.csv:

Clean and tokenize test['text'] using the same preprocessing as training.

Convert to padded sequences X_test.

Predict probabilities:

test_pred_probs = model_bilstm.predict(X_test)


Threshold at 0.5:

test_pred = (test_pred_probs >= 0.5).astype(int).reshape(-1)


Build submission DataFrame:

submission = pd.DataFrame({
    "id": test["id"],
    "target": test_pred
})


Save to CSV:

submission.to_csv("submission.csv", index=False)


Upload submission.csv on the Kaggle competition “Submit Predictions” page.

The required format is:

id,target
0,1
2,0
3,1
...

Future Work

Use pretrained word embeddings (e.g., GloVe) instead of training embeddings from scratch.

Try CNN + RNN hybrids for feature extraction on tweet text.

Fine-tune transformer models like BERT or DistilBERT for higher accuracy.

Perform more systematic hyperparameter tuning (learning rate, batch size, number of units, dropout rates).

References

Kaggle Competition: Real or Not? Disaster Tweets
https://www.kaggle.com/competitions/nlp-getting-started

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.

Chung, J. et al. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.

Keras / TensorFlow documentation: https://keras.io