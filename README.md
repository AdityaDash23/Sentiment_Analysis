# Sentiment Analysis

This project demonstrates a sentiment analysis model built using Logistic Regression and TF-IDF vectorization. The model is trained to classify text into positive or negative sentiment.

## Table of Contents

  - [Introduction](https://www.google.com/search?q=%23introduction)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Preprocessing](https://www.google.com/search?q=%23preprocessing)
  - [Model](https://www.google.com/search?q=%23model)
  - [Results](https://www.google.com/search?q=%23results)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)

## Introduction

Sentiment analysis is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. This project focuses on binary sentiment classification (positive/negative) using a Logistic Regression model.

## Dataset

The dataset used is loaded from a CSV file named `Sentiment_Analysis.csv`. It contains tweets with associated sentiment polarity. The relevant columns are extracted and renamed to `polarity` and `text`.

  * `polarity`: Original sentiment label (0 for negative, 4 for positive).
  * `text`: The tweet content.

Neutral sentiment (polarity `2`) is removed from the dataset. The polarity values `0` and `4` are mapped to `0` and `1` respectively for binary classification, resulting in 800,000 negative and 800,000 positive samples.

## Preprocessing

1.  **Text Cleaning**: A `clean_text` function is applied to convert all text to lowercase.
2.  **Train-Test Split**: The dataset is split into training and testing sets with an 80/20 ratio, using `clean_text` as features and `polarity` as the target.
      * Training set size: 1,280,000 samples.
      * Testing set size: 320,000 samples.
3.  **TF-IDF Vectorization**: `TfidfVectorizer` is used to convert the text data into numerical feature vectors.
      * `max_features` is set to 5000, limiting the vocabulary size.
      * `ngram_range` is set to (1,2), considering both unigrams and bigrams.
      * The TF-IDF transformed training and testing data shapes are (1,280,000, 5000) and (320,000, 5000) respectively.

## Model

A Logistic Regression model is used for classification.

  * `max_iter` is set to 100 for the Logistic Regression solver.
  * The model is trained on the TF-IDF transformed training data (`X_train_tfidf`, `y_train`).

## Results

The trained Logistic Regression model's performance on the test set is as follows:

  * **Accuracy**: 0.79539375 (approximately 79.54%).
  * **Classification Report**:
      * **Precision (Class 0 - Negative)**: 0.80
      * **Recall (Class 0 - Negative)**: 0.78
      * **F1-score (Class 0 - Negative)**: 0.79
      * **Precision (Class 1 - Positive)**: 0.79
      * **Recall (Class 1 - Positive)**: 0.81
      * **F1-score (Class 1 - Positive)**: 0.80
      * **Macro Average F1-score**: 0.80
      * **Weighted Average F1-score**: 0.80

Sample predictions for new tweets:

  * "I love this\!" -\> Predicted: Positive (1)
  * "I hate that\!" -\> Predicted: Negative (0)
  * "It was okay, not great." -\> Predicted: Positive (1)

## Usage

To run this sentiment analysis model:

1.  Ensure you have the necessary dependencies installed.
2.  Place your `Sentiment_Analysis.csv` file in the same directory as the notebook.
3.  Execute the Jupyter Notebook cells sequentially.

## Dependencies

  * `pandas`
  * `scikit-learn` (sklearn)
