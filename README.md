# Amazon Reviews Sentiment Classification

This repository contains the implementation for: sentiment classification of Amazon product reviews using TF‑IDF and traditional ML models.

## Overview

The goal is to classify Amazon reviews into three sentiment classes:
- **Negative**
- **Neutral**
- **Positive**

The dataset (`amazon_reviews.csv`) includes over 17,000 records. The pipeline covers:
1. Reviews Pre‑processing (stop‑word removal & stemming)
2. Label Mapping & Train/Test Split
3. TF‑IDF Vectorization
4. Model Training & Evaluation (SVM, Logistic Regression, Naïve Bayes)
5. Interactive Prediction for New Reviews

## Features
### 1. Reviews Pre‑processing
- Tokenize and lowercase text

- Remove NLTK stop‑words

- Apply NLTK Porter stemmer

### 2. Label Mapping & Splitting
- Map labels: negative → 0, neutral → 1, positive → 2

- Split: 80% train / 20% test

### 3. TF‑IDF Vectorization
- Fit TfidfVectorizer on training set

- Transform both train and test reviews

### 4. Model Training & Evaluation
- Support Vector Machine (sklearn.svm.SVC)

- Logistic Regression (sklearn.linear_model.LogisticRegression)

- Multinomial Naïve Bayes (sklearn.naive_bayes.MultinomialNB)

- Print classification report for each model

### 5. Interactive Prediction
- Load best‑performing model & vectorizer

- Prompt user for a new review

- Output predicted sentiment label
