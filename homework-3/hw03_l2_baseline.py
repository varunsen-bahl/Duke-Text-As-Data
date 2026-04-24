"""
IDS 570: Text as Data
Homework 3

This file contains the pipeline for the baseline model (logistic regression with L2)

Author: Varun Sen Bahl

"""

## load packages
from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer

## load data directory
DATA_DIR = Path("data")

## loading the datasets
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

## separate texts and labels
X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

## vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,  # ignore very rare words
    max_df=0.9,  # ignore extremely common words; Explanation [B]
)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

## logistic regression

clf = LogisticRegression(max_iter=1000, n_jobs=1)
clf.fit(X_train, y_train)

# test set predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

## ROC AUC
auc = roc_auc_score(y_test, y_prob)

# top 15 positive and negative words
features = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
top_pos = np.argsort(coefs)[-15:][::-1]
top_neg = np.argsort(coefs)[:15]

# saving outputs
with open("l2_baseline_outputs.txt", "w") as f:

    ## Confusion matrix
    print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}", file=f)

    ## Classification report
    print(f"\nClassification report: {classification_report(y_test, y_pred)}", file=f)

    ## ROC AUC
    print(f"\nROC AUC: {round(auc, 3)}", file=f)

    ## Sparsity diagnostic
    print(f"\nNon-zero coefficients: {np.count_nonzero(clf.coef_[0])}", file=f)

    print("\nTop 15 positive words (CORE):", file=f)
    for i in top_pos:
        print(f"    {features[i]:20s} {coefs[i]:.4f}", file=f)

    print("\nTop 15 negative words (NEG):", file=f)
    for i in top_neg:
        print(f"    {features[i]:20s} {coefs[i]:.4f}", file=f)
