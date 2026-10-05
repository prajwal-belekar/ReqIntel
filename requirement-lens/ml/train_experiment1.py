#!/usr/bin/env python3
"""
Experiment 1: Baseline TF-IDF + Logistic Regression
"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

# Set up paths
base_path = Path('D:/AIandML/requirement-lens/ml')
data_path = base_path / 'data'
models_path = base_path / 'models'
results_path = base_path / 'results'
reports_path = base_path / 'reports'

# Load data
print("Loading data...")
train_df = pd.read_csv(data_path / 'train.csv')
test_df = pd.read_csv(data_path / 'test.csv')

X_train_text = train_df['requirement_text']
y_train = train_df['category']
X_test_text = test_df['requirement_text']
y_test = test_df['category']

print(f"Train samples: {len(X_train_text)}")
print(f"Test samples: {len(X_test_text)}")

# Experiment 1: Baseline TF-IDF + Logistic Regression
print("\n" + "="*60)
print("Experiment 1: Baseline TF-IDF + Logistic Regression")
print("="*60)

# TF-IDF Vectorizer
print("Creating TF-IDF vectorizer...")
start_time = time.time()
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 1),  # unigrams only for baseline
    min_df=2,
    max_df=0.9,
    stop_words='english',
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)
vectorizer_time = time.time() - start_time
print(f"Vectorizer created in {vectorizer_time:.2f}s")

# Train Logistic Regression
print("Training Logistic Regression...")
start_time = time.time()
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    multi_class='multinomial',
    random_state=42
)
model.fit(X_train_tfidf, y_train)
train_time = time.time() - start_time
print(f"Model trained in {train_time:.2f}s")

# Predict
print("Making predictions...")
start_time = time.time()
y_pred = model.predict(X_test_tfidf)
predict_time = time.time() - start_time
print(f"Predictions made in {predict_time:.4f}s")

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# Per-class metrics
print("\nPer-class metrics:")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(results_path / 'experiment1_classification_report.csv')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
cm_df.to_csv(results_path / 'experiment1_confusion_matrix.csv')

# Save model and vectorizer
print("\nSaving model...")
joblib.dump(model, models_path / 'experiment1_logistic_regression.pkl')
joblib.dump(vectorizer, models_path / 'experiment1_tfidf_vectorizer.pkl')

# Save results summary
with open(reports_path / 'experiment1_summary.txt', 'w') as f:
    f.write("Experiment 1: Baseline TF-IDF + Logistic Regression\n")
    f.write("="*60 + "\n\n")
    f.write(f"TF-IDF features: 5000\n")
    f.write(f"N-gram range: (1, 1)\n")
    f.write(f"Stop words: english\n")
    f.write(f"\nMetrics:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")
    f.write(f"\nTiming:\n")
    f.write(f"Vectorizer: {vectorizer_time:.2f}s\n")
    f.write(f"Training: {train_time:.2f}s\n")
    f.write(f"Prediction: {predict_time:.4f}s\n")

print("Experiment 1 complete!")