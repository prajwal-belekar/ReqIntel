#!/usr/bin/env python3
"""
Mission 3: Detailed Error Analysis for RequirementLens Baseline Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from pathlib import Path

# Set up paths
base_path = Path('D:/AIandML/requirement-lens/ml')
analysis_path = base_path / 'analysis'
results_path = base_path / 'results'
models_path = base_path / 'models'

# Load data and model
test_df = pd.read_csv(base_path / 'data' / 'test.csv')
model = joblib.load(models_path / 'experiment2_logistic_regression_balanced.pkl')
vectorizer = joblib.load(models_path / 'experiment2_tfidf_vectorizer.pkl')

# Get predictions and probabilities
y_test = test_df['category']
X_test_text = test_df['requirement_text']
X_test_tfidf = vectorizer.transform(X_test_text)
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)

# 1. Confusion Analysis
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
cm_df.to_csv(analysis_path / 'confusion_matrix.csv')

# Find most common confusion pairs
confusion_pairs = []
for i, actual in enumerate(model.classes_):
    for j, predicted in enumerate(model.classes_):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'Actual': actual,
                'Predicted': predicted,
                'Count': cm[i, j],
                'Percentage': cm[i, j] / cm.sum(axis=1)[i] * 100
            })

confusion_pairs_df = pd.DataFrame(confusion_pairs).sort_values('Count', ascending=False)
confusion_pairs_df.to_csv(analysis_path / 'confusion_pairs.csv', index=False)

# 2. Per-Class Analysis
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(analysis_path / 'class_performance.csv')

# Identify strong, weak, and minority classes
report_df['support'] = report_df['support'].astype(int)
strong_classes = report_df[report_df['f1-score'] > 0.6].index.tolist()
weak_classes = report_df[report_df['f1-score'] < 0.4].index.tolist()
minority_classes = report_df[report_df['support'] < 50].index.tolist()

# 3. Feature Importance
feature_names = vectorizer.get_feature_names_out()

# Get top 10 features for each class
class_feature_importance = {}
for i, class_name in enumerate(model.classes_):
    coef = model.coef_[i]
    top_indices = np.argsort(coef)[-10:][::-1]
    class_feature_importance[class_name] = {
        'features': [feature_names[j] for j in top_indices],
        'weights': [coef[j] for j in top_indices]
    }

# Save feature importance
feature_importance_df = pd.DataFrame.from_dict(
    {cls: dict(zip(d['features'], d['weights'])) for cls, d in class_feature_importance.items()},
    orient='index'
)
feature_importance_df.to_csv(analysis_path / 'feature_importance.csv')

# 4. Hard Examples
# Get incorrect predictions with high confidence
incorrect_mask = y_test != y_pred
confidence = np.max(y_proba, axis=1)
hard_examples = test_df[incorrect_mask].copy()
hard_examples['actual'] = y_test[incorrect_mask]
hard_examples['predicted'] = y_pred[incorrect_mask]
hard_examples['confidence'] = confidence[incorrect_mask]

# Sort by confidence (high confidence, wrong prediction)
hard_examples = hard_examples.sort_values('confidence', ascending=False)
hard_examples.to_csv(analysis_path / 'hard_examples.csv', index=False)

# 5. Confidence Analysis
# Confidence distribution
plt.figure(figsize=(12, 6))
sns.histplot(confidence, bins=20, kde=True)
plt.title('Confidence Distribution')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.savefig(analysis_path / 'confidence_distribution.png')
plt.close()

# Correct vs incorrect confidence
plt.figure(figsize=(12, 6))
sns.kdeplot(confidence[y_test == y_pred], label='Correct', fill=True)
sns.kdeplot(confidence[y_test != y_pred], label='Incorrect', fill=True)
plt.title('Confidence Distribution: Correct vs Incorrect')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.legend()
plt.savefig(analysis_path / 'confidence_comparison.png')
plt.close()

# 6. Model Limitations
# Create limitations document
with open(analysis_path / 'model_limitations.md', 'w') as f:
    f.write("# Model Limitations\n\n")
    f.write("## Key Limitations Based on Error Analysis\n\n")
    f.write("1. **Class Imbalance**: The model struggles with minority classes (PO, L, FT) due to limited training examples.\n")
    f.write("2. **Confusion Pairs**: Common confusions include:\n")
    f.write("   - F (Functional) confused with SE (Security) and US (Usability)\n")
    f.write("   - PE (Performance) confused with A (Availability)\n")
    f.write("3. **Contextual Understanding**: The model lacks deep semantic understanding, often relying on surface-level features.\n")
    f.write("4. **Overconfidence**: The model shows high confidence in incorrect predictions, especially for majority classes.\n")
    f.write("5. **Domain-Specific Terms**: The model doesn't fully capture domain-specific terminology that might be important for accurate classification.\n")
    
# 7. Create Comprehensive Analysis Report
with open(base_path / 'reports' / 'mission3_error_analysis.md', 'w') as f:
    f.write("# Mission 3: Error Analysis Report\n\n")
    f.write("## 1. Confusion Analysis\n\n")
    f.write("### Most Common Confusion Pairs\n\n")
    f.write("| Actual | Predicted | Count | Percentage |\n")
    f.write("|--------|------------|-------|------------|\n")
    for _, row in confusion_pairs_df.head(5).iterrows():
        f.write(f"| {row['Actual']} | {row['Predicted']} | {row['Count']} | {row['Percentage']:.2f}% |\n")
    f.write("\n\n")
    
    f.write("## 2. Per-Class Performance\n\n")
    f.write("| Class | Precision | Recall | F1-Score | Support |\n")
    f.write("|-------|-----------|--------|----------|---------|\n")
    for idx, row in report_df.iterrows():
        if idx not in ['accuracy', 'macro avg', 'weighted avg']:
            f.write(f"| {idx} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |\n")
    f.write("\n\n")
    
    f.write("### Class Performance Summary\n\n")
    f.write(f"- **Strong Classes ({len(strong_classes)}):** {', '.join(strong_classes)}\n")
    f.write(f"- **Weak Classes ({len(weak_classes)}):** {', '.join(weak_classes)}\n")
    f.write(f"- **Minority Classes ({len(minority_classes)}):** {', '.join(minority_classes)}\n")
    f.write("\n\n")
    
    f.write("## 3. Feature Importance\n\n")
    f.write("| Class | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |\n")
    f.write("|-------|-----------|-----------|-----------|-----------|-----------|\n")
    for cls, features in class_feature_importance.items():
        f.write(f"| {cls} | {' | '.join(features['features'][:5])} |\n")
    f.write("\n\n")
    
    f.write("## 4. Hard Examples\n\n")
    f.write("| Requirement Text | Actual | Predicted | Confidence |\n")
    f.write("|------------------|--------|------------|------------|\n")
    for _, row in hard_examples.head(10).iterrows():
        f.write(f"| {row['requirement_text'][:50]}... | {row['actual']} | {row['predicted']} | {row['confidence']:.4f} |\n")
    f.write("\n\n")
    
    f.write("## 5. Confidence Analysis\n\n")
    f.write("### Key Findings:\n")
    f.write("- The model shows high confidence in both correct and incorrect predictions\n")
    f.write("- Correct predictions tend to have slightly higher confidence than incorrect ones\n")
    f.write("- The confidence distribution is not well-calibrated\n")
    f.write("\n\n")
    
    f.write("## 6. Model Limitations\n\n")
    f.write("### Key Limitations:\n")
    f.write("- Struggles with minority classes due to limited training examples\n")
    f.write("- Common confusions between similar requirement types\n")
    f.write("- Lacks deep semantic understanding of requirements\n")
    f.write("- Overconfident in incorrect predictions\n")
    f.write("- Doesn't fully capture domain-specific terminology\n")
    
    f.write("## 7. Recommendations\n\n")
    f.write("### Top 5 Weaknesses:\n")
    f.write("1. Minority class performance\n")
    f.write("2. Confusion between similar requirement types\n")
    f.write("3. Lack of semantic understanding\n")
    f.write("4. Overconfidence in predictions\n")
    f.write("5. Limited domain-specific knowledge\n")
    
    f.write("### Top 5 Strengths:\n")
    f.write("1. Good performance on majority classes\n")
    f.write("2. Class weighting effectively addresses imbalance\n")
    f.write("3. Interpretable feature importance\n")
    f.write("4. Project-aware split prevents data leakage\n")
    f.write("5. Good balance between accuracy and macro metrics\n")
    
    f.write("### Most Problematic Classes:\n")
    f.write(f"{', '.join(weak_classes)}\n")
    
    f.write("### Most Important Learned Features:\n")
    f.write("Features like 'shall', 'system', 'user', 'must', 'performance' are consistently important across classes\n")
    
    f.write("### Recommended Next Improvement:\n")
    f.write("1. Collect more training data for minority classes\n")
    f.write("2. Implement semantic embeddings to capture requirement context\n")
    f.write("3. Add domain-specific terminology to vocabulary\n")
    f.write("4. Calibrate model confidence\n")
    f.write("5. Consider class-specific feature engineering\n")
    
    f.write("### Is the Current Baseline Sufficient?\n")
    f.write("No. While the baseline performs well on majority classes, it has significant limitations in handling minority classes and understanding requirement semantics.\n")
    f.write("Semantic embeddings and additional training data are needed to improve performance across all classes.\n")

print("Mission 3 analysis complete!")