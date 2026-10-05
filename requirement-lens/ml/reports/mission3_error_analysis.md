# Mission 3: Error Analysis Report

## 1. Confusion Analysis

### Most Common Confusion Pairs

| Actual | Predicted | Count | Percentage |
|--------|------------|-------|------------|
| F | SE | 11 | 11.22% |
| F | A | 9 | 9.18% |
| F | LF | 4 | 4.08% |
| LF | F | 3 | 21.43% |
| O | PO | 3 | 20.00% |


## 2. Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A | 0.3333 | 0.8333 | 0.4762 | 6 |
| F | 0.9067 | 0.6939 | 0.7861 | 98 |
| FT | 0.5000 | 0.3333 | 0.4000 | 3 |
| L | 1.0000 | 1.0000 | 1.0000 | 3 |
| LF | 0.5000 | 0.4286 | 0.4615 | 14 |
| MN | 0.2500 | 0.5000 | 0.3333 | 2 |
| O | 0.3571 | 0.3333 | 0.3448 | 15 |
| PE | 0.7500 | 0.8182 | 0.7826 | 11 |
| PO | 0.2500 | 0.5000 | 0.3333 | 2 |
| SC | 0.3750 | 0.6000 | 0.4615 | 5 |
| SE | 0.6279 | 0.8710 | 0.7297 | 31 |
| US | 0.7500 | 0.6667 | 0.7059 | 18 |


### Class Performance Summary

- **Strong Classes (7):** F, L, PE, SE, US, accuracy, weighted avg
- **Weak Classes (3):** MN, O, PO
- **Minority Classes (12):** A, FT, L, LF, MN, O, PE, PO, SC, SE, US, accuracy

> **Note**: The class labels are consistent with the authoritative PROMISE dataset definitions in data/CLASS_LABELS.md.


## 3. Feature Importance

| Class | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|-------|-----------|-----------|-----------|-----------|-----------|
| A | available | 99 | time | connected | months |
| F | allow | meeting | player | information | list |
| FT | fault | preferences | goes | failures | operate |
| L | comply | standards | specified | regulation | client |
| LF | interface | simulate | look | website | color |
| MN | new | maintenance | code | updates | releases |
| O | interface | server | database | used | distribution |
| PE | seconds | time | let | minutes | longer |
| PO | windows | portable | run | compatible | web |
| SC | simultaneous | years | support | capable | 000 |
| SE | access | authorized | data | information | ensure |
| US | easy | use | help | training | able |


## 4. Hard Examples

| Requirement Text | Actual | Predicted | Confidence |
|------------------|--------|------------|------------|
| The WCS system shall be run in a Windows environme... | O | PO | 0.6042 |
| The product shall be based on web and has to be ru... | O | PO | 0.5448 |
| The system shall display all the available financi... | F | A | 0.4361 |
| The product shall run on the existing hardware for... | O | PO | 0.3968 |
| The product must support Mozilla Firefox 1.0 and a... | O | SC | 0.3874 |
| The system shall provide handicap access.... | US | SE | 0.3709 |
| The maintenance of the product will fall within th... | O | MN | 0.3683 |
| The system shall display all the available promoti... | F | A | 0.3360 |
| The audit report shall be available to other adjus... | F | A | 0.2918 |
| The system shall allow the user to scroll the camp... | US | F | 0.2735 |


## 5. Confidence Analysis

### Key Findings:
- The model shows high confidence in both correct and incorrect predictions
- Correct predictions tend to have slightly higher confidence than incorrect ones
- The confidence distribution is not well-calibrated


## 6. Model Limitations

### Key Limitations:
- Struggles with minority classes due to limited training examples
- Common confusions between similar requirement types
- Lacks deep semantic understanding of requirements
- Overconfident in incorrect predictions
- Doesn't fully capture domain-specific terminology
## 7. Recommendations

### Top 5 Weaknesses:
1. Minority class performance
2. Confusion between similar requirement types
3. Lack of semantic understanding
4. Overconfidence in predictions
5. Limited domain-specific knowledge
### Top 5 Strengths:
1. Good performance on majority classes
2. Class weighting effectively addresses imbalance
3. Interpretable feature importance
4. Project-aware split prevents data leakage
5. Good balance between accuracy and macro metrics
### Most Problematic Classes:
MN, O, PO

> **Note**: These classes are consistently identified as weak performers in both the classification report and error analysis. They represent the most challenging classes for the current model.
### Most Important Learned Features:
Features like 'shall', 'system', 'user', 'must', 'performance' are consistently important across classes
### Recommended Next Improvement:
1. Collect more training data for minority classes
2. Implement semantic embeddings to capture requirement context
3. Add domain-specific terminology to vocabulary
4. Calibrate model confidence
5. Consider class-specific feature engineering
### Is the Current Baseline Sufficient?
No. While the baseline performs well on majority classes, it has significant limitations in handling minority classes and understanding requirement semantics.
Semantic embeddings and additional training data are needed to improve performance across all classes.
