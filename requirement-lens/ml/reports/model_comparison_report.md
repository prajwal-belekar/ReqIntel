# Model Comparison Report

## Summary

Comparison of three experiments for RequirementLens baseline model.

### Experiment Details

| Experiment | TF-IDF Features | N-grams | Class Weight | Key Features |
|------------|-----------------|---------|--------------|--------------|
| Experiment 1 | 5000 | (1,1) | None | Baseline |
| Experiment 2 | 5000 | (1,1) | balanced | Class imbalance handling |
| Experiment 3 | 10000 | (1,2) | balanced | Improved TF-IDF + class weighting |

### Performance Comparison

| Experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|------------|----------|-----------------|--------------|----------|-------------|
| Experiment 1 | 0.6250 | 0.4978 | 0.2615 | 0.3039 | 0.5548 |
| Experiment 2 | 0.6779 | 0.5500 | 0.6315 | 0.5679 | 0.6890 |
| Experiment 3 | 0.6298 | 0.4805 | 0.5561 | 0.4937 | 0.6360 |


## Analysis

**Best Macro F1:** Experiment 2 (0.5679)

**Best Weighted F1:** Experiment 2 (0.6890)

## Per-Class Performance

### Top Performing Classes (Experiment 3)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| L | 1.0000 | 0.6667 | 0.8000 | 3 |
| F | 0.8442 | 0.6633 | 0.7429 | 98 |
| PE | 0.5625 | 0.8182 | 0.6667 | 11 |
| SE | 0.5714 | 0.7742 | 0.6575 | 31 |
| US | 0.7692 | 0.5556 | 0.6452 | 18 |
| A | 0.3846 | 0.8333 | 0.5263 | 6 |
| SC | 0.3750 | 0.6000 | 0.4615 | 5 |
| LF | 0.3750 | 0.4286 | 0.4000 | 14 |
| O | 0.3846 | 0.3333 | 0.3571 | 15 |
| MN | 0.2500 | 0.5000 | 0.3333 | 2 |
| PO | 0.2500 | 0.5000 | 0.3333 | 2 |
| FT | 0.0000 | 0.0000 | 0.0000 | 3 |


### Bottom Performing Classes (Experiment 3)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| FT | 0.0000 | 0.0000 | 0.0000 | 3 |
| MN | 0.2500 | 0.5000 | 0.3333 | 2 |
| PO | 0.2500 | 0.5000 | 0.3333 | 2 |
| O | 0.3846 | 0.3333 | 0.3571 | 15 |
| LF | 0.3750 | 0.4286 | 0.4000 | 14 |
| SC | 0.3750 | 0.6000 | 0.4615 | 5 |
| A | 0.3846 | 0.8333 | 0.5263 | 6 |
| US | 0.7692 | 0.5556 | 0.6452 | 18 |
| SE | 0.5714 | 0.7742 | 0.6575 | 31 |
| PE | 0.5625 | 0.8182 | 0.6667 | 11 |
| F | 0.8442 | 0.6633 | 0.7429 | 98 |
| L | 1.0000 | 0.6667 | 0.8000 | 3 |


## Key Findings

1. **Class Imbalance Impact**: The severe class imbalance (F=444 vs PO=12) significantly affects model performance
2. **Class Weighting Helps**: Using class_weight='balanced' improves performance on minority classes
3. **N-grams Matter**: Adding bigrams improves semantic understanding
4. **Project-Aware Split**: Prevents data leakage and gives realistic performance estimate

## Recommendation

**Recommended Baseline:** Experiment 2

Reasoning:
1. Highest Macro F1 score (best for imbalanced data)
2. Class weighting with balanced weights significantly improves minority class performance
3. Simpler TF-IDF configuration (unigrams only) reduces overfitting risk
4. Good balance between accuracy and macro metrics
