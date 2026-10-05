# PROMISE Dataset Exploratory Data Analysis Report

## 1. Dataset Overview

- **Total Records:** 969
- **Unique Classes:** 12
- **Unique Projects:** 47
- **Columns:** ProjectID, RequirementText, _class_

## 2. Class Distribution

```
_class_
F     444
SE    125
US     85
O      77
PE     67
LF     49
A      31
MN     24
SC     22
FT     18
L      15
PO     12
Name: count, dtype: int64
```

**Key Findings:**
- **Severe class imbalance**: F class has 444 samples (45.8%), PO has only 12 (1.2%)
- **Top 3 classes**: F (444), SE (125), US (85)
- **Bottom 3 classes**: PO (12), L (15), FT (18)

## 3. Project Distribution

- Requirements are distributed across 47 projects
- Average requirements per project: 20.6
- Most prolific project: Project 8 with 92 requirements

## 4. Text Statistics

- **Text Length**: Min=28, Max=546, Mean=109.2, Median=94.0
- **Word Count**: Min=4, Max=86, Mean=17.9, Median=15.0

## 5. Data Quality Issues

### Missing Values
- **None** - All columns complete

### Duplicates
- **Exact duplicates**: 0
- **Cross-project duplicates**: 0
- Database appears clean with no duplication issues

## 6. Potential Data Leakage Risks

### High Risk:
1. **Project-specific terminology**: Projects may use domain-specific terms
2. **ProjectID in model**: Cannot use ProjectID as feature
3. **Class imbalance**: Model may be biased toward majority class

### Mitigation Strategies:
1. **Stratified train/test split** by project (group-aware splitting)
2. **Project-aware cross-validation** to prevent leakage
3. **Class weighting** or resampling for imbalance

## 7. Recommended Train/Test Split

**Strategy**: Group-aware stratified split

- Use **ProjectID as grouping variable**
- **Stratify by class** to preserve class distribution
- **80/20 split**: ~775 train, ~194 test
- Ensure no project appears in both train and test sets

**Rationale**: Prevents model from learning project-specific patterns that won't generalize to new projects.

