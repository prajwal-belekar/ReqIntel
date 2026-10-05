#!/usr/bin/env python3
"""
Project-aware stratified train/test split for PROMISE dataset.

This script creates a train/test split that:
1. Preserves class distribution
2. Ensures no project appears in both train and test sets
3. Prevents data leakage from project-specific patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Load the cleaned dataset
df = pd.read_csv('D:/AIandML/requirement-lens/data/processed/requirements_cleaned.csv')

# Define classes and projects
df['class_label'] = df['category']  # Use the class column

# Create project-aware stratified split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the split indices
for train_index, test_index in gss.split(df, df['class_label'], groups=df['ProjectID']):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    # Verify no project appears in both sets
    train_projects = set(train_df['ProjectID'])
    test_projects = set(test_df['ProjectID'])
    
    if train_projects.intersection(test_projects):
        raise ValueError("Error: Project overlap between train and test sets")
    
    # Verify class distribution is preserved
    train_class_dist = train_df['class_label'].value_counts(normalize=True).sort_index()
    test_class_dist = test_df['class_label'].value_counts(normalize=True).sort_index()
    
    # Align the distributions by index
    all_classes = sorted(set(train_class_dist.index) | set(test_class_dist.index))
    train_aligned = pd.Series([train_class_dist.get(cls, 0) for cls in all_classes], index=all_classes)
    test_aligned = pd.Series([test_class_dist.get(cls, 0) for cls in all_classes], index=all_classes)
    
    max_diff = np.max(np.abs(train_aligned - test_aligned))
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Max class distribution difference: {max_diff:.4f}")
    print(f"Class distribution reasonably preserved: {max_diff < 0.1}")
    
    # Save the splits
    train_df.to_csv('D:/AIandML/requirement-lens/ml/data/train.csv', index=False)
    test_df.to_csv('D:/AIandML/requirement-lens/ml/data/test.csv', index=False)
    
    # Save metadata
    with open('D:/AIandML/requirement-lens\ml\data\split_metadata.txt', 'w') as f:
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Train records: {len(train_df)}\n")
        f.write(f"Test records: {len(test_df)}\n")
        f.write(f"Unique projects: {df['ProjectID'].nunique()}\n")
        f.write(f"Projects in train: {len(train_projects)}\n")
        f.write(f"Projects in test: {len(test_projects)}\n")
        f.write(f"Max class distribution difference: {max_diff:.4f}\n")
        f.write(f"Class distribution preserved: {max_diff < 0.1}")
    
    print("Train/test split created successfully!")