import pandas as pd
import numpy as np
from collections import Counter

# Read the CSV
df = pd.read_csv('D:/AIandML/requirement-lens/data/raw/PROMISE.csv')

print('='*60)
print('PROMISE DATASET - INITIAL INSPECTION')
print('='*60)
print(f'Total records: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Missing values per column:')
print(df.isnull().sum())
print()

# Class distribution
print('CLASS DISTRIBUTION:')
class_counts = df['_class_'].value_counts()
print(class_counts)
print(f'Number of unique classes: {df["_class_"].nunique()}')
print()

# Project distribution
print('PROJECT DISTRIBUTION:')
project_counts = df['ProjectID'].value_counts()
print(f'Number of unique projects: {df["ProjectID"].nunique()}')
print(f'Top 10 projects by record count:')
print(project_counts.head(10))
print()

# Text statistics
df['text_length'] = df['RequirementText'].str.len()
df['word_count'] = df['RequirementText'].str.split().str.len()

print('TEXT STATISTICS:')
print(f'Min length: {df["text_length"].min()}')
print(f'Max length: {df["text_length"].max()}')
print(f'Mean length: {df["text_length"].mean():.1f}')
print(f'Median length: {df["text_length"].median():.1f}')
print()
print(f'Min words: {df["word_count"].min()}')
print(f'Max words: {df["word_count"].max()}')
print(f'Mean words: {df["word_count"].mean():.1f}')
print(f'Median words: {df["word_count"].median():.1f}')
print()

# Duplicates
print('DUPLICATE ANALYSIS:')
exact_duplicates = df.duplicated(subset=['RequirementText']).sum()
print(f'Exact duplicate requirements: {exact_duplicates}')
print()

# Duplicates across projects
text_project_counts = df.groupby('RequirementText')['ProjectID'].nunique()
duplicate_across_projects = (text_project_counts > 1).sum()
print(f'Requirements appearing in multiple projects: {duplicate_across_projects}')
print()

# Save summary statistics to file
with open('D:/AIandML/requirement-lens/data/eda/statistics_summary.txt', 'w') as f:
    f.write('PROMISE DATASET STATISTICS SUMMARY\n')
    f.write('='*60 + '\n')
    f.write(f'Total records: {len(df)}\n')
    f.write(f'Unique classes: {df["_class_"].nunique()}\n')
    f.write(f'Unique projects: {df["ProjectID"].nunique()}\n')
    f.write('\nClass distribution:\n')
    f.write(str(class_counts) + '\n')
    f.write('\nText length stats:\n')
    f.write(f'Mean: {df["text_length"].mean():.1f}\n')
    f.write(f'Median: {df["text_length"].median():.1f}\n')
    f.write(f'Std: {df["text_length"].std():.1f}\n')
    
print('Statistics saved to eda/statistics_summary.txt')
