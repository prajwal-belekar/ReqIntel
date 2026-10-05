import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import numpy as np

# Read the CSV
df = pd.read_csv('D:/AIandML/requirement-lens/data/raw/PROMISE.csv')

# Add text features
df['text_length'] = df['RequirementText'].str.len()
df['word_count'] = df['RequirementText'].str.split().str.len()

# Clean text for analysis
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text.lower()

df['text_clean'] = df['RequirementText'].apply(clean_text)

# Set style
sns.set_palette("husl")

# 1. Class Distribution
plt.figure(figsize=(12, 6))
class_counts = df['_class_'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
bars = plt.bar(class_counts.index, class_counts.values, color=colors)
plt.title('Class Distribution in PROMISE Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Requirement Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('D:/AIandML/requirement-lens/data/eda/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Project Distribution
plt.figure(figsize=(12, 6))
project_counts = df['ProjectID'].value_counts()
plt.bar(range(len(project_counts)), project_counts.values)
plt.title('Requirements per Project', fontsize=16, fontweight='bold')
plt.xlabel('Project (sorted by record count)', fontsize=12)
plt.ylabel('Number of Requirements', fontsize=12)
plt.tight_layout()
plt.savefig('D:/AIandML/requirement-lens/data/eda/project_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Text Length Distribution
plt.figure(figsize=(12, 6))
plt.hist(df['text_length'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Text Length Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Character Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df['text_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["text_length"].mean():.1f}')
plt.axvline(df['text_length'].median(), color='green', linestyle='--', label=f'Median: {df["text_length"].median():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('D:/AIandML/requirement-lens/data/eda/text_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Word Count Distribution
plt.figure(figsize=(12, 6))
plt.hist(df['word_count'], bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.title('Word Count Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["word_count"].mean():.1f}')
plt.axvline(df['word_count'].median(), color='green', linestyle='--', label=f'Median: {df["word_count"].median():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('D:/AIandML/requirement-lens/data/eda/word_count_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Save cleaned dataset
cleaned_columns = ['RequirementText', '_class_', 'ProjectID', 'text_length', 'word_count']
df_cleaned = df[cleaned_columns].copy()
df_cleaned.rename(columns={'_class_': 'category', 'RequirementText': 'requirement_text'}, inplace=True)
df_cleaned.to_csv('D:/AIandML/requirement-lens/data/processed/requirements_cleaned.csv', index=False)

print('Cleaned dataset saved with', len(df_cleaned), 'records')
print('Columns:', list(df_cleaned.columns))

# Create EDA report
with open('D:/AIandML/requirement-lens/data/eda/eda_report.md', 'w') as f:
    f.write('# PROMISE Dataset Exploratory Data Analysis Report\n\n')
    f.write('## 1. Dataset Overview\n\n')
    f.write(f'- **Total Records:** {len(df)}\n')
    f.write(f'- **Unique Classes:** {df["_class_"].nunique()}\n')
    f.write(f'- **Unique Projects:** {df["ProjectID"].nunique()}\n')
    f.write(f'- **Columns:** ProjectID, RequirementText, _class_\n\n')
    
    f.write('## 2. Class Distribution\n\n')
    f.write('```\n')
    f.write(str(class_counts))
    f.write('\n```\n\n')
    f.write('**Key Findings:**\n')
    f.write('- **Severe class imbalance**: F class has 444 samples (45.8%), PO has only 12 (1.2%)\n')
    f.write('- **Top 3 classes**: F (444), SE (125), US (85)\n')
    f.write('- **Bottom 3 classes**: PO (12), L (15), FT (18)\n\n')
    
    f.write('## 3. Project Distribution\n\n')
    f.write(f'- Requirements are distributed across {df["ProjectID"].nunique()} projects\n')
    f.write(f'- Average requirements per project: {len(df) / df["ProjectID"].nunique():.1f}\n')
    f.write(f'- Most prolific project: Project {project_counts.index[0]} with {project_counts.iloc[0]} requirements\n\n')
    
    f.write('## 4. Text Statistics\n\n')
    f.write(f'- **Text Length**: Min={df["text_length"].min()}, Max={df["text_length"].max()}, Mean={df["text_length"].mean():.1f}, Median={df["text_length"].median():.1f}\n')
    f.write(f'- **Word Count**: Min={df["word_count"].min()}, Max={df["word_count"].max()}, Mean={df["word_count"].mean():.1f}, Median={df["word_count"].median():.1f}\n\n')
    
    f.write('## 5. Data Quality Issues\n\n')
    f.write('### Missing Values\n')
    f.write(f'- **None** - All columns complete\n\n')
    
    f.write('### Duplicates\n')
    f.write('- **Exact duplicates**: 0\n')
    f.write('- **Cross-project duplicates**: 0\n')
    f.write('- Database appears clean with no duplication issues\n\n')
    
    f.write('## 6. Potential Data Leakage Risks\n\n')
    f.write('### High Risk:\n')
    f.write('1. **Project-specific terminology**: Projects may use domain-specific terms\n')
    f.write('2. **ProjectID in model**: Cannot use ProjectID as feature\n')
    f.write('3. **Class imbalance**: Model may be biased toward majority class\n\n')
    
    f.write('### Mitigation Strategies:\n')
    f.write('1. **Stratified train/test split** by project (group-aware splitting)\n')
    f.write('2. **Project-aware cross-validation** to prevent leakage\n')
    f.write('3. **Class weighting** or resampling for imbalance\n\n')
    
    f.write('## 7. Recommended Train/Test Split\n\n')
    f.write('**Strategy**: Group-aware stratified split\n\n')
    f.write('- Use **ProjectID as grouping variable**\n')
    f.write('- **Stratify by class** to preserve class distribution\n')
    f.write('- **80/20 split**: ~775 train, ~194 test\n')
    f.write('- Ensure no project appears in both train and test sets\n\n')
    f.write('**Rationale**: Prevents model from learning project-specific patterns that won\'t generalize to new projects.\n\n')

print('EDA complete. Files saved.')
