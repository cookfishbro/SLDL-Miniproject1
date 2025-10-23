"""
Quick Start Script - Minimal example to get started

This script provides a simple baseline implementation that you can run immediately.
For full pipeline with model comparison and tuning, use train_model.py or main.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb
import re
import jieba
from utils import smape

print("="*80)
print("QUICK START - PRODUCT PRICE PREDICTION")
print("="*80)

# 1. Load data
print("\n1. Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

# 2. Simple preprocessing
print("\n2. Preprocessing...")

def simple_preprocess(text):
    """Quick text cleaning"""
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', ' ', text)
    return ' '.join(text.split())

def tokenize(text):
    """Simple tokenization with jieba"""
    return ' '.join(jieba.cut(text))

train_df['text'] = train_df['name'].apply(simple_preprocess).apply(tokenize)
test_df['text'] = test_df['name'].apply(simple_preprocess).apply(tokenize)

print("   ✓ Text cleaned and tokenized")

# 3. Create features
print("\n3. Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

X_train_full = tfidf.fit_transform(train_df['text'])
y_train_full = train_df['price']
X_test = tfidf.transform(test_df['text'])

print(f"   ✓ Feature shape: {X_train_full.shape}")

# 4. Train-validation split
print("\n4. Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)
print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# 5. Train baseline model (Ridge)
print("\n5. Training Ridge Regression (baseline)...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_val)
ridge_smape = smape(y_val, ridge_pred)
print(f"   ✓ Ridge Validation SMAPE: {ridge_smape:.2f}%")

# 6. Train LightGBM (better model)
print("\n6. Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_val)
lgb_smape = smape(y_val, lgb_pred)
print(f"   ✓ LightGBM Validation SMAPE: {lgb_smape:.2f}%")

# 7. Compare models
print("\n7. Model Comparison:")
print(f"   Ridge:    {ridge_smape:.2f}% SMAPE")
print(f"   LightGBM: {lgb_smape:.2f}% SMAPE")
print(f"   → LightGBM is {ridge_smape - lgb_smape:.2f}% better")

# 8. Choose best model and train on full data
print("\n8. Training final model on full data...")
if lgb_smape < ridge_smape:
    print("   Using LightGBM (better performance)")
    final_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
else:
    print("   Using Ridge (better performance)")
    final_model = Ridge(alpha=1.0)

final_model.fit(X_train_full, y_train_full)

# 9. Generate predictions
print("\n9. Generating predictions...")
test_predictions = final_model.predict(X_test)
test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative

print(f"   Min: {test_predictions.min():.2f}")
print(f"   Max: {test_predictions.max():.2f}")
print(f"   Mean: {test_predictions.mean():.2f}")

# 10. Create submission
print("\n10. Creating submission file...")
submission = pd.DataFrame({
    'name': test_df['name'],
    'price': test_predictions
})

submission.to_csv('submission_quickstart.csv', index=False)
print("   ✓ Saved to: submission_quickstart.csv")

print("\n" + "="*80)
print("QUICK START COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Upload 'submission_quickstart.csv' to Kaggle to get baseline score")
print("2. Run 'main.ipynb' for full analysis with:")
print("   - Detailed visualizations")
print("   - 4 model comparisons")
print("   - Hyperparameter tuning")
print("   - Feature importance analysis")
print("3. Or run 'train_model.py' for automated full pipeline")
print("\nExpected performance:")
print("  - This quick baseline: ~30-40% SMAPE")
print("  - Full pipeline with tuning: ~25-35% SMAPE")
print("="*80)

