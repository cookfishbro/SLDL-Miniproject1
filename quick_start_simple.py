"""
Quick Start Script - Simple Version (No LightGBM dependency)

This version works without OpenMP/LightGBM installation issues.
Uses Ridge Regression and XGBoost which work on macOS without additional setup.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import re
import jieba
from utils import smape

print("="*80)
print("QUICK START - PRODUCT PRICE PREDICTION (Simple Version)")
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

print("   Cleaning training data...")
train_df['text'] = train_df['name'].apply(simple_preprocess)
print("   Tokenizing training data...")
train_df['text'] = train_df['text'].apply(tokenize)

print("   Cleaning test data...")
test_df['text'] = test_df['name'].apply(simple_preprocess)
print("   Tokenizing test data...")
test_df['text'] = test_df['text'].apply(tokenize)

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

# 5. Train Ridge Regression
print("\n5. Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_val)
ridge_smape = smape(y_val, ridge_pred)
print(f"   ✓ Ridge Validation SMAPE: {ridge_smape:.2f}%")

# 6. Try different Ridge alpha values
print("\n6. Testing different Ridge configurations...")
best_alpha = 1.0
best_smape = ridge_smape

for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    current_smape = smape(y_val, pred)
    print(f"   Ridge (alpha={alpha:4.1f}): {current_smape:.2f}% SMAPE")
    
    if current_smape < best_smape:
        best_smape = current_smape
        best_alpha = alpha

print(f"\n   → Best: Ridge with alpha={best_alpha} ({best_smape:.2f}% SMAPE)")

# 7. Train final model on full data
print("\n7. Training final model on full data...")
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_full, y_train_full)

# 8. Generate predictions
print("\n8. Generating predictions...")
test_predictions = final_model.predict(X_test)
test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative

print(f"   Min: {test_predictions.min():.2f}")
print(f"   Max: {test_predictions.max():.2f}")
print(f"   Mean: {test_predictions.mean():.2f}")

# 9. Create submission
print("\n9. Creating submission file...")
submission = pd.DataFrame({
    'name': test_df['name'],
    'price': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("   ✓ Saved to: submission.csv")

print("\n" + "="*80)
print("QUICK START COMPLETE!")
print("="*80)
print("\nResults:")
print(f"  Best Model: Ridge Regression (alpha={best_alpha})")
print(f"  Validation SMAPE: {best_smape:.2f}%")
print(f"  Submission file: submission.csv")
print("\nNext steps:")
print("1. Upload 'submission.csv' to Kaggle")
print("2. To use better models (XGBoost, LightGBM), see instructions below")
print("\n" + "="*80)
print("NOTE: For better results with LightGBM/XGBoost:")
print("  Run: brew install libomp")
print("  Then you can use train_model.py or main.ipynb")
print("="*80)

