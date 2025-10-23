"""
IMPROVED Version - Much Better Performance
Uses advanced techniques:
- Log transformation of prices
- Better feature engineering
- Ensemble of multiple models
- More aggressive Random Forest
- Better preprocessing
"""

import pandas as pd
import numpy as np
import re
import jieba
import warnings
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def smape(y_true, y_pred):
    """Calculate SMAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 0)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    mask = denominator != 0
    smape_val = np.zeros_like(numerator, dtype=float)
    smape_val[mask] = numerator[mask] / denominator[mask]
    
    return 100 * np.mean(smape_val)


def extract_features(df):
    """Extract additional numerical features from product names"""
    print("   Extracting numerical features...")
    
    # Extract all numbers
    df['numbers'] = df['name'].apply(lambda x: re.findall(r'\d+', str(x)))
    df['num_count'] = df['numbers'].apply(len)
    df['max_num'] = df['numbers'].apply(lambda x: max([int(n) for n in x]) if x else 0)
    df['sum_num'] = df['numbers'].apply(lambda x: sum([int(n) for n in x]) if x else 0)
    
    # Text length features
    df['name_length'] = df['name'].str.len()
    df['word_count'] = df['name'].str.split().str.len()
    
    # Character type features
    df['has_english'] = df['name'].str.contains(r'[a-zA-Z]').astype(int)
    df['has_chinese'] = df['name'].str.contains(r'[\u4e00-\u9fff]').astype(int)
    df['has_number'] = df['name'].str.contains(r'\d').astype(int)
    
    # Special characters that might indicate value
    df['has_brand'] = df['name'].str.contains(r'[A-Z]{2,}').astype(int)
    df['has_parenthesis'] = df['name'].str.contains(r'[\(\)]').astype(int)
    df['has_bracket'] = df['name'].str.contains(r'[\[\]]').astype(int)
    
    feature_cols = ['num_count', 'max_num', 'sum_num', 'name_length', 'word_count',
                    'has_english', 'has_chinese', 'has_number', 'has_brand',
                    'has_parenthesis', 'has_bracket']
    
    return df[feature_cols]


def preprocess_text(text):
    """Clean text"""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s\[\]\(\)\-\+\*\/]', ' ', text)
    return ' '.join(text.split()).strip()


def tokenize_chinese(text):
    """Tokenize with jieba"""
    return ' '.join(jieba.cut(text))


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name, use_log=False):
    """Train and evaluate model"""
    start_time = time.time()
    
    if use_log:
        # Train on log-transformed prices
        y_train_log = np.log1p(y_train)
        model.fit(X_train, y_train_log)
        
        # Predict and inverse transform
        y_train_pred_log = model.predict(X_train)
        y_val_pred_log = model.predict(X_val)
        y_train_pred = np.expm1(y_train_pred_log)
        y_val_pred = np.expm1(y_val_pred_log)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
    
    training_time = time.time() - start_time
    
    train_smape = smape(y_train, y_train_pred)
    val_smape = smape(y_val, y_val_pred)
    
    results = {
        'Model': model_name,
        'Train SMAPE': f"{train_smape:.2f}%",
        'Val SMAPE': f"{val_smape:.2f}%",
        'Training Time (s)': f"{training_time:.2f}"
    }
    
    print(f"   {model_name}: Val SMAPE = {val_smape:.2f}% (Time: {training_time:.1f}s)")
    
    return results, model, val_smape


def main():
    print("="*80)
    print("IMPROVED PRODUCT PRICE PREDICTION")
    print("="*80)
    
    # 1. Load Data
    print("\n1. Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 2. Extract Features
    print("\n2. Feature extraction...")
    train_features = extract_features(train_df.copy())
    test_features = extract_features(test_df.copy())
    
    # 3. Preprocessing
    print("\n3. Text preprocessing (this takes time)...")
    tqdm.pandas(desc="   Train")
    train_df['text'] = train_df['name'].progress_apply(preprocess_text).progress_apply(tokenize_chinese)
    
    tqdm.pandas(desc="   Test")
    test_df['text'] = test_df['name'].progress_apply(preprocess_text).progress_apply(tokenize_chinese)
    
    # 4. TF-IDF with better parameters
    print("\n4. Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=8000,  # More features
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,  # More lenient
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_full = train_df['text']
    y_train_full = train_df['price']
    X_test = test_df['text']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE
    )
    
    # Get indices for features
    train_idx = X_train.index
    val_idx = X_val.index
    
    # TF-IDF transform
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_train_full_tfidf = tfidf.transform(X_train_full)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Add numerical features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, train_features.iloc[train_idx].values])
    X_val_combined = hstack([X_val_tfidf, train_features.iloc[val_idx].values])
    X_train_full_combined = hstack([X_train_full_tfidf, train_features.values])
    X_test_combined = hstack([X_test_tfidf, test_features.values])
    
    print(f"   Combined feature shape: {X_train_combined.shape}")
    
    # 5. Train Multiple Models
    print("\n5. Training models...")
    results = []
    models = {}
    
    # Model 1: Ridge with log transform
    print("\n   Training Ridge (log-transformed)...")
    ridge = Ridge(alpha=5.0, random_state=RANDOM_STATE)
    r1, m1, s1 = evaluate_model(ridge, X_train_combined, y_train, X_val_combined, y_val, 
                                  "Ridge (log)", use_log=True)
    results.append(r1)
    models['ridge_log'] = (m1, s1, True)
    
    # Model 2: Random Forest with more trees
    print("\n   Training Random Forest (aggressive)...")
    rf = RandomForestRegressor(
        n_estimators=200,  # More trees
        max_depth=30,  # Deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    r2, m2, s2 = evaluate_model(rf, X_train_combined, y_train, X_val_combined, y_val,
                                 "Random Forest", use_log=False)
    results.append(r2)
    models['rf'] = (m2, s2, False)
    
    # Model 3: Random Forest with log
    print("\n   Training Random Forest (log-transformed)...")
    rf_log = RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    r3, m3, s3 = evaluate_model(rf_log, X_train_combined, y_train, X_val_combined, y_val,
                                 "Random Forest (log)", use_log=True)
    results.append(r3)
    models['rf_log'] = (m3, s3, True)
    
    # Model 4: Gradient Boosting with log
    print("\n   Training Gradient Boosting (log)...")
    gb = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    r4, m4, s4 = evaluate_model(gb, X_train_combined, y_train, X_val_combined, y_val,
                                 "Gradient Boosting (log)", use_log=True)
    results.append(r4)
    models['gb_log'] = (m4, s4, True)
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # 6. Find best model
    best_name = min(models.items(), key=lambda x: x[1][1])[0]
    best_model, best_smape, use_log = models[best_name]
    
    print(f"\n6. Best model: {best_name} ({best_smape:.2f}% SMAPE)")
    
    # 7. Train on full data
    print("\n7. Training final model on full data...")
    
    if use_log:
        y_train_full_log = np.log1p(y_train_full)
        best_model.fit(X_train_full_combined, y_train_full_log)
        test_pred_log = best_model.predict(X_test_combined)
        test_predictions = np.expm1(test_pred_log)
    else:
        best_model.fit(X_train_full_combined, y_train_full)
        test_predictions = best_model.predict(X_test_combined)
    
    # Ensure non-negative
    test_predictions = np.maximum(test_predictions, 0)
    
    print(f"   Predictions: min={test_predictions.min():.2f}, "
          f"max={test_predictions.max():.2f}, mean={test_predictions.mean():.2f}")
    
    # 8. Create ensemble (average top 2 models)
    print("\n8. Creating ensemble prediction...")
    sorted_models = sorted(models.items(), key=lambda x: x[1][1])[:2]
    
    ensemble_preds = []
    for name, (model, smape_score, use_log_flag) in sorted_models:
        print(f"   Using {name} ({smape_score:.2f}% SMAPE)")
        
        if use_log_flag:
            y_train_full_log = np.log1p(y_train_full)
            model.fit(X_train_full_combined, y_train_full_log)
            pred_log = model.predict(X_test_combined)
            pred = np.expm1(pred_log)
        else:
            model.fit(X_train_full_combined, y_train_full)
            pred = model.predict(X_test_combined)
        
        ensemble_preds.append(np.maximum(pred, 0))
    
    # Weighted average (better model gets more weight)
    weights = [0.6, 0.4]  # First model (best) gets 60%
    final_predictions = np.average(ensemble_preds, axis=0, weights=weights)
    
    print(f"   Ensemble: min={final_predictions.min():.2f}, "
          f"max={final_predictions.max():.2f}, mean={final_predictions.mean():.2f}")
    
    # 9. Save submission
    print("\n9. Creating submission...")
    submission = pd.DataFrame({
        'name': test_df['name'],
        'price': final_predictions
    })
    
    submission.to_csv('submission_improved.csv', index=False)
    print("   ✓ Saved to: submission_improved.csv")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nBest Single Model: {best_name} ({best_smape:.2f}% SMAPE)")
    print(f"Ensemble: Top 2 models averaged")
    print(f"\nExpected Kaggle Score: ~{best_smape-2:.1f}-{best_smape+2:.1f}% SMAPE")
    print("\nFile: submission_improved.csv")
    print("="*80)
    
    return submission


if __name__ == "__main__":
    submission = main()

