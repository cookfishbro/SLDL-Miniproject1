"""
Mini-Project 1: Product Price Prediction
Training script using ONLY scikit-learn (works without OpenMP/libomp)

This version uses Ridge, Lasso, GradientBoosting, and RandomForest
All models work perfectly on macOS without additional dependencies
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re
import jieba
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 0)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    mask = denominator != 0
    smape_val = np.zeros_like(numerator, dtype=float)
    smape_val[mask] = numerator[mask] / denominator[mask]
    
    return 100 * np.mean(smape_val)


def preprocess_text(text):
    """Clean and preprocess product name text"""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s\[\]\(\)\-\+\*\/]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()


def tokenize_chinese(text):
    """Tokenize Chinese text using jieba"""
    return ' '.join(jieba.cut(text))


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """Train and evaluate a model"""
    start_time = time.time()
    
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_smape = smape(y_train, y_train_pred)
    val_smape = smape(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    results = {
        'Model': model_name,
        'Train SMAPE': f"{train_smape:.2f}%",
        'Val SMAPE': f"{val_smape:.2f}%",
        'Val MAE': f"{val_mae:.2f}",
        'Val RMSE': f"{val_rmse:.2f}",
        'Training Time (s)': f"{training_time:.2f}"
    }
    
    print(f"\n{model_name} Results:")
    for key, value in results.items():
        if key != 'Model':
            print(f"  {key}: {value}")
    
    return results, model, val_smape


def main():
    """Main training pipeline"""
    
    print("="*80)
    print("MINI-PROJECT 1: PRODUCT PRICE PREDICTION")
    print("Using scikit-learn models (works without OpenMP)")
    print("="*80)
    
    # 1. Load Data
    print("\n1. Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    print(f"   Training set: {train_df.shape}")
    print(f"   Test set: {test_df.shape}")
    
    # 2. Data Exploration
    print("\n2. Data exploration...")
    print(f"   Price statistics:")
    print(f"   - Range: {train_df['price'].min()} - {train_df['price'].max()}")
    print(f"   - Mean: {train_df['price'].mean():.2f}")
    print(f"   - Median: {train_df['price'].median():.2f}")
    print(f"   - Missing values: {train_df.isnull().sum().sum()}")
    
    # 3. Preprocessing
    print("\n3. Preprocessing data...")
    print("   This will take a few minutes (jieba tokenization)...")
    
    tqdm.pandas(desc="   Cleaning train")
    train_df['name_cleaned'] = train_df['name'].progress_apply(preprocess_text)
    train_df['name_tokenized'] = train_df['name_cleaned'].progress_apply(tokenize_chinese)
    
    tqdm.pandas(desc="   Cleaning test")
    test_df['name_cleaned'] = test_df['name'].progress_apply(preprocess_text)
    test_df['name_tokenized'] = test_df['name_cleaned'].progress_apply(tokenize_chinese)
    
    # 4. Feature Engineering
    print("\n4. Feature engineering...")
    print("   Applying TF-IDF vectorization...")
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    X_train_full = train_df['name_tokenized']
    y_train_full = train_df['price']
    X_test = test_df['name_tokenized']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_train_full_tfidf = tfidf.transform(X_train_full)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"   TF-IDF shape: {X_train_tfidf.shape}")
    
    # 5. Model Training and Comparison
    print("\n5. Training and comparing models...")
    print("="*80)
    
    results = []
    models_trained = {}
    
    # Model 1: Ridge Regression
    print("\nTraining Ridge Regression...")
    ridge_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge_results, ridge_fitted, ridge_smape = evaluate_model(
        ridge_model, X_train_tfidf, y_train, X_val_tfidf, y_val, "Ridge Regression"
    )
    results.append(ridge_results)
    models_trained['ridge'] = (ridge_fitted, ridge_smape)
    
    # Model 2: Lasso Regression
    print("\nTraining Lasso Regression...")
    lasso_model = Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=2000)
    lasso_results, lasso_fitted, lasso_smape = evaluate_model(
        lasso_model, X_train_tfidf, y_train, X_val_tfidf, y_val, "Lasso Regression"
    )
    results.append(lasso_results)
    models_trained['lasso'] = (lasso_fitted, lasso_smape)
    
    # Model 3: Gradient Boosting
    print("\nTraining Gradient Boosting (this takes longer)...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    gb_results, gb_fitted, gb_smape = evaluate_model(
        gb_model, X_train_tfidf, y_train, X_val_tfidf, y_val, "Gradient Boosting"
    )
    results.append(gb_results)
    models_trained['gb'] = (gb_fitted, gb_smape)
    
    # Model 4: Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_results, rf_fitted, rf_smape = evaluate_model(
        rf_model, X_train_tfidf, y_train, X_val_tfidf, y_val, "Random Forest"
    )
    results.append(rf_results)
    models_trained['rf'] = (rf_fitted, rf_smape)
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # 6. Hyperparameter Tuning (Ridge)
    print("\n6. Hyperparameter tuning on Ridge Regression...")
    
    tuning_results = []
    best_alpha = 1.0
    best_smape = ridge_smape
    
    for alpha in [0.1, 1.0, 10.0, 50.0]:
        print(f"\n   Testing Ridge with alpha={alpha}...")
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        result, fitted, current_smape = evaluate_model(
            model, X_train_tfidf, y_train, X_val_tfidf, y_val, f"Ridge (alpha={alpha})"
        )
        tuning_results.append(result)
        
        if current_smape < best_smape:
            best_smape = current_smape
            best_alpha = alpha
            best_model = fitted
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    tuning_df = pd.DataFrame(tuning_results)
    print(tuning_df.to_string(index=False))
    print("="*80)
    print(f"\nBest Ridge configuration: alpha={best_alpha} ({best_smape:.2f}% SMAPE)")
    
    # 7. Final Training
    print("\n7. Training final model on full training data...")
    print(f"   Using Ridge Regression with alpha={best_alpha}")
    
    final_model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    final_model.fit(X_train_full_tfidf, y_train_full)
    
    print("   Generating predictions on test set...")
    test_predictions = final_model.predict(X_test_tfidf)
    test_predictions = np.maximum(test_predictions, 0)
    
    print(f"\n   Prediction statistics:")
    print(f"   - Min: {test_predictions.min():.2f}")
    print(f"   - Max: {test_predictions.max():.2f}")
    print(f"   - Mean: {test_predictions.mean():.2f}")
    print(f"   - Median: {np.median(test_predictions):.2f}")
    
    # 8. Create Submission
    print("\n8. Creating submission file...")
    submission = pd.DataFrame({
        'name': test_df['name'],
        'price': test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("   ✓ Submission file created: submission.csv")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print(f"  Best Model: Ridge Regression (alpha={best_alpha})")
    print(f"  Validation SMAPE: {best_smape:.2f}%")
    print(f"  Submission file: submission.csv")
    print("\nNext steps:")
    print("1. Upload 'submission.csv' to Kaggle")
    print("2. Use comparison table results for your report")
    print("3. For better models (XGBoost/LightGBM), install OpenMP:")
    print("   brew install libomp")
    print("="*80)
    
    return submission, final_model, comparison_df


if __name__ == "__main__":
    submission, model, comparison = main()

