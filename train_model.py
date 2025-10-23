"""
Mini-Project 1: Product Price Prediction
Main training script with complete pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
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
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Formula: SMAPE = 100/n * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    Args:
        y_true: actual values
        y_pred: predicted values
    
    Returns:
        SMAPE score (0-200, lower is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure predictions are non-negative (prices can't be negative)
    y_pred = np.maximum(y_pred, 0)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    smape_val = np.zeros_like(numerator)
    smape_val[mask] = numerator[mask] / denominator[mask]
    
    return 100 * np.mean(smape_val)


def preprocess_text(text):
    """
    Clean and preprocess product name text
    
    Strategy:
    - Convert to lowercase for English text
    - Keep numbers as they often indicate size/quantity (important for pricing)
    - Remove excessive punctuation but keep meaningful symbols
    - Preserve Chinese characters
    """
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s\[\]\(\)\-\+\*\/]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()


def extract_numerical_features(text):
    """
    Extract numerical features from product name
    
    Rationale: Numbers in product names often indicate:
    - Quantity (e.g., "2kg", "100入")
    - Size (e.g., "16CM", "500ml")
    - Model numbers
    These are strong indicators of price
    """
    text = str(text)
    numbers = re.findall(r'\d+', text)
    
    features = {
        'num_count': len(numbers),
        'max_number': max([int(n) for n in numbers]) if numbers else 0,
        'min_number': min([int(n) for n in numbers]) if numbers else 0,
        'sum_numbers': sum([int(n) for n in numbers]) if numbers else 0,
        'avg_number': np.mean([int(n) for n in numbers]) if numbers else 0,
    }
    
    return features


def tokenize_chinese(text):
    """
    Tokenize Chinese text using jieba
    
    Jieba is chosen because:
    - Specifically designed for Chinese word segmentation
    - Better than character-level tokenization for understanding context
    - Can recognize product-related terms
    """
    return ' '.join(jieba.cut(text))


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train and evaluate a model, returning performance metrics
    """
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
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
    
    # Split data
    X_train_full = train_df['name_tokenized']
    y_train_full = train_df['price']
    X_test = test_df['name_tokenized']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Vectorize
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
    
    # Model 2: XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_results, xgb_fitted, xgb_smape = evaluate_model(
        xgb_model, X_train_tfidf, y_train, X_val_tfidf, y_val, "XGBoost"
    )
    results.append(xgb_results)
    models_trained['xgb'] = (xgb_fitted, xgb_smape)
    
    # Model 3: LightGBM
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_results, lgb_fitted, lgb_smape = evaluate_model(
        lgb_model, X_train_tfidf, y_train, X_val_tfidf, y_val, "LightGBM"
    )
    results.append(lgb_results)
    models_trained['lgb'] = (lgb_fitted, lgb_smape)
    
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
    
    # 6. Select Best Model and Hyperparameter Tuning
    print("\n6. Hyperparameter tuning on best model...")
    
    # Find best model
    best_model_name = min(models_trained.items(), key=lambda x: x[1][1])[0]
    print(f"   Best base model: {best_model_name}")
    
    # Tune LightGBM (typically best for this task)
    print("\n   Testing different LightGBM configurations...")
    
    configs = [
        {
            'name': 'Config 1: Conservative',
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
        },
        {
            'name': 'Config 2: Moderate',
            'params': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 7,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
        },
        {
            'name': 'Config 3: Aggressive',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.15,
                'max_depth': 10,
                'num_leaves': 50,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
        }
    ]
    
    tuning_results = []
    best_config = None
    best_config_smape = float('inf')
    
    for config in configs:
        print(f"\n   Testing {config['name']}...")
        model = lgb.LGBMRegressor(**config['params'])
        result, fitted_model, config_smape = evaluate_model(
            model, X_train_tfidf, y_train, X_val_tfidf, y_val, config['name']
        )
        tuning_results.append(result)
        
        if config_smape < best_config_smape:
            best_config_smape = config_smape
            best_config = config
            best_tuned_model = fitted_model
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    tuning_df = pd.DataFrame(tuning_results)
    print(tuning_df.to_string(index=False))
    print("="*80)
    print(f"\nBest configuration: {best_config['name']}")
    
    # 7. Final Training and Prediction
    print("\n7. Training final model on full training data...")
    final_model = lgb.LGBMRegressor(**best_config['params'])
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
    
    # 9. Feature Importance
    print("\n9. Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Top 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Submit 'submission.csv' to Kaggle")
    print("2. Review model comparison and feature importance")
    print("3. Consider additional improvements (ensembles, deep learning, etc.)")
    
    return submission, final_model, comparison_df


if __name__ == "__main__":
    submission, model, comparison = main()

