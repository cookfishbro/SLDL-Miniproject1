"""
COMPREHENSIVE TRAINING - Goal: SMAPE < 30%

This script tests MANY configurations systematically:
- Multiple preprocessing strategies
- Various feature engineering approaches
- Different model architectures
- Extensive hyperparameter tuning
- Ensemble methods

All results are logged to results.csv
"""

import pandas as pd
import numpy as np
import re
import jieba
import warnings
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import hstack
import csv
import sys

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Global log file
LOG_FILE = None

def log_print(message):
    """Print to both console and log file"""
    print(message)
    if LOG_FILE:
        LOG_FILE.write(message + '\n')
        LOG_FILE.flush()

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def smape(y_true, y_pred):
    """Calculate SMAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 1.0)  # Minimum price of 1
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    mask = denominator != 0
    smape_val = np.zeros_like(numerator, dtype=float)
    smape_val[mask] = numerator[mask] / denominator[mask]
    
    return 100 * np.mean(smape_val)


def extract_comprehensive_features(df):
    """Extract comprehensive features from product names"""
    features = pd.DataFrame()
    
    # Basic text statistics
    features['name_length'] = df['name'].str.len()
    features['word_count'] = df['name'].str.split().str.len()
    features['char_count'] = df['name'].apply(lambda x: len(str(x).replace(' ', '')))
    
    # Number extraction
    df['numbers'] = df['name'].apply(lambda x: re.findall(r'\d+', str(x)))
    features['num_count'] = df['numbers'].apply(len)
    features['max_num'] = df['numbers'].apply(lambda x: max([int(n) for n in x]) if x else 0)
    features['min_num'] = df['numbers'].apply(lambda x: min([int(n) for n in x]) if x else 0)
    features['sum_num'] = df['numbers'].apply(lambda x: sum([int(n) for n in x]) if x else 0)
    features['mean_num'] = df['numbers'].apply(lambda x: np.mean([int(n) for n in x]) if x else 0)
    features['first_num'] = df['numbers'].apply(lambda x: int(x[0]) if x else 0)
    
    # Character type presence
    features['has_english'] = df['name'].str.contains(r'[a-zA-Z]', regex=True).astype(int)
    features['has_chinese'] = df['name'].str.contains(r'[\u4e00-\u9fff]', regex=True).astype(int)
    features['has_number'] = df['name'].str.contains(r'\d', regex=True).astype(int)
    features['english_count'] = df['name'].apply(lambda x: len(re.findall(r'[a-zA-Z]', str(x))))
    features['chinese_count'] = df['name'].apply(lambda x: len(re.findall(r'[\u4e00-\u9fff]', str(x))))
    
    # Special characters
    features['has_parenthesis'] = df['name'].str.contains(r'[\(\)]', regex=True).astype(int)
    features['has_bracket'] = df['name'].str.contains(r'[\[\]]', regex=True).astype(int)
    features['has_dash'] = df['name'].str.contains(r'-', regex=False).astype(int)
    features['has_slash'] = df['name'].str.contains(r'/', regex=False).astype(int)
    features['has_asterisk'] = df['name'].str.contains(r'\*', regex=True).astype(int)
    features['special_char_count'] = df['name'].apply(lambda x: len(re.findall(r'[^\w\s]', str(x))))
    
    # Brand/model indicators (uppercase sequences)
    features['has_brand'] = df['name'].str.contains(r'[A-Z]{2,}', regex=True).astype(int)
    features['uppercase_count'] = df['name'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    
    # Unit indicators (common price-relevant patterns)
    features['has_kg'] = df['name'].str.contains(r'kg|KG|公斤', regex=True).astype(int)
    features['has_g'] = df['name'].str.contains(r'(?<!\d)g(?!\d)|克', regex=True).astype(int)
    features['has_ml'] = df['name'].str.contains(r'ml|ML|毫升', regex=True).astype(int)
    features['has_l'] = df['name'].str.contains(r'(?<!\d)[Ll](?!\d)|公升', regex=True).astype(int)
    features['has_cm'] = df['name'].str.contains(r'cm|CM|公分', regex=True).astype(int)
    features['has_inch'] = df['name'].str.contains(r'inch|吋', regex=True).astype(int)
    features['has_pack'] = df['name'].str.contains(r'包|入|組|盒', regex=True).astype(int)
    
    # Price-related keywords
    features['has_set'] = df['name'].str.contains(r'組|套裝|set|Set|SET', regex=True).astype(int)
    features['has_luxury'] = df['name'].str.contains(r'豪華|高級|premium|Premium', regex=True).astype(int)
    
    return features


def preprocess_text_v1(text):
    """Version 1: Keep most characters"""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text_v2(text):
    """Version 2: Remove special chars"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text_v3(text):
    """Version 3: Aggressive cleaning"""
    text = str(text).lower()
    # Keep numbers, letters, Chinese, and important symbols
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff\s\-\+\*\/\(\)\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_chinese(text):
    """Tokenize with jieba"""
    return ' '.join(jieba.cut(str(text)))


def log_result(results_file, result_dict):
    """Log result to CSV"""
    file_exists = False
    try:
        with open(results_file, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(results_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name, use_log=False):
    """Train and evaluate a model"""
    start_time = time.time()
    
    try:
        if use_log:
            y_train_log = np.log1p(y_train)
            model.fit(X_train, y_train_log)
            y_val_pred_log = model.predict(X_val)
            y_val_pred = np.expm1(y_val_pred_log)
        else:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
        
        y_val_pred = np.maximum(y_val_pred, 1.0)
        training_time = time.time() - start_time
        val_smape = smape(y_val, y_val_pred)
        
        return val_smape, training_time, model
    except Exception as e:
        print(f"   ERROR in {model_name}: {str(e)}")
        return 999.0, 0, None


def main():
    global LOG_FILE
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_log_{timestamp}.txt'
    LOG_FILE = open(log_filename, 'w')
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_print("="*80)
    log_print("GRADIENT BOOSTING + LOG TRANSFORMATION - TARGET: SMAPE < 30%")
    log_print("="*80)
    log_print(f"Start time: {start_datetime}")
    log_print(f"Log file: {log_filename}")
    log_print("="*80)
    
    results_file = 'training_results.csv'
    
    # 1. Load Data
    step_start = time.time()
    log_print("\n[1/8] Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    log_print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    log_print(f"   Price range: {train_df['price'].min()} - {train_df['price'].max()}")
    log_print(f"   Price mean: {train_df['price'].mean():.2f}, median: {train_df['price'].median():.2f}")
    log_print(f"   ⏱️  Step time: {format_time(time.time() - step_start)}")
    
    # 2. Extract Comprehensive Features
    step_start = time.time()
    log_print("\n[2/8] Extracting comprehensive features...")
    train_features = extract_comprehensive_features(train_df.copy())
    test_features = extract_comprehensive_features(test_df.copy())
    log_print(f"   Extracted {train_features.shape[1]} numerical features")
    log_print(f"   ⏱️  Step time: {format_time(time.time() - step_start)}")
    
    # 3. Test Multiple Preprocessing Approaches
    step_start = time.time()
    log_print("\n[3/8] Testing preprocessing approaches...")
    
    preprocessing_methods = {
        'v1_notok': (preprocess_text_v1, False),
        'v1_tok': (preprocess_text_v1, True),
        'v2_notok': (preprocess_text_v2, False),
        'v2_tok': (preprocess_text_v2, True),
        'v3_tok': (preprocess_text_v3, True),
    }
    
    preprocessed_data = {}
    
    for name, (preprocess_fn, use_tokenize) in preprocessing_methods.items():
        log_print(f"   Processing: {name}...")
        
        if use_tokenize:
            train_text = train_df['name'].apply(preprocess_fn).apply(tokenize_chinese)
            test_text = test_df['name'].apply(preprocess_fn).apply(tokenize_chinese)
        else:
            train_text = train_df['name'].apply(preprocess_fn)
            test_text = test_df['name'].apply(preprocess_fn)
        
        preprocessed_data[name] = (train_text, test_text)
    
    log_print(f"   ⏱️  Step time: {format_time(time.time() - step_start)}")
    
    # 4. Prepare train/val split
    step_start = time.time()
    log_print("\n[4/8] Preparing train/validation split...")
    y_train_full = train_df['price']
    
    # Use stratified split by price bins to get representative validation set
    price_bins = pd.qcut(y_train_full, q=10, labels=False, duplicates='drop')
    
    train_idx, val_idx = train_test_split(
        range(len(train_df)), 
        test_size=0.15, 
        random_state=RANDOM_STATE,
        stratify=price_bins
    )
    
    y_train = y_train_full.iloc[train_idx]
    y_val = y_train_full.iloc[val_idx]
    
    log_print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")
    log_print(f"   ⏱️  Step time: {format_time(time.time() - step_start)}")
    
    # 5. Test Different Vectorization Strategies
    step_start = time.time()
    log_print("\n[5/8] Testing vectorization strategies...")
    
    vectorization_configs = [
        ('tfidf_5k_12', TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3, max_df=0.9, sublinear_tf=True)),
        ('tfidf_8k_13', TfidfVectorizer(max_features=8000, ngram_range=(1,3), min_df=2, max_df=0.95, sublinear_tf=True)),
        ('tfidf_10k_13', TfidfVectorizer(max_features=10000, ngram_range=(1,3), min_df=2, max_df=0.95, sublinear_tf=True)),
        ('tfidf_15k_23', TfidfVectorizer(max_features=15000, ngram_range=(2,3), min_df=2, max_df=0.95, sublinear_tf=True)),
    ]
    
    log_print(f"   ⏱️  Step time: {format_time(time.time() - step_start)}")
    
    # 6. Model Configurations
    step_start = time.time()
    log_print("\n[6/8] Setting up model configurations...")
    
    model_configs = [
        # ONLY GRADIENT BOOSTING - Testing with log transformation
        
        # Basic configurations
        ('GB_100_5_lr01', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=RANDOM_STATE)),
        ('GB_150_7_lr01', GradientBoostingRegressor(n_estimators=150, max_depth=7, learning_rate=0.1, subsample=0.8, random_state=RANDOM_STATE)),
        ('GB_200_7_lr01', GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.8, random_state=RANDOM_STATE)),
        
        # Deeper trees
        ('GB_200_10_lr008', GradientBoostingRegressor(n_estimators=200, max_depth=10, learning_rate=0.08, subsample=0.8, random_state=RANDOM_STATE)),
        ('GB_300_8_lr005', GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)),
        ('GB_300_10_lr008', GradientBoostingRegressor(n_estimators=300, max_depth=10, learning_rate=0.08, subsample=0.8, random_state=RANDOM_STATE)),
        
        # More aggressive (more trees, deeper)
        ('GB_400_10_lr005', GradientBoostingRegressor(n_estimators=400, max_depth=10, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)),
        ('GB_500_12_lr005', GradientBoostingRegressor(n_estimators=500, max_depth=12, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)),
        
        # Even more aggressive
        ('GB_500_15_lr003', GradientBoostingRegressor(n_estimators=500, max_depth=15, learning_rate=0.03, subsample=0.8, random_state=RANDOM_STATE)),
        
        # COMMENTED OUT - Other models for later testing
        # # Random Forest - Various configurations
        # ('RF_300_30', RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=2, random_state=RANDOM_STATE, n_jobs=-1)),
        # ('RF_500_40', RandomForestRegressor(n_estimators=500, max_depth=40, min_samples_split=2, random_state=RANDOM_STATE, n_jobs=-1)),
        # 
        # # Extra Trees (often better than RF)
        # ('ExtraTrees_200', ExtraTreesRegressor(n_estimators=200, max_depth=25, random_state=RANDOM_STATE, n_jobs=-1)),
        # ('ExtraTrees_300', ExtraTreesRegressor(n_estimators=300, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1)),
        # 
        # # Ridge with different alphas
        # ('Ridge_a0.1', Ridge(alpha=0.1, random_state=RANDOM_STATE)),
        # ('Ridge_a1', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        # ('Ridge_a5', Ridge(alpha=5.0, random_state=RANDOM_STATE)),
    ]
    
    log_print(f"   ⏱️  Step time: {format_time(time.time() - step_start)}")
    
    # 7. Systematic Testing
    step_start = time.time()
    log_print("\n[7/8] SYSTEMATIC TESTING (this will take time)...")
    log_print("="*80)
    
    best_smape = 999.0
    best_config = None
    test_count = 0
    # Only testing with log transformation now (no *2)
    total_tests = len(preprocessed_data) * len(vectorization_configs) * len(model_configs)
    
    log_print(f"Total configurations to test: {total_tests}")
    log_print("Testing ONLY with log transformation (better results)")
    log_print("="*80)
    
    for preproc_name, (train_text, test_text) in preprocessed_data.items():
        for vec_name, vectorizer in vectorization_configs:
            combo_start = time.time()
            log_print(f"\n{'='*80}")
            log_print(f"Preprocessing: {preproc_name}, Vectorization: {vec_name}")
            log_print(f"{'='*80}")
            
            # Fit vectorizer
            try:
                X_train_vec = vectorizer.fit_transform(train_text.iloc[train_idx])
                X_val_vec = vectorizer.transform(train_text.iloc[val_idx])
                
                # Combine with numerical features
                X_train_combined = hstack([X_train_vec, train_features.iloc[train_idx].values])
                X_val_combined = hstack([X_val_vec, train_features.iloc[val_idx].values])
                
                log_print(f"   Feature shape: {X_train_combined.shape}")
                
            except Exception as e:
                log_print(f"   ERROR in vectorization: {str(e)}")
                continue
            
            # Test all models - ONLY WITH LOG TRANSFORMATION
            for model_name, model in model_configs:
                # Only test with log transformation (skip non-log)
                use_log = True
                test_count += 1
                log_suffix = "_log"
                full_name = f"{preproc_name}|{vec_name}|{model_name}{log_suffix}"
                
                # Train and evaluate
                val_smape, train_time, trained_model = train_and_evaluate(
                    model, X_train_combined, y_train, X_val_combined, y_val,
                    full_name, use_log
                )
                
                # Log result
                result = {
                    'preprocessing': preproc_name,
                    'vectorization': vec_name,
                    'model': model_name,
                    'use_log': use_log,
                    'val_smape': f"{val_smape:.4f}",
                    'train_time': f"{train_time:.2f}",
                    'full_name': full_name
                }
                log_result(results_file, result)
                
                # Print progress
                elapsed = time.time() - start_time
                status = "✓" if val_smape < 40 else "✗"
                log_print(f"   [{test_count}/{total_tests}] {status} {model_name}{log_suffix}: {val_smape:.2f}% SMAPE (train: {train_time:.1f}s) [elapsed: {format_time(elapsed)}]")
                
                # Track best
                if val_smape < best_smape:
                    best_smape = val_smape
                    best_config = {
                        'name': full_name,
                        'preproc': preproc_name,
                        'vec': vec_name,
                        'vec_obj': vectorizer,
                        'model': model,
                        'use_log': use_log,
                        'smape': val_smape
                    }
                    log_print(f"   🎉 NEW BEST: {val_smape:.2f}% SMAPE")
            
            # Log time for this combination
            combo_time = time.time() - combo_start
            log_print(f"   ⏱️  Combination time: {format_time(combo_time)}")
    
    log_print(f"\n⏱️  Total testing time: {format_time(time.time() - step_start)}")
    
    # 8. Train Best Model on Full Data
    step_start = time.time()
    log_print("\n" + "="*80)
    log_print("[8/8] TRAINING BEST MODEL ON FULL DATA")
    log_print("="*80)
    
    if best_config:
        log_print(f"\nBest Configuration:")
        log_print(f"  Name: {best_config['name']}")
        log_print(f"  Validation SMAPE: {best_smape:.2f}%")
        log_print(f"  Preprocessing: {best_config['preproc']}")
        log_print(f"  Vectorization: {best_config['vec']}")
        log_print(f"  Use Log: {best_config['use_log']}")
        
        # Get the preprocessed text
        train_text, test_text = preprocessed_data[best_config['preproc']]
        
        # Vectorize
        log_print("\nVectorizing full dataset...")
        vectorizer = best_config['vec_obj']
        X_train_full_vec = vectorizer.fit_transform(train_text)
        X_test_vec = vectorizer.transform(test_text)
        
        # Combine features
        X_train_full_combined = hstack([X_train_full_vec, train_features.values])
        X_test_combined = hstack([X_test_vec, test_features.values])
        
        # Train
        log_print("Training final model...")
        model = best_config['model']
        
        if best_config['use_log']:
            y_train_log = np.log1p(y_train_full)
            model.fit(X_train_full_combined, y_train_log)
            test_pred_log = model.predict(X_test_combined)
            test_predictions = np.expm1(test_pred_log)
        else:
            model.fit(X_train_full_combined, y_train_full)
            test_predictions = model.predict(X_test_combined)
        
        test_predictions = np.maximum(test_predictions, 1.0)
        
        log_print(f"   ⏱️  Final training time: {format_time(time.time() - step_start)}")
        
        # Create submission
        log_print("Creating submission...")
        submission = pd.DataFrame({
            'name': test_df['name'],
            'price': test_predictions
        })
        
        submission.to_csv('submission_best.csv', index=False)
        
        # Calculate total time
        total_time = time.time() - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_print("\n" + "="*80)
        log_print("TRAINING COMPLETE!")
        log_print("="*80)
        log_print(f"\nStart time: {start_datetime}")
        log_print(f"End time: {end_datetime}")
        log_print(f"Total time: {format_time(total_time)}")
        log_print(f"\nBest Model: {best_config['name']}")
        log_print(f"Validation SMAPE: {best_smape:.2f}%")
        log_print(f"\nExpected Kaggle Score: ~{best_smape-2:.1f}%-{best_smape+2:.1f}%")
        log_print(f"\nFiles created:")
        log_print(f"  - submission_best.csv (predictions)")
        log_print(f"  - training_results.csv (all results)")
        log_print(f"  - {log_filename} (training log)")
        log_print("\nCheck training_results.csv for all model performances!")
        log_print("="*80)
        
        # Show top 10 models
        log_print("\nTop 10 Models:")
        results_df = pd.read_csv(results_file)
        results_df['val_smape'] = results_df['val_smape'].astype(float)
        top_10 = results_df.nsmallest(10, 'val_smape')
        log_print(top_10[['full_name', 'val_smape', 'train_time']].to_string(index=False))
        
    else:
        log_print("ERROR: No valid configuration found!")
    
    # Close log file
    if LOG_FILE:
        LOG_FILE.close()
    
    return submission if best_config else None


if __name__ == "__main__":
    submission = main()

