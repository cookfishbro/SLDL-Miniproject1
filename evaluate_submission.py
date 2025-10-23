"""
Evaluate submission file against training data (for local testing)

This script can be used to check if your submission file is valid
and get a rough estimate of SMAPE on a validation set.
"""

import pandas as pd
import numpy as np
from utils import smape
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def validate_submission_format(submission_df, sample_df):
    """
    Validate that submission file has correct format
    """
    print("Validating submission format...")
    
    errors = []
    
    # Check columns
    if list(submission_df.columns) != ['name', 'price']:
        errors.append(f"❌ Incorrect columns. Expected ['name', 'price'], got {list(submission_df.columns)}")
    else:
        print("✓ Columns are correct")
    
    # Check shape
    if submission_df.shape[0] != sample_df.shape[0]:
        errors.append(f"❌ Incorrect number of rows. Expected {sample_df.shape[0]}, got {submission_df.shape[0]}")
    else:
        print(f"✓ Number of rows is correct: {submission_df.shape[0]}")
    
    # Check for missing values
    if submission_df.isnull().any().any():
        errors.append(f"❌ Submission contains missing values")
    else:
        print("✓ No missing values")
    
    # Check for negative prices
    if (submission_df['price'] < 0).any():
        errors.append(f"❌ Submission contains negative prices")
    else:
        print("✓ All prices are non-negative")
    
    # Check price range
    print(f"\nPrice statistics:")
    print(f"  Min: {submission_df['price'].min():.2f}")
    print(f"  Max: {submission_df['price'].max():.2f}")
    print(f"  Mean: {submission_df['price'].mean():.2f}")
    print(f"  Median: {submission_df['price'].median():.2f}")
    
    if errors:
        print("\n" + "="*80)
        print("VALIDATION ERRORS:")
        print("="*80)
        for error in errors:
            print(error)
        return False
    else:
        print("\n✓ Submission format is valid!")
        return True


def get_validation_score(train_csv='train.csv'):
    """
    Split training data and calculate validation SMAPE
    to estimate model performance
    """
    print("\nCalculating validation score on training data split...")
    
    # Load training data
    train_df = pd.read_csv(train_csv)
    
    # This is a placeholder - you would need to train your model
    # and predict on validation set
    print("Note: This requires training a model on train split and predicting on val split")
    print("Use main.ipynb or train_model.py to get actual validation scores")
    
    # Split data
    train, val = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)
    
    print(f"Training set: {len(train)} samples")
    print(f"Validation set: {len(val)} samples")
    
    return train, val


def main():
    """
    Main evaluation function
    """
    print("="*80)
    print("SUBMISSION EVALUATION")
    print("="*80)
    
    # Load files
    try:
        submission = pd.read_csv('submission.csv')
        sample = pd.read_csv('sample_submission.csv')
        print("\n✓ Files loaded successfully")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure 'submission.csv' and 'sample_submission.csv' exist")
        return
    
    # Validate format
    is_valid = validate_submission_format(submission, sample)
    
    if is_valid:
        print("\n" + "="*80)
        print("✓ Submission is ready for Kaggle upload!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ Please fix validation errors before submitting")
        print("="*80)
    
    # Show sample predictions
    print("\nSample predictions (first 10):")
    print(submission.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("To test your model locally:")
    print("1. Run main.ipynb or train_model.py")
    print("2. Check validation SMAPE scores in the output")
    print("3. Compare different models and configurations")
    print("="*80)


if __name__ == "__main__":
    main()

