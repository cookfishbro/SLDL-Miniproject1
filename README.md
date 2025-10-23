# Mini-Project 1: Product Price Prediction

## Overview
This project implements a machine learning pipeline to predict product prices based on product names using the SMAPE (Symmetric Mean Absolute Percentage Error) metric.

## Project Structure
```
miniproject1/
├── train.csv                 # Training data (495,387 samples)
├── test.csv                  # Test data (55,043 samples)
├── sample_submission.csv     # Submission format template
├── main.ipynb                # Main Jupyter notebook (recommended)
├── train_model.py            # Standalone Python script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── submission.csv            # Generated predictions (after running)
```

## Setup Instructions

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The first run will download jieba dictionary (this is normal and only happens once).

## Usage

### Option 1: Jupyter Notebook (Recommended)
The notebook provides interactive exploration and visualization.

```bash
jupyter notebook main.ipynb
```

Then run all cells sequentially. The notebook includes:
- Data exploration and visualization
- Detailed preprocessing explanations
- Multiple model comparisons with timing
- Hyperparameter tuning results
- Feature importance analysis
- Submission generation

### Option 2: Python Script
For automated pipeline execution:

```bash
python train_model.py
```

This will:
1. Load and preprocess data
2. Train 4 different models
3. Perform hyperparameter tuning
4. Generate `submission.csv`
5. Print comparison tables and statistics

## Pipeline Overview

### 1. Data Preprocessing
- **Text Cleaning**: Remove special characters while preserving numbers and meaningful symbols
- **Tokenization**: Use jieba for Chinese word segmentation
- **Vectorization**: Apply TF-IDF (Term Frequency-Inverse Document Frequency)

**Key Insight**: Numbers in product names (e.g., "500ml", "2kg", "16CM") are strong price indicators.

### 2. Models Implemented
| Model | Type | Best Use Case |
|-------|------|---------------|
| Ridge Regression | Linear | Fast baseline for high-dimensional data |
| XGBoost | Tree-based | Excellent for structured/tabular data |
| LightGBM | Tree-based | Optimized for speed and large datasets |
| Random Forest | Tree-based | Robust ensemble baseline |

### 3. Evaluation Metric: SMAPE
```
SMAPE = 100/n * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
```
- Range: 0% (perfect) to 200% (worst)
- Symmetric: treats over/under-estimation equally
- Scale-independent: works across different price ranges

## Results

The pipeline tests 4 models with the following general performance order:
1. **LightGBM** - Best balance of speed and accuracy
2. **XGBoost** - Similar to LightGBM but slower
3. **Ridge Regression** - Fast but limited by linearity
4. **Random Forest** - Slower training, moderate performance

**Why tree-based models work better**:
- Product pricing is non-linear (brand effects, categories)
- Can capture word interactions (e.g., "iPhone 14" vs "case iPhone")
- Handle sparse TF-IDF features efficiently

## Hyperparameter Tuning

Three configurations tested:
1. **Conservative**: Lower learning rate, smaller trees → Better generalization
2. **Moderate**: Balanced parameters → Best overall performance
3. **Aggressive**: Higher learning rate, deeper trees → Faster learning, risk of overfitting

## Report Requirements Checklist

### Part 1: Workflow Overview (5 points)
✅ Complete workflow documented in notebook and README

### Part 2: Data Pipeline (6 points)
✅ **Data Preprocessing** (4 pts):
- Observation → Action → Rationale format
- Text cleaning strategy explained
- Feature extraction justified

✅ **Tokenizer** (2 pts):
- Jieba used for Chinese segmentation
- Settings: Default jieba.cut()
- Reason: Better than character-level for Chinese

### Part 3: Model (19 points)
✅ **Model chosen** (2 pts): LightGBM
✅ **Model description** (2 pts): Gradient boosting, parameters documented
✅ **Reason for choice** (3 pts): Compared with other models, justified selection
✅ **Hyperparameter tuning** (3 pts): 3 configurations tested with results
✅ **Model comparison** (6 pts): 4 models compared with training times
✅ **Result analysis** (2 pts): Results match intuition, explained why

## Files for Submission

### For Kaggle:
- `submission.csv` - Generated predictions

### For Report:
- `main.ipynb` - Full code with explanations
- Report document (create separately) with:
  - Part 1: Workflow overview
  - Part 2: Data pipeline description
  - Part 3: Model analysis and comparison

## Tips for Improvement

1. **Ensemble Methods**: Combine predictions from multiple models
2. **Deep Learning**: Try BERT or other transformers for text
3. **Feature Engineering**: Extract brands, categories, product attributes
4. **Data Augmentation**: Handle similar products differently
5. **Cross-Validation**: Use k-fold CV for more robust evaluation

## Troubleshooting

### Common Issues

**Issue**: Jieba taking long time on first run
- **Solution**: Normal behavior, downloading dictionary (one-time only)

**Issue**: Out of memory
- **Solution**: Reduce `max_features` in TfidfVectorizer or use smaller models

**Issue**: Slow training
- **Solution**: Reduce `n_estimators` or use `n_jobs=-1` for parallel processing

## Dependencies

Main libraries used:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: ML models and preprocessing
- `xgboost`: XGBoost model
- `lightgbm`: LightGBM model
- `jieba`: Chinese text segmentation
- `matplotlib/seaborn`: Visualization

## Timeline

- **Kaggle submission deadline**: 2025/11/05 23:59
- **Report & Code submission deadline**: 2025/11/12 23:59

## Contact & Notes

- Test predictions are saved to `submission.csv`
- Model comparisons are printed with timing information
- Feature importance shows which words/terms matter most for pricing
- All random seeds set to 42 for reproducibility

Good luck with your submission! 🚀

