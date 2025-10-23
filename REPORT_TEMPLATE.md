# Mini-Project 1 Report Template

**Student Name**: [Your Name]  
**Student ID**: [Your ID]  
**Date**: [Date]

---

## Part 1 – Workflow Overview (5 points)

### Overall Process

The complete workflow for this product price prediction task follows these steps:

1. **Data Loading and Exploration**
   - Load training data (495,387 samples) and test data (55,043 samples)
   - Analyze price distributions, check for missing values
   - Examine product name characteristics (Chinese/English text, numbers, special characters)

2. **Data Preprocessing**
   - Clean text: remove excessive punctuation, preserve numbers and meaningful symbols
   - Tokenize Chinese text using jieba word segmentation
   - Create train/validation split (80/20) for model evaluation

3. **Feature Engineering**
   - Apply TF-IDF vectorization to convert text to numerical features
   - Settings: max_features=5000, ngram_range=(1,2), min_df=3, max_df=0.9

4. **Model Development**
   - Train and compare 4 different models:
     - Ridge Regression (linear baseline)
     - XGBoost (gradient boosting)
     - LightGBM (optimized gradient boosting)
     - Random Forest (ensemble method)
   - Evaluate using SMAPE metric on validation set

5. **Hyperparameter Tuning**
   - Test 3 different configurations on best-performing model
   - Compare training time vs. performance trade-offs

6. **Final Training and Prediction**
   - Train best model on full training dataset
   - Generate predictions on test set
   - Create submission file

7. **Analysis**
   - Analyze feature importance
   - Compare prediction distributions
   - Document insights and findings

---

## Part 2 – Data Pipeline (6 points)

### Data Preprocessing (4 points)

**Observation → Action → Rationale format**:

1. **Product Name Text Cleaning**
   - **Observation**: Product names contain mixed Chinese/English text with special characters (brackets, symbols) and numbers
   - **Action**: Applied regex-based cleaning: `re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s\[\]\(\)\-\+\*\/]', ' ', text)`
   - **Rationale**: Numbers often indicate size/quantity (e.g., "500ml", "2kg") which strongly correlate with price. Brackets may indicate product variants. Removed noise while preserving important information.
   - **Implementation**: Used `preprocess_text()` function

2. **Chinese Text Segmentation**
   - **Observation**: Chinese text is continuous without word boundaries, making it difficult for standard tokenization
   - **Action**: Applied jieba word segmentation using `jieba.cut()`
   - **Rationale**: Word-level tokenization captures semantic meaning better than character-level. For example, "蛋黃粽" (egg yolk rice dumpling) is more meaningful as one term than three characters.
   - **Implementation**: Used `tokenize_chinese()` function with progress tracking

3. **Text Vectorization (TF-IDF)**
   - **Observation**: Machine learning models require numerical input, but we have text data
   - **Action**: Applied TF-IDF vectorization with specific parameters:
     - `max_features=5000`: Limit dimensionality while keeping most important terms
     - `ngram_range=(1,2)`: Capture single words and two-word phrases
     - `min_df=3`: Ignore very rare terms (reduce noise)
     - `max_df=0.9`: Ignore very common terms (low information value)
   - **Rationale**: TF-IDF assigns higher weights to distinctive terms that appear in fewer documents. Product-specific terms (brands, models) are more indicative of price than common words.

4. **Train/Validation Split**
   - **Observation**: Need to evaluate model performance before final testing
   - **Action**: Split training data 80/20 using stratified random sampling
   - **Rationale**: Validation set allows us to compare models and tune hyperparameters without touching test data, preventing overfitting.

### Tokenizer (2 points)

**Tokenizer Used**: Jieba (结巴) Chinese text segmentation library

**Settings**:
- Function: `jieba.cut(text, cut_all=False)` (default precision mode)
- No custom dictionary added
- Default settings used for general product names

**Justification**:
- **Why Jieba**: Specifically designed for Chinese text segmentation, uses dynamic programming and HMM for unknown words
- **Why not character-level**: Character-level tokenization loses semantic meaning (e.g., "手" "機" separately vs "手機" meaning phone)
- **Why not other tokenizers**: Alternatives like THULAC or pkuseg require more setup and aren't significantly better for product names
- **Precision mode**: Chosen over full mode to avoid over-segmentation which creates noise in TF-IDF

---

## Part 3 – Model (19 points)

### Model Chosen (2 points)

**Final Model**: LightGBM (Light Gradient Boosting Machine) with moderate configuration

### Model Description (2 points)

**LightGBM** is a gradient boosting framework that uses tree-based learning algorithms. 

**Key characteristics**:
- Type: Ensemble learning using gradient boosting
- Base learners: Decision trees
- Training: Iteratively builds trees, each correcting errors of previous ones
- Optimization: Uses leaf-wise (best-first) tree growth strategy

**Final Model Parameters**:
```python
LGBMRegressor(
    n_estimators=150,        # Number of boosting iterations
    learning_rate=0.1,       # Step size shrinkage
    max_depth=7,             # Maximum tree depth
    num_leaves=31,           # Maximum leaves per tree
    subsample=0.8,           # Row sampling ratio
    colsample_bytree=0.8,    # Column sampling ratio
    random_state=42          # For reproducibility
)
```

### Reason for Choice (3 points)

**Why LightGBM over other models**:

1. **Better Performance**:
   - [Fill in actual SMAPE scores from your runs]
   - LightGBM: ~[XX.XX]% validation SMAPE
   - XGBoost: ~[XX.XX]% validation SMAPE
   - Ridge: ~[XX.XX]% validation SMAPE
   - Random Forest: ~[XX.XX]% validation SMAPE

2. **Training Efficiency**:
   - LightGBM: ~[XX.XX] seconds
   - XGBoost: ~[XX.XX] seconds (slower)
   - Random Forest: ~[XX.XX] seconds (much slower)
   - Ridge: ~[XX.XX] seconds (fastest but less accurate)

3. **Technical Advantages**:
   - Handles sparse TF-IDF features efficiently
   - Memory-efficient (important for 495k training samples)
   - Captures non-linear price patterns (brand effects, product categories)
   - Built-in regularization reduces overfitting

**Other Models Considered**:

- **Ridge Regression**: Fast baseline but too simple for complex price relationships
- **XGBoost**: Similar performance but slower training time
- **Random Forest**: Decent performance but inefficient for large sparse matrices

### Hyperparameter Tuning (3 points)

**Manual Tuning Approach**: Tested 3 configurations

| Configuration | n_estimators | learning_rate | max_depth | num_leaves | Val SMAPE | Train Time |
|---------------|--------------|---------------|-----------|------------|-----------|------------|
| Conservative  | 100          | 0.05          | 5         | 20         | [XX.XX]%  | [XX.XX]s   |
| Moderate      | 150          | 0.1           | 7         | 31         | [XX.XX]%  | [XX.XX]s   |
| Aggressive    | 200          | 0.15          | 10        | 50         | [XX.XX]%  | [XX.XX]s   |

**Analysis**:

[Fill in based on actual results, but typical analysis:]

- **Conservative**: Lower risk of overfitting, but may underfit (higher SMAPE)
- **Moderate**: Best balance between bias and variance
- **Aggressive**: Faster learning but risk of overfitting to training data

**Pros of Manual Tuning**:
- Full control over parameter exploration
- Can apply domain knowledge
- Interpretable and reproducible
- Computationally efficient (only 3 trials)

**Cons of Manual Tuning**:
- Limited search space (may miss optimal combinations)
- Requires expertise to choose parameter ranges
- Sequential testing (not parallel optimization)

**Alternative Approach**: Could use Optuna/GridSearch for automated tuning, but this requires more computational resources and may overfit to validation set.

### Model Comparison (6 points)

**Comparison Table**:

| Model              | Train SMAPE | Val SMAPE | Val MAE  | Val RMSE  | Training Time (s) |
|--------------------|-------------|-----------|----------|-----------|-------------------|
| Ridge Regression   | [XX.XX]%    | [XX.XX]%  | [XX.XX]  | [XX.XX]   | [XX.XX]           |
| XGBoost            | [XX.XX]%    | [XX.XX]%  | [XX.XX]  | [XX.XX]   | [XX.XX]           |
| LightGBM           | [XX.XX]%    | [XX.XX]%  | [XX.XX]  | [XX.XX]   | [XX.XX]           |
| Random Forest      | [XX.XX]%    | [XX.XX]%  | [XX.XX]  | [XX.XX]   | [XX.XX]           |

**Analysis**:

[Fill in based on actual results, example analysis:]

1. **Ridge Regression**:
   - Fastest training (~X seconds)
   - Higher SMAPE (~XX%) indicating limited by linear assumption
   - Good baseline but insufficient for complex price patterns

2. **XGBoost**:
   - Strong performance (~XX% SMAPE)
   - Slower training (~XX seconds)
   - Captures non-linear relationships well

3. **LightGBM** (Selected):
   - Best SMAPE (~XX%)
   - Faster than XGBoost (~XX seconds)
   - Optimal for this task

4. **Random Forest**:
   - Moderate performance (~XX% SMAPE)
   - Slowest training (~XX seconds)
   - Less efficient for sparse TF-IDF features

### Result Analysis (2 points)

**Do the results match intuition?**

**Yes**, the results align with expectations:

1. **Tree-based models outperform linear models** ✓
   - **Why**: Product pricing exhibits non-linear patterns:
     - Brand effects (iPhone vs generic phone)
     - Category differences (furniture vs food)
     - Interaction effects (size × brand)
   - Linear models cannot capture these complex relationships

2. **LightGBM performs best** ✓
   - **Why**: Optimized for large datasets with sparse features
   - TF-IDF creates high-dimensional sparse matrices (5000 features)
   - LightGBM's histogram-based algorithm handles this efficiently

3. **Moderate hyperparameters work best** ✓
   - **Why**: Balance between:
     - Conservative: Underfits complex price patterns
     - Aggressive: Overfits to training data noise
     - Moderate: Captures true patterns without memorizing noise

**Unexpected Finding**:

[If you observe anything surprising, discuss here. For example:]
- "Random Forest performed worse than expected, possibly due to inefficiency with sparse TF-IDF features"
- "Ridge Regression was surprisingly competitive, suggesting product names have strong linear signals"

**Feature Importance Insights**:

[After running feature importance analysis, note top features:]
- Top features include: [brand names, product categories, size numbers]
- Numbers (e.g., "500", "2kg") have high importance
- Brand-specific terms correlate strongly with price ranges

---

## Conclusion

This project successfully implemented a machine learning pipeline for product price prediction achieving [XX.XX]% SMAPE on validation data. The key insights are:

1. Text preprocessing (especially Chinese tokenization) is critical for feature quality
2. Tree-based models significantly outperform linear models for this task
3. TF-IDF effectively captures price-relevant information from product names
4. LightGBM provides the best balance of accuracy and efficiency

**Future Improvements**:
- Ensemble methods (combine multiple models)
- Deep learning approaches (BERT, transformers)
- Additional feature engineering (extract brands, categories)
- Handling duplicate or similar products

---

## Appendix

### Code Submission
- `main.ipynb`: Complete Jupyter notebook with all code and visualizations
- `train_model.py`: Standalone Python script for pipeline execution
- `submission.csv`: Final predictions for Kaggle submission

### References
- Jieba: https://github.com/fxsjy/jieba
- LightGBM: https://lightgbm.readthedocs.io/
- Scikit-learn TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html

---

**Note**: Fill in [XX.XX] placeholders with actual values from your model runs.

