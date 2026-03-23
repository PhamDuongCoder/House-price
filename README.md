# House Price Prediction System

A machine learning project that predicts real estate prices using XGBoost regression with comprehensive data preprocessing and feature engineering.

## 📊 Project Overview

This project implements a **multivariate real estate price prediction system** based on the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The model applies advanced data preprocessing, feature engineering, and hyperparameter tuning to achieve high prediction accuracy.

### Key Results
- **RMSLE Score**: **0.13273** (after hyperparameter tuning)
- **Previous Score**: 0.14043 (baseline without tuning)
- **Data**: 1,460 training samples, 1,459 test samples
- **Features**: 79 engineered features after preprocessing and encoding

## 🏗️ Project Pipeline

### 1. **Data Cleaning** (`Preprocessing/cleaning data.ipynb`)
- **Column Dropping**: Removed 14 low-variance or redundant columns (Utilities, Street, RoofMatl, PoolArea, etc.)
- **Feature Grouping**: Consolidated categorical variables by significance
  - `KitchenAbvGr`: Group ≠1 into 0 (binary: 1 kitchen vs others)
  - `GarageQual/GarageCond`: Group non-'TA' into 'Other'
  - `EnclosedPorch`: Binary feature (0 or 1)
  - `SaleType/SaleCondition`: Group rare types into 'Other'
- **EDA-Driven Approach**: Data-driven decisions backed by boxplots and correlation analysis

### 2. **Missing Value Imputation** (`Preprocessing/fillna.ipynb`)
- **Neighborhood-based filling**: `LotFrontage` filled with neighborhood median
- **Categorical defaults**: Quality/condition columns filled with 'None'
- **Numeric defaults**: Garage/basement features filled with 0
- **Mode-based filling**: `Electrical` filled with neighborhood mode

### 3. **Feature Engineering** (`Preprocessing/Feature_Engineering.ipynb`)
Created 7 new features to capture domain knowledge:
- **`TotalBath`**: Aggregated full and half bathrooms
- **`HouseAge`**: Years since construction (YrSold - YearBuilt)
- **`YrSinceRemod`**: Years since remodeling
- **`Remodeled`**: Binary indicator (was remodeled or not)
- **`MixedExterior`**: Binary indicator (exterior materials match)
- **`SeasonSold`**: Categorical seasonal feature (Winter/Spring/Summer/Fall)
- **`TotalSF`**: Total square footage (1st floor + 2nd floor + basement)
- **`NewHouse`**: Binary indicator (built < 2 years ago)

### 4. **Feature Encoding** (`Preprocessing/Feature_Encoding.ipynb`)
- **Ordinal Encoding**: Quality features mapped to 0-5 scale (None, Po, Fa, TA, Gd, Ex)
- **One-Hot Encoding**: Nominal categorical features encoded with drop_first=True
- **Outlier Removal**: 
  - Removed 5 anomalies: `GrLivArea > 4500` AND `SalePrice < 300000`
  - Removed 1 anomaly: `TotalBsmtSF > 4000`
  - Total removed: 6 outliers (~0.4% of data)

### 5. **Model Training & Hyperparameter Tuning** (`training model.ipynb`)
- **Base Model**: XGBoost Regression with 300 estimators
- **Hyperparameter Search**:
  - Grid search with 5-fold cross-validation
  - Tuned parameters: `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`
  - R² scoring
- **Final Model**: 1000 estimators with early stopping (50 rounds)
- **Regularization**: L1 (0.1) and L2 (1.0) penalties
- **Learning Rate**: 0.05 (conservative learning)

## 📈 Performance Metrics

| Metric | Value | Dataset |
|--------|-------|---------|
| R² Score | 0.9145 | Validation Set |
| Train R² | 0.9875 | Training Set |
| RMSLE | 0.13273 | Kaggle Leaderboard |
| Model | XGBoost | - |

*Note: Minor overfitting (9.7% gap) addressed through regularization.*

## 🗂️ Project Structure

```
House-price/
├── README.md                          # This file
├── training model.ipynb               # Model training and prediction
├── Data/
│   ├── train.csv                      # Training data (1460 samples)
│   ├── test.csv                       # Test data (1459 samples)
│   └── data_description.txt           # Feature descriptions
├── Preprocessing/
│   ├── cleaning data.ipynb            # EDA and column dropping
│   ├── fillna.ipynb                   # Missing value imputation
│   ├── Feature_Engineering.ipynb      # Feature creation
│   ├── Feature_Encoding.ipynb         # Encoding & outlier removal
│   ├── Cleaned_Data.csv               # After cleaning
│   ├── Nan_filled_data.csv            # After imputation
│   ├── Feature_Engineered_train.csv   # After feature engineering
│   └── Feature_Encoded_data.csv       # Final encoded data
├── test/                              # Experimental notebooks (optional)
└── submission.csv                     # Predictions on test set
```

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Cross-validation, GridSearchCV, preprocessing
- **XGBoost**: Gradient boosting regression model
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### Run the Pipeline
1. **Data Cleaning**: Execute `Preprocessing/cleaning data.ipynb`
2. **Imputation**: Execute `Preprocessing/fillna.ipynb`
3. **Feature Engineering**: Execute `Preprocessing/Feature_Engineering.ipynb`
4. **Encoding**: Execute `Preprocessing/Feature_Encoding.ipynb`
5. **Model Training**: Execute `training model.ipynb`

### Make Predictions on New Data
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_houseprice_predictor.joblib')

# Prepare your data (must follow same preprocessing pipeline)
X_new = preprocess_and_engineer(new_data)

# Generate predictions
predictions = model.predict(X_new)
```

## 💡 Key Insights & Decisions

### Why These Features Were Dropped
- **Utilities**: 99.3% of homes had 'AllPub' → no variance
- **PoolArea/PoolQC**: <3% of homes with pools → sparse and low predictive power
- **ScreenPorch**: Similar SalePrice distribution regardless of value
- **MiscFeature/MiscVal**: Weak correlation with target variable

### Why These Features Were Engineered
- **TotalBath**: Improves bathroom count representation
- **HouseAge**: Captures structural depreciation over time
- **Remodeled**: Indicates potential value restoration
- **SeasonSold**: Seasonal market variations impact prices
- **TotalSF**: Better representation of living space than multiple columns

### Outlier Removal Justification
Model investigated GrLivArea vs SalePrice and identified 5 clear anomalies (houses >4500 sqft selling <$300k), which are likely data entry errors or special sales. These were removed to prevent model degradation.

## 📊 Model Validation Strategy

- **Train-Test Split**: 80-20 with random_state=42 (reproducibility)
- **Cross-Validation**: 5-fold CV during hyperparameter search
- **Early Stopping**: 50 rounds to prevent overfitting
- **Regularization**: Both L1 and L2 penalties applied

## 🔮 Future Improvements

1. **Feature Scaling**: Normalize features before encoding for other models
2. **Ensemble Methods**: Combine XGBoost with LightGBM/CatBoost
3. **Advanced Feature Engineering**: Polynomial features, interaction terms
4. **Hyperparameter Optimization**: Bayesian optimization vs grid search
5. **Stacking/Blending**: Combine multiple models
6. **Logging**: Implement experiment tracking (MLflow/Weights&Biases)

## 📝 Notes

- All preprocessing is reproducible with fixed random states
- The same transformations must be applied to test data
- Model is saved in `best_houseprice_predictor.joblib` for production use
- Pipeline designed to prevent data leakage

## 👨‍💻 Author

Created as a Kaggle competition project demonstrating ML fundamentals: EDA, data cleaning, feature engineering, and model tuning.

## 📜 License

This project uses the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) - Educational use only.

---

**Last Updated**: March 2026
