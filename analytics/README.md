
# Module 2 - Titanic Analytics and Machine Learning

## Overview

This module performs exploratory data analysis, data cleaning, classification,
regression, model evaluation, imbalance handling, hyperparameter tuning, and
model persistence using the Titanic dataset.

The dataset is loaded with `sns.load_dataset("titanic")` exactly once and is
immediately saved as `titanic.csv` as an offline fallback.

## Dataset and Cleaning

The original Titanic dataset contains 891 rows and 15 columns.

Missing-value handling follows the required threshold rules:

- Less than 5% missing: drop affected rows.
- 5% to 30% missing: impute the affected values.
- More than 30% missing: drop the column.

Results:

- `age`: 19.87% missing, so median imputation was applied.
- `embarked`: 0.22% missing, so affected rows were removed.
- `embark_town`: 0.22% missing, so affected rows were removed.
- `deck`: 77.22% missing, so the column was removed.

The cleaned dataset contains 889 rows and 14 columns.

## Exploratory Data Analysis

Age and fare distributions were analyzed using descriptive statistics and
visualizations.

IQR analysis identified:

- Age outliers: 65 observations (7.31%).
- Fare outliers: 114 observations (12.82%).

The fare distribution is positively skewed, with the mean above the median,
indicating that a relatively small number of high-fare observations influence
the average.

## Survival Analysis

Survival rates were calculated using boolean masks and grouped analysis.

Observed survival rates:

- Female passengers: 74.04%.
- Male passengers: 18.89%.
- 1st class: 62.62%.
- 2nd class: 47.28%.
- 3rd class: 24.24%.

The combined sex and passenger-class analysis shows substantial differences
between groups. Female passengers in 1st and 2nd class had particularly high
observed survival rates, while male passengers in 2nd and 3rd class had much
lower observed survival rates.

## Correlation Analysis

The required six-column correlation matrix contains:

- survived
- pclass
- age
- sibsp
- parch
- fare

The two strongest absolute correlations were:

1. `pclass` vs `fare`: -0.548
2. `pclass` vs `age`: -0.337

These are associations in the dataset and should not be interpreted as causal
relationships.

## Standardization

Z-score standardization was demonstrated for `age` and `fare`.

Before standardization:

- Age mean: approximately 29.32
- Age standard deviation: approximately 12.98
- Fare mean: approximately 32.10
- Fare standard deviation: approximately 49.70

After standardization, both variables have approximately zero mean and unit
standard deviation.

## Classification

The target variable is `survived`.

Classification features:

- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

A stratified 80/20 train-test split was used.

All preprocessing is performed through scikit-learn pipelines so that
transformations are learned from training data and then applied to test data.

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Balanced Logistic Regression
- SMOTE Logistic Regression
- Tuned Random Forest

## Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8090 | 0.7656 | 0.7206 | 0.7424 | 0.8196 |
| Balanced Logistic Regression | 0.7921 | 0.7183 | 0.7500 | 0.7338 | 0.8612 |
| SMOTE Logistic Regression | 0.7978 | 0.7353 | 0.7353 | 0.7353 | 0.8666 |
| Tuned Random Forest | 0.8315 | 0.8654 | 0.6618 | 0.7500 | 0.8389 |

The tuned Random Forest used:

- `n_estimators = 200`
- `max_depth = 5`
- `max_features = sqrt`

Its 5-fold cross-validation F1 score was 0.7408 and its OOB score was 0.8214.

## Imbalance Handling

Class imbalance was evaluated using:

- Logistic Regression with class weights
- SMOTE applied only to the training data

Balanced Logistic Regression achieved 75.00% recall.

SMOTE Logistic Regression achieved 86.66% ROC-AUC.

These experiments demonstrate that changing class treatment can improve recall
or ranking performance while changing other evaluation metrics.

## Regression

The regression task predicts `fare` using the other available Titanic features.

Linear Regression was evaluated using:

- MAE
- RMSE
- R-squared
- Adjusted R-squared

Results:

| Model | MAE | RMSE | R-squared | Adjusted R-squared |
|---|---:|---:|---:|---:|
| Linear Regression | 21.0986 | 41.7021 | 0.3482 | 0.3091 |

The residual plot shows that prediction errors are not evenly distributed
across all predicted fare values, suggesting possible non-constant residual
variance. This indicates potential heteroscedasticity.

## Deployment Recommendation

The tuned Random Forest is used as the final saved classification pipeline
because it achieved 83.15% test accuracy and a 75.00% F1 score, with 86.54%
precision. Its recall was 66.18%, so the model can still miss some positive
cases. SMOTE Logistic Regression achieved the highest ROC-AUC at 86.66%, while
Balanced Logistic Regression achieved 75.00% recall. Therefore, the operational
metric should be considered when selecting a model for a particular use case.

## Model Persistence

The complete fitted pipeline is saved as:

`models/best_model_pipeline.joblib`

The saved pipeline includes preprocessing and the tuned Random Forest model.

The pipeline was reloaded and tested using raw input containing:

- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

This confirms that preprocessing and prediction can be performed together
without manually transforming the input.

## Project Architecture

analytics/
├── 01_eda.py
├── 02_modeling.py
├── README.md
├── requirements.txt
├── titanic.csv
├── cleaned_titanic.csv
├── model_comparison.csv
├── charts/
│   ├── age_distribution.png
│   ├── fare_distribution.png
│   ├── survival_by_sex.png
│   ├── survival_by_class.png
│   ├── survival_by_sex_class.png
│   ├── correlation_heatmap.png
│   ├── decision_tree.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── regression_residuals.png
└── models/
    └── best_model_pipeline.joblib

## How to Run

Install dependencies:

pip install -r requirements.txt

Run EDA:

python 01_eda.py

Run modeling:

python 02_modeling.py

The scripts generate the analysis outputs, charts, model comparison results,
and saved model pipeline.

## Reproducibility

A fixed `random_state=42` is used for train/test splitting and relevant
machine-learning models.

Classification uses a stratified train/test split.

Preprocessing is contained within pipelines to prevent test-data leakage.

SMOTE is applied inside an imbalanced-learn pipeline so that oversampling is
performed only during model fitting.
