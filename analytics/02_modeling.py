
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

cleaned_titanic = pd.read_csv("cleaned_titanic.csv")

Path("charts").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Classification train/test split
# --------------------------------------------------

y = cleaned_titanic["survived"]

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = cleaned_titanic[classification_features].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Classification preprocessing
# --------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 4. Helper function for classification evaluation
# --------------------------------------------------

def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Confusion Matrix": confusion_matrix(
            y_test,
            predictions
        ),
        "Predictions": predictions,
        "Probabilities": probabilities
    }

# --------------------------------------------------
# 5. Logistic Regression
# --------------------------------------------------

logistic_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_pipeline.fit(X_train, y_train)

logistic_results = evaluate_classifier(
    logistic_pipeline,
    X_test,
    y_test
)

# --------------------------------------------------
# 6. Decision Tree
# --------------------------------------------------

decision_tree_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])

decision_tree_pipeline.fit(X_train, y_train)

decision_tree_results = evaluate_classifier(
    decision_tree_pipeline,
    X_test,
    y_test
)

# --------------------------------------------------
# 7. Random Forest
# --------------------------------------------------

random_forest_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_pipeline.fit(X_train, y_train)

random_forest_results = evaluate_classifier(
    random_forest_pipeline,
    X_test,
    y_test
)

# --------------------------------------------------
# 8. Decision Tree visualization
# --------------------------------------------------

tree_model = decision_tree_pipeline.named_steps["model"]
tree_preprocessor = decision_tree_pipeline.named_steps["preprocess"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classifier")
plt.tight_layout()

plt.savefig(
    "charts/decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 9. Confusion matrices
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

confusion_matrices = [
    ("Logistic Regression", logistic_results["Confusion Matrix"]),
    ("Decision Tree", decision_tree_results["Confusion Matrix"]),
    ("Random Forest", random_forest_results["Confusion Matrix"])
]

for ax, (model_name, cm) in zip(
    axes,
    confusion_matrices
):
    ax.imshow(cm)

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

plt.suptitle("Classification Confusion Matrices")
plt.tight_layout()

plt.savefig(
    "charts/confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 10. ROC curves
# --------------------------------------------------

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_results["Probabilities"]
)

decision_tree_fpr, decision_tree_tpr, _ = roc_curve(
    y_test,
    decision_tree_results["Probabilities"]
)

random_forest_fpr, random_forest_tpr, _ = roc_curve(
    y_test,
    random_forest_results["Probabilities"]
)

plt.figure(figsize=(9, 7))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_results['ROC-AUC']:.3f})"
)

plt.plot(
    decision_tree_fpr,
    decision_tree_tpr,
    label=f"Decision Tree (AUC = {decision_tree_results['ROC-AUC']:.3f})"
)

plt.plot(
    random_forest_fpr,
    random_forest_tpr,
    label=f"Random Forest (AUC = {random_forest_results['ROC-AUC']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curves - Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    "charts/roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 11. Class-weight balancing
# --------------------------------------------------

balanced_logistic_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

balanced_logistic_pipeline.fit(
    X_train,
    y_train
)

balanced_logistic_results = evaluate_classifier(
    balanced_logistic_pipeline,
    X_test,
    y_test
)

# --------------------------------------------------
# 12. SMOTE
# --------------------------------------------------

smote_logistic_pipeline = ImbPipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

smote_logistic_pipeline.fit(
    X_train,
    y_train
)

smote_logistic_results = evaluate_classifier(
    smote_logistic_pipeline,
    X_test,
    y_test
)

# --------------------------------------------------
# 13. Random Forest OOB evaluation
# --------------------------------------------------

random_forest_oob_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    ))
])

random_forest_oob_pipeline.fit(
    X_train,
    y_train
)

oob_score = (
    random_forest_oob_pipeline
    .named_steps["model"]
    .oob_score_
)

print("Random Forest OOB Score:", round(oob_score, 4))

# --------------------------------------------------
# 14. Random Forest GridSearchCV
# --------------------------------------------------

rf_grid_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=True
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

rf_grid_search = GridSearchCV(
    estimator=rf_grid_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid_search.fit(
    X_train,
    y_train
)

best_rf_pipeline = rf_grid_search.best_estimator_

best_rf_results = evaluate_classifier(
    best_rf_pipeline,
    X_test,
    y_test
)

best_rf_oob_score = (
    best_rf_pipeline
    .named_steps["model"]
    .oob_score_
)

print("\nBest Random Forest Parameters:")
print(rf_grid_search.best_params_)

print(
    "Best Cross-Validation F1:",
    round(rf_grid_search.best_score_, 4)
)

print(
    "Best Random Forest OOB:",
    round(best_rf_oob_score, 4)
)

# --------------------------------------------------
# 15. Regression: predict fare
# --------------------------------------------------

y_regression = cleaned_titanic["fare"]

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_regression = cleaned_titanic[
    regression_features
].copy()

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression,
    y_regression,
    test_size=0.20,
    random_state=42
)

regression_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked"
]

regression_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

regression_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

regression_preprocessor = ColumnTransformer([
    (
        "numeric",
        regression_numeric_pipeline,
        regression_numeric_features
    ),
    (
        "categorical",
        regression_categorical_pipeline,
        regression_categorical_features
    )
])

regression_pipeline = Pipeline([
    ("preprocess", regression_preprocessor),
    ("model", LinearRegression())
])

regression_pipeline.fit(
    X_train_reg,
    y_train_reg
)

regression_predictions = regression_pipeline.predict(
    X_test_reg
)

regression_mae = mean_absolute_error(
    y_test_reg,
    regression_predictions
)

regression_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        regression_predictions
    )
)

regression_r2 = r2_score(
    y_test_reg,
    regression_predictions
)

transformed_regression_data = (
    regression_pipeline
    .named_steps["preprocess"]
    .transform(X_test_reg)
)

n = len(y_test_reg)
p = transformed_regression_data.shape[1]

regression_adjusted_r2 = (
    1
    - (1 - regression_r2)
    * (n - 1)
    / (n - p - 1)
)

print("\nRegression Results:")
print("MAE:", round(regression_mae, 4))
print("RMSE:", round(regression_rmse, 4))
print("R²:", round(regression_r2, 4))
print(
    "Adjusted R²:",
    round(regression_adjusted_r2, 4)
)

# --------------------------------------------------
# 16. Regression residual analysis
# --------------------------------------------------

regression_residuals = (
    y_test_reg -
    regression_predictions
)

plt.figure(figsize=(9, 6))

plt.scatter(
    regression_predictions,
    regression_residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residuals vs Predicted Fare")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.tight_layout()

plt.savefig(
    "charts/regression_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 17. Final comparison table
# --------------------------------------------------

final_comparison = pd.DataFrame([
    {
        "Group": "Classification",
        "Model": "Logistic Regression",
        "Accuracy": logistic_results["Accuracy"],
        "Precision": logistic_results["Precision"],
        "Recall": logistic_results["Recall"],
        "F1": logistic_results["F1"],
        "ROC-AUC": logistic_results["ROC-AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R²": np.nan,
        "Adjusted R²": np.nan
    },
    {
        "Group": "Classification",
        "Model": "Decision Tree",
        "Accuracy": decision_tree_results["Accuracy"],
        "Precision": decision_tree_results["Precision"],
        "Recall": decision_tree_results["Recall"],
        "F1": decision_tree_results["F1"],
        "ROC-AUC": decision_tree_results["ROC-AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R²": np.nan,
        "Adjusted R²": np.nan
    },
    {
        "Group": "Classification",
        "Model": "Random Forest",
        "Accuracy": random_forest_results["Accuracy"],
        "Precision": random_forest_results["Precision"],
        "Recall": random_forest_results["Recall"],
        "F1": random_forest_results["F1"],
        "ROC-AUC": random_forest_results["ROC-AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R²": np.nan,
        "Adjusted R²": np.nan
    },
    {
        "Group": "Classification",
        "Model": "Balanced Logistic Regression",
        "Accuracy": balanced_logistic_results["Accuracy"],
        "Precision": balanced_logistic_results["Precision"],
        "Recall": balanced_logistic_results["Recall"],
        "F1": balanced_logistic_results["F1"],
        "ROC-AUC": balanced_logistic_results["ROC-AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R²": np.nan,
        "Adjusted R²": np.nan
    },
    {
        "Group": "Classification",
        "Model": "SMOTE Logistic Regression",
        "Accuracy": smote_logistic_results["Accuracy"],
        "Precision": smote_logistic_results["Precision"],
        "Recall": smote_logistic_results["Recall"],
        "F1": smote_logistic_results["F1"],
        "ROC-AUC": smote_logistic_results["ROC-AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R²": np.nan,
        "Adjusted R²": np.nan
    },
    {
        "Group": "Classification",
        "Model": "Tuned Random Forest",
        "Accuracy": best_rf_results["Accuracy"],
        "Precision": best_rf_results["Precision"],
        "Recall": best_rf_results["Recall"],
        "F1": best_rf_results["F1"],
        "ROC-AUC": best_rf_results["ROC-AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R²": np.nan,
        "Adjusted R²": np.nan
    },
    {
        "Group": "Regression",
        "Model": "Linear Regression",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "ROC-AUC": np.nan,
        "MAE": regression_mae,
        "RMSE": regression_rmse,
        "R²": regression_r2,
        "Adjusted R²": regression_adjusted_r2
    }
])

final_comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nFinal comparison:")
print(final_comparison.round(4))

# --------------------------------------------------
# 18. Save final fitted pipeline
# --------------------------------------------------

full_pipeline = best_rf_pipeline

joblib.dump(
    full_pipeline,
    "models/best_model_pipeline.joblib"
)

print(
    "\nFinal fitted pipeline saved to "
    "models/best_model_pipeline.joblib"
)

# --------------------------------------------------
# 19. Reload and test saved pipeline
# --------------------------------------------------

loaded_pipeline = joblib.load(
    "models/best_model_pipeline.joblib"
)

raw_input = X_test.head(5).copy()

raw_predictions = loaded_pipeline.predict(
    raw_input
)

print("\nReloaded pipeline predictions:")
print(raw_predictions)
