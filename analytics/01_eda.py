
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Load Titanic dataset exactly once
# --------------------------------------------------

titanic = sns.load_dataset("titanic")
titanic.to_csv("titanic.csv", index=False)

# --------------------------------------------------
# 2. Inspect dataset
# --------------------------------------------------

print("Dataset shape:", titanic.shape)
print("Columns:", titanic.columns.tolist())

print("\nMissing values:")
print(titanic.isnull().sum())

# --------------------------------------------------
# 3. Missing-value percentage analysis
# --------------------------------------------------

missing = titanic.isnull().sum()
missing_percent = (missing / len(titanic)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing,
    "Missing Percentage": missing_percent.round(2)
})

print("\nMissing-value analysis:")
print(missing_table[missing_table["Missing Count"] > 0])

# --------------------------------------------------
# 4. Clean missing values using threshold rules
# --------------------------------------------------

cleaned_titanic = titanic.copy()

# >30% missing: drop column
cleaned_titanic = cleaned_titanic.drop(columns=["deck"])

# 5%-30% missing: median imputation
cleaned_titanic["age"] = cleaned_titanic["age"].fillna(
    cleaned_titanic["age"].median()
)

# <5% missing: drop affected rows
cleaned_titanic = cleaned_titanic.dropna(
    subset=["embarked", "embark_town"]
)

cleaned_titanic.to_csv(
    "cleaned_titanic.csv",
    index=False
)

print("\nOriginal shape:", titanic.shape)
print("Cleaned shape:", cleaned_titanic.shape)

# --------------------------------------------------
# 5. Age EDA
# --------------------------------------------------

print("\nAge statistics:")
print(cleaned_titanic["age"].describe())

print("Age median:", cleaned_titanic["age"].median())
print("Age mean:", cleaned_titanic["age"].mean())
print("Age standard deviation:", cleaned_titanic["age"].std())

# --------------------------------------------------
# 6. Fare EDA
# --------------------------------------------------

fare_mean = cleaned_titanic["fare"].mean()
fare_median = cleaned_titanic["fare"].median()
fare_mode = cleaned_titanic["fare"].mode()[0]
fare_skewness = cleaned_titanic["fare"].skew()

print("\nFare statistics:")
print("Mean:", round(fare_mean, 2))
print("Median:", round(fare_median, 2))
print("Mode:", round(fare_mode, 2))
print("Skewness:", round(fare_skewness, 2))

# --------------------------------------------------
# 7. IQR outlier analysis
# --------------------------------------------------

def iqr_outlier_summary(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    print(f"\n{column.upper()} OUTLIER ANALYSIS")
    print("Q1:", round(q1, 2))
    print("Q3:", round(q3, 2))
    print("IQR:", round(iqr, 2))
    print("Lower bound:", round(lower_bound, 2))
    print("Upper bound:", round(upper_bound, 2))
    print("Number of outliers:", len(outliers))
    print(
        "Outlier percentage:",
        round(len(outliers) / len(data) * 100, 2),
        "%"
    )

iqr_outlier_summary(cleaned_titanic, "age")
iqr_outlier_summary(cleaned_titanic, "fare")

# --------------------------------------------------
# 8. Survival analysis using boolean masks
# --------------------------------------------------

female_mask = cleaned_titanic["sex"] == "female"
male_mask = cleaned_titanic["sex"] == "male"

print("\nSurvival rate by sex:")
print(
    "Female:",
    round(
        cleaned_titanic.loc[female_mask, "survived"].mean() * 100,
        2
    ),
    "%"
)
print(
    "Male:",
    round(
        cleaned_titanic.loc[male_mask, "survived"].mean() * 100,
        2
    ),
    "%"
)

for passenger_class in [1, 2, 3]:
    class_mask = cleaned_titanic["pclass"] == passenger_class
    survival_rate = (
        cleaned_titanic.loc[class_mask, "survived"].mean() * 100
    )

    print(
        f"{passenger_class}st/nd/rd Class:",
        round(survival_rate, 2),
        "%"
    )

# Sex + passenger class
print("\nSurvival rate by sex and passenger class:")

for sex in ["female", "male"]:
    for passenger_class in [1, 2, 3]:
        mask = (
            (cleaned_titanic["sex"] == sex) &
            (cleaned_titanic["pclass"] == passenger_class)
        )

        survival_rate = (
            cleaned_titanic.loc[mask, "survived"].mean() * 100
        )

        print(
            f"{sex.title()}, Class {passenger_class}:",
            round(survival_rate, 2),
            "%"
        )

# --------------------------------------------------
# 9. Correlation analysis
# --------------------------------------------------

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = cleaned_titanic[
    correlation_columns
].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(3))

corr_pairs = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

strongest_correlations = (
    corr_pairs.abs()
    .stack()
    .sort_values(ascending=False)
    .head(2)
)

print("\nTwo strongest absolute correlations:")

for (column1, column2), value in strongest_correlations.items():
    original_value = correlation_matrix.loc[
        column1,
        column2
    ]

    print(
        f"{column1} vs {column2}: "
        f"{original_value:.3f}"
    )

# --------------------------------------------------
# 10. Create chart folder
# --------------------------------------------------

Path("charts").mkdir(exist_ok=True)

# --------------------------------------------------
# 11. Age distribution
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    cleaned_titanic["age"],
    bins=30,
    kde=True
)

plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig(
    "charts/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 12. Fare distribution
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    cleaned_titanic["fare"],
    bins=30,
    kde=True
)

plt.title("Titanic Passenger Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig(
    "charts/fare_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 13. Survival rate by sex
# --------------------------------------------------

survival_by_sex = (
    cleaned_titanic
    .groupby("sex")["survived"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=survival_by_sex.index,
    y=survival_by_sex.values
)

plt.title("Titanic Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "charts/survival_by_sex.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 14. Survival rate by passenger class
# --------------------------------------------------

survival_by_class = (
    cleaned_titanic
    .groupby("pclass")["survived"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=survival_by_class.index,
    y=survival_by_class.values
)

plt.title("Titanic Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(
    [0, 1, 2],
    ["1st Class", "2nd Class", "3rd Class"]
)
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "charts/survival_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 15. Survival rate by sex and class
# --------------------------------------------------

survival_by_sex_class = (
    cleaned_titanic
    .groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .reset_index()
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=survival_by_sex_class,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Titanic Survival Rate by Sex and Passenger Class"
)
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(
    [0, 1, 2],
    ["1st Class", "2nd Class", "3rd Class"]
)
plt.ylim(0, 100)
plt.legend(title="Sex")
plt.tight_layout()

plt.savefig(
    "charts/survival_by_sex_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 16. Correlation heatmap
# --------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "charts/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 17. Z-score standardization
# --------------------------------------------------

from sklearn.preprocessing import StandardScaler

standardization_data = cleaned_titanic[
    ["age", "fare"]
].copy()

print("\nBefore standardization:")
print(
    "Age mean:",
    round(standardization_data["age"].mean(), 2)
)
print(
    "Age standard deviation:",
    round(standardization_data["age"].std(), 2)
)
print(
    "Fare mean:",
    round(standardization_data["fare"].mean(), 2)
)
print(
    "Fare standard deviation:",
    round(standardization_data["fare"].std(), 2)
)

scaler = StandardScaler()

standardized_values = scaler.fit_transform(
    standardization_data
)

standardized_data = pd.DataFrame(
    standardized_values,
    columns=["age", "fare"]
)

print("\nAfter standardization:")
print(
    "Age mean:",
    round(standardized_data["age"].mean(), 2)
)
print(
    "Age standard deviation:",
    round(standardized_data["age"].std(), 2)
)
print(
    "Fare mean:",
    round(standardized_data["fare"].mean(), 2)
)
print(
    "Fare standard deviation:",
    round(standardized_data["fare"].std(), 2)
)
