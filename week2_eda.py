"""
Week 2 Task: Exploratory Data Analysis and Visualization
Virtual Data Science with Python Trainee
Dataset: Titanic - Machine Learning from Disaster
Input: titanic_cleaned.csv (prepared in Week 1)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("titanic_cleaned.csv")
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe(include="all").T)

# Overall survival
print("\nOverall survival rate:", round(df["Survived"].mean() * 100, 2), "%")

# Grouped analysis
print("\nSurvival rate by sex:")
print(df.groupby("Sex")["Survived"].mean().mul(100).round(2))

print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass")["Survived"].mean().mul(100).round(2))

# Visualization 1: survival counts
labels = df["Survived"].map({0: "Did not survive", 1: "Survived"})
counts = labels.value_counts().reindex(["Did not survive", "Survived"])
fig, ax = plt.subplots()
bars = ax.bar(counts.index, counts.values)
ax.set_title("Titanic Survival Outcome Counts")
ax.set_xlabel("Outcome")
ax.set_ylabel("Number of passengers")
ax.bar_label(bars)
fig.tight_layout()
fig.savefig(CHART_DIR / "01_survival_counts.png", dpi=180)
plt.close(fig)

# Visualization 2: survival by sex
rates = df.groupby("Sex")["Survived"].mean().mul(100)
fig, ax = plt.subplots()
bars = ax.bar(rates.index, rates.values)
ax.set_title("Survival Rate by Sex")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 100)
ax.bar_label(bars, fmt="%.1f%%")
fig.tight_layout()
fig.savefig(CHART_DIR / "02_survival_by_sex.png", dpi=180)
plt.close(fig)

# Visualization 3: survival by class
rates = df.groupby("Pclass")["Survived"].mean().mul(100)
fig, ax = plt.subplots()
bars = ax.bar(rates.index.astype(str), rates.values)
ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 100)
ax.bar_label(bars, fmt="%.1f%%")
fig.tight_layout()
fig.savefig(CHART_DIR / "03_survival_by_class.png", dpi=180)
plt.close(fig)

# Visualization 4: age distribution
fig, ax = plt.subplots()
ax.hist(df["Age"].dropna(), bins=20)
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Number of passengers")
fig.tight_layout()
fig.savefig(CHART_DIR / "04_age_distribution.png", dpi=180)
plt.close(fig)

# Visualization 5: fare by class
groups = [df.loc[df["Pclass"] == c, "Fare"].dropna() for c in [1, 2, 3]]
fig, ax = plt.subplots()
ax.boxplot(groups, labels=["1", "2", "3"], showfliers=False)
ax.set_title("Fare Distribution by Passenger Class")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Fare")
fig.tight_layout()
fig.savefig(CHART_DIR / "05_fare_by_class_boxplot.png", dpi=180)
plt.close(fig)

# Visualization 6: correlation matrix
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare",
             "CabinKnown", "FamilySize", "IsAlone"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
ax.set_title("Correlation Matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(CHART_DIR / "06_correlation_heatmap.png", dpi=180)
plt.close(fig)

print("\nEDA complete. Charts saved in:", CHART_DIR)
