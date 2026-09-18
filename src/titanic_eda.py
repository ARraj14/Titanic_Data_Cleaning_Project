from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "outputs" / "titanic_cleaned.csv"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_figures"

EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset

df = pd.read_csv(DATA_PATH)

print("Titanic cleaned dataset loaded successfully.")

# Initial dataset overview

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

# Statistical summary

print("\nStatistical summary:")
print(df.describe())

# Data quality verification

print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Univariate Analysis

print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

# 1. Survival distribution

survival_counts = df["Survived"].value_counts().sort_index()
survival_percentages = df["Survived"].value_counts(normalize=True).sort_index() * 100

print("\nSurvival counts:")
print(survival_counts)

print("\nSurvival percentages:")
print(survival_percentages.round(2))


plt.figure(figsize=(7, 5))

ax = sns.countplot(data=df, x="Survived")

plt.title("Distribution of Titanic Passenger Survival")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "survival_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 2. Passenger class distribution

print("\nPassenger class distribution:")
print(df["Pclass"].value_counts().sort_index())


plt.figure(figsize=(7, 5))

ax = sns.countplot(data=df, x="Pclass")

plt.title("Distribution of Passengers by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks([0, 1, 2], ["1st Class", "2nd Class", "3rd Class"])

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "passenger_class_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 3. Gender distribution

print("\nGender distribution:")
print(df["Sex"].value_counts())


plt.figure(figsize=(7, 5))

ax = sns.countplot(data=df, x="Sex")

plt.title("Distribution of Passengers by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "gender_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 4. Age distribution

print("\nAge summary:")
print(df["Age"].describe())

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=30,
    kde=True,
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "age_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 5. Fare distribution

print("\nFare summary after outlier capping:")
print(df["Fare_capped"].describe())


plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Fare_capped",
    bins=30,
    kde=True,
)

plt.title("Distribution of Passenger Fares After Outlier Capping")
plt.xlabel("Capped Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "fare_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 6. Embarkation port distribution

embarked_counts = df["Embarked"].value_counts()

embarked_percentages = df["Embarked"].value_counts(normalize=True) * 100

print("\nEmbarkation port counts:")
print(embarked_counts)

print("\nEmbarkation port percentages:")
print(embarked_percentages.round(2))


plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x="Embarked",
    order=["S", "C", "Q"],
)

plt.title("Distribution of Passengers by Embarkation Port")
plt.xlabel("Embarkation Port")
plt.ylabel("Number of Passengers")

plt.xticks(
    [0, 1, 2],
    ["Southampton", "Cherbourg", "Queenstown"],
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "embarkation_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 7. Family size distribution

family_size_counts = df["FamilySize"].value_counts().sort_index()

print("\nFamily size distribution:")
print(family_size_counts)


plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="FamilySize",
)

plt.title("Distribution of Passenger Family Size")
plt.xlabel("Family Size")
plt.ylabel("Number of Passengers")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "family_size_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 8. Travelling alone distribution

alone_counts = df["IsAlone"].value_counts().sort_index()

alone_percentages = df["IsAlone"].value_counts(normalize=True).sort_index() * 100

print("\nTravelling alone counts:")
print(alone_counts)

print("\nTravelling alone percentages:")
print(alone_percentages.round(2))


plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x="IsAlone",
)

plt.title("Passengers Travelling Alone vs With Family")
plt.xlabel("Travel Status")
plt.ylabel("Number of Passengers")

plt.xticks(
    [0, 1],
    ["With Family", "Alone"],
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "travelling_alone_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

# 9. Cabin information availability

cabin_counts = df["Cabin_known"].value_counts().sort_index()

cabin_percentages = df["Cabin_known"].value_counts(normalize=True).sort_index() * 100

print("\nCabin information availability:")
print(cabin_counts)

print("\nCabin information percentages:")
print(cabin_percentages.round(2))


plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x="Cabin_known",
)

plt.title("Availability of Passenger Cabin Information")
plt.xlabel("Cabin Information")
plt.ylabel("Number of Passengers")

plt.xticks(
    [0, 1],
    ["Unknown", "Known"],
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    EDA_OUTPUT_DIR / "cabin_information_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()
