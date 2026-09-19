# Titanic Dataset - Exploratory Data Analysis and Visualization

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "outputs" / "titanic_cleaned.csv"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_figures"

EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

CLASS_ORDER = [1, 2, 3]
CLASS_LABELS = ["1st Class", "2nd Class", "3rd Class"]
EMBARKED_ORDER = ["S", "C", "Q"]
EMBARKED_LABELS = ["Southampton", "Cherbourg", "Queenstown"]


# Helper functions


def save_plot(filename: str) -> None:
    """Save the current figure, display it, and close it."""
    plt.tight_layout()
    plt.savefig(
        EDA_OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def add_count_labels(ax) -> None:
    """Add count labels above bars."""
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)


def add_percentage_labels(ax, decimals: int = 1) -> None:
    """Add percentage labels above bars."""
    for container in ax.containers:
        labels = [f"{bar.get_height():.{decimals}f}%" for bar in container]
        ax.bar_label(container, labels=labels, padding=3)


def survival_rate_table(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Return passenger count and survival rate for one or more grouping columns."""
    table = (
        df.groupby(group_columns, observed=False)["Survived"]
        .agg(count="count", Survival_Rate="mean")
        .reset_index()
    )

    table["Survival_Rate"] = (table["Survival_Rate"] * 100).round(2)
    return table


# Data loading and preparation


def load_data() -> pd.DataFrame:
    """Load the cleaned Titanic dataset."""
    df = pd.read_csv(DATA_PATH)

    df["Survival_Status"] = df["Survived"].map(
        {
            0: "Did Not Survive",
            1: "Survived",
        }
    )

    df["Passenger_Class"] = df["Pclass"].map(
        {
            1: "1st Class",
            2: "2nd Class",
            3: "3rd Class",
        }
    )

    print("Titanic cleaned dataset loaded successfully.")
    return df


# Initial dataset overview


def dataset_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nFirst five rows:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).head())

    print("\nDataset shape:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).shape)

    print("\nColumn names:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).columns.tolist())

    print("\nData types:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).dtypes)

    print("\nDataset information:")
    df.drop(columns=["Survival_Status", "Passenger_Class"]).info()

    print("\nStatistical summary:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).describe())

    print("\nMissing values:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).isnull().sum())

    print("\nNumber of duplicate rows:")
    print(df.drop(columns=["Survival_Status", "Passenger_Class"]).duplicated().sum())


# Univariate analysis


def univariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)

    # 1. Survival distribution
    survival_counts = df["Survived"].value_counts().sort_index()
    survival_percentages = (
        df["Survived"].value_counts(normalize=True).sort_index().mul(100)
    )

    print("\nSurvival counts:")
    print(survival_counts)

    print("\nSurvival percentages:")
    print(survival_percentages.round(2))

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x="Survival_Status",
        order=["Did Not Survive", "Survived"],
    )
    ax.set_title("Distribution of Titanic Passenger Survival")
    ax.set_xlabel("Survival Status")
    ax.set_ylabel("Number of Passengers")
    add_count_labels(ax)
    save_plot("survival_distribution.png")

    # 2. Passenger class distribution
    print("\nPassenger class distribution:")
    print(df["Pclass"].value_counts().sort_index())

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x="Pclass",
        order=CLASS_ORDER,
    )
    ax.set_title("Distribution of Passengers by Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticks(range(3), CLASS_LABELS)
    add_count_labels(ax)
    save_plot("passenger_class_distribution.png")

    # 3. Sex distribution
    print("\nGender distribution:")
    print(df["Sex"].value_counts())

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Sex")
    ax.set_title("Distribution of Passengers by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Number of Passengers")
    add_count_labels(ax)
    save_plot("gender_distribution.png")

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
    save_plot("age_distribution.png")

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
    save_plot("fare_distribution.png")

    # 6. Embarkation port distribution
    embarked_counts = df["Embarked"].value_counts()
    embarked_percentages = df["Embarked"].value_counts(normalize=True).mul(100)

    print("\nEmbarkation port counts:")
    print(embarked_counts)

    print("\nEmbarkation port percentages:")
    print(embarked_percentages.round(2))

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x="Embarked",
        order=EMBARKED_ORDER,
    )
    ax.set_title("Distribution of Passengers by Embarkation Port")
    ax.set_xlabel("Embarkation Port")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticks(range(3), EMBARKED_LABELS)
    add_count_labels(ax)
    save_plot("embarkation_distribution.png")

    # 7. Family size distribution
    family_size_counts = df["FamilySize"].value_counts().sort_index()

    print("\nFamily size distribution:")
    print(family_size_counts)

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="FamilySize",
        order=sorted(df["FamilySize"].unique()),
    )
    ax.set_title("Distribution of Passenger Family Size")
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Number of Passengers")
    add_count_labels(ax)
    save_plot("family_size_distribution.png")

    # 8. Travelling alone distribution
    alone_counts = df["IsAlone"].value_counts().sort_index()
    alone_percentages = df["IsAlone"].value_counts(normalize=True).sort_index().mul(100)

    print("\nTravelling alone counts:")
    print(alone_counts)

    print("\nTravelling alone percentages:")
    print(alone_percentages.round(2))

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x="IsAlone",
        order=[0, 1],
    )
    ax.set_title("Passengers Travelling Alone vs With Family")
    ax.set_xlabel("Travel Status")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticks([0, 1], ["With Family", "Alone"])
    add_count_labels(ax)
    save_plot("travelling_alone_distribution.png")

    # 9. Cabin information availability
    cabin_counts = df["Cabin_known"].value_counts().sort_index()
    cabin_percentages = (
        df["Cabin_known"].value_counts(normalize=True).sort_index().mul(100)
    )

    print("\nCabin information availability:")
    print(cabin_counts)

    print("\nCabin information percentages:")
    print(cabin_percentages.round(2))

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x="Cabin_known",
        order=[0, 1],
    )
    ax.set_title("Availability of Passenger Cabin Information")
    ax.set_xlabel("Cabin Information")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticks([0, 1], ["Unknown", "Known"])
    add_count_labels(ax)
    save_plot("cabin_information_distribution.png")


# Bivariate analysis


def bivariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("BIVARIATE ANALYSIS")
    print("=" * 60)

    # 1. Survival rate by sex
    survival_by_sex = survival_rate_table(df, ["Sex"])

    print("\nSurvival rate by sex (%):")
    print(survival_by_sex.to_string(index=False))

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=survival_by_sex,
        x="Sex",
        y="Survival_Rate",
    )
    ax.set_title("Survival Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    add_percentage_labels(ax)
    save_plot("survival_by_sex.png")

    # 2. Survival rate by passenger class
    survival_by_class = survival_rate_table(df, ["Pclass"])

    print("\nSurvival rate by passenger class (%):")
    print(survival_by_class.to_string(index=False))

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=survival_by_class,
        x="Pclass",
        y="Survival_Rate",
        order=CLASS_ORDER,
    )
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(3), CLASS_LABELS)
    add_percentage_labels(ax)
    save_plot("survival_by_class.png")

    # 3. Survival rate by embarkation port
    survival_by_embarked = survival_rate_table(df, ["Embarked"])

    print("\nSurvival rate by embarkation port (%):")
    print(survival_by_embarked.to_string(index=False))

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=survival_by_embarked,
        x="Embarked",
        y="Survival_Rate",
        order=EMBARKED_ORDER,
    )
    ax.set_title("Survival Rate by Embarkation Port")
    ax.set_xlabel("Embarkation Port")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(3), EMBARKED_LABELS)
    add_percentage_labels(ax)
    save_plot("survival_by_embarkation.png")

    # 4. Survival rate by travel status
    survival_by_alone = survival_rate_table(df, ["IsAlone"])

    print("\nSurvival rate by travelling status (%):")
    print(survival_by_alone.to_string(index=False))

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=survival_by_alone,
        x="IsAlone",
        y="Survival_Rate",
        order=[0, 1],
    )
    ax.set_title("Survival Rate: Alone vs With Family")
    ax.set_xlabel("Travel Status")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1], ["With Family", "Alone"])
    add_percentage_labels(ax)
    save_plot("survival_by_travel_status.png")

    # 5. Survival rate by cabin information availability
    survival_by_cabin = survival_rate_table(df, ["Cabin_known"])

    print("\nSurvival rate by cabin information availability (%):")
    print(survival_by_cabin.to_string(index=False))

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=survival_by_cabin,
        x="Cabin_known",
        y="Survival_Rate",
        order=[0, 1],
    )
    ax.set_title("Survival Rate by Cabin Information Availability")
    ax.set_xlabel("Cabin Information")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1], ["Unknown", "Known"])
    add_percentage_labels(ax)
    save_plot("survival_by_cabin_information.png")

    # 6. Age and survival
    age_by_survival = (
        df.groupby("Survived")["Age"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )

    print("\nAge statistics by survival status:")
    print(age_by_survival)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Survival_Status",
        y="Age",
        order=["Did Not Survive", "Survived"],
    )
    plt.title("Age Distribution by Survival Status")
    plt.xlabel("Survival Status")
    plt.ylabel("Age")
    save_plot("age_by_survival.png")

    # 7. Age distribution by survival status
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="Age",
        hue="Survival_Status",
        hue_order=["Did Not Survive", "Survived"],
        bins=30,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Age Distribution by Survival Status")
    plt.xlabel("Age")
    plt.ylabel("Density")
    save_plot("age_survival_distribution.png")

    # 8. Fare and survival
    fare_by_survival = (
        df.groupby("Survived")["Fare_capped"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )

    print("\nFare statistics by survival status:")
    print(fare_by_survival)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Survival_Status",
        y="Fare_capped",
        order=["Did Not Survive", "Survived"],
    )
    plt.title("Fare Distribution by Survival Status")
    plt.xlabel("Survival Status")
    plt.ylabel("Capped Fare")
    save_plot("fare_by_survival.png")

    # 9. Survival rate by family size
    survival_by_family_size = survival_rate_table(df, ["FamilySize"])

    print("\nSurvival rate by family size:")
    print(survival_by_family_size.to_string(index=False))

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=survival_by_family_size,
        x="FamilySize",
        y="Survival_Rate",
    )
    ax.set_title("Survival Rate by Family Size")
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    add_percentage_labels(ax)
    save_plot("survival_by_family_size.png")


# Multivariate analysis


def multivariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 60)

    # 1. Survival by passenger class and sex
    survival_sex_class = survival_rate_table(
        df,
        ["Pclass", "Sex"],
    )

    print("\nSurvival rate by passenger class and sex:")
    print(survival_sex_class.to_string(index=False))

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=survival_sex_class,
        x="Pclass",
        y="Survival_Rate",
        hue="Sex",
        order=CLASS_ORDER,
    )
    ax.set_title("Survival Rate by Passenger Class and Sex")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(3), CLASS_LABELS)
    ax.legend(title="Sex")
    add_percentage_labels(ax)
    save_plot("survival_by_class_and_sex.png")

    # 2. Age and fare relationship by survival and passenger class
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Age",
        y="Fare_capped",
        hue="Survival_Status",
        style="Passenger_Class",
        alpha=0.7,
    )
    plt.title("Age vs Fare by Survival Status and Passenger Class")
    plt.xlabel("Age")
    plt.ylabel("Capped Fare")
    plt.legend(
        title="Survival / Class",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    save_plot("age_fare_survival_class.png")

    # 3. Survival by passenger class and cabin availability
    survival_class_cabin = survival_rate_table(
        df,
        ["Pclass", "Cabin_known"],
    )

    survival_class_cabin["Cabin_Status"] = survival_class_cabin["Cabin_known"].map(
        {0: "Unknown", 1: "Known"}
    )

    print("\nSurvival rate by class and cabin availability:")
    print(
        survival_class_cabin[
            ["Pclass", "Cabin_known", "count", "Survival_Rate"]
        ].to_string(index=False)
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=survival_class_cabin,
        x="Pclass",
        y="Survival_Rate",
        hue="Cabin_Status",
        order=CLASS_ORDER,
    )
    ax.set_title("Survival Rate by Passenger Class and Cabin Information")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(3), CLASS_LABELS)
    ax.legend(title="Cabin Information")
    add_percentage_labels(ax)
    save_plot("survival_class_cabin.png")

    # 4. Correlation analysis
    correlation_columns = [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare_capped",
        "Cabin_known",
        "FamilySize",
        "IsAlone",
    ]

    correlation_matrix = df[correlation_columns].corr()

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(2))

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Correlation Heatmap of Titanic Features")
    save_plot("correlation_heatmap.png")


# Main program


def main() -> None:
    df = load_data()

    dataset_overview(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Figures saved to: {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
