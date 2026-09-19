# Titanic Data Science Project

## Project Overview

This repository contains a progressive data science project based on Kaggle's **Titanic - Machine Learning from Disaster** dataset.

The project is being developed as part of a Data Science with Python internship and currently covers two complete stages:

* **Week 1 — Data Acquisition, Cleaning, and Preprocessing**
* **Week 2 — Exploratory Data Analysis and Visualization**

The project begins with the original Titanic passenger dataset, identifies and resolves data-quality issues, engineers useful features, produces a cleaned dataset, and then performs detailed exploratory analysis to uncover meaningful patterns and relationships.

The overall workflow is:

```text
Raw Titanic Dataset
        │
        ▼
Week 1
Data Acquisition
        │
        ▼
Initial Exploration
        │
        ▼
Missing-Value Analysis
        │
        ▼
Outlier and Consistency Analysis
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Cleaned Dataset
        │
        ▼
Week 2
Dataset Overview
        │
        ▼
Univariate Analysis
        │
        ▼
Bivariate Analysis
        │
        ▼
Multivariate Analysis
        │
        ▼
Correlation Analysis
        │
        ▼
Interpretation and Insights
```

---

# Dataset Information

The project uses the Titanic dataset from Kaggle's **Titanic - Machine Learning from Disaster** competition.

The primary dataset used throughout the project is:

```text
data/train.csv
```

The repository also contains:

```text
data/test.csv
```

The cleaning and exploratory-analysis workflow uses `train.csv` because it contains the `Survived` variable required for survival analysis.

## Original Dataset Shape

* **Rows:** 891
* **Columns:** 12

## Original Columns

| Column        | Description                       |
| ------------- | --------------------------------- |
| `PassengerId` | Unique passenger identifier       |
| `Survived`    | Survival status: 0 = No, 1 = Yes  |
| `Pclass`      | Passenger class                   |
| `Name`        | Passenger name                    |
| `Sex`         | Passenger sex                     |
| `Age`         | Passenger age                     |
| `SibSp`       | Number of siblings/spouses aboard |
| `Parch`       | Number of parents/children aboard |
| `Ticket`      | Ticket number                     |
| `Fare`        | Passenger fare                    |
| `Cabin`       | Cabin number                      |
| `Embarked`    | Port of embarkation               |

---

# Week 1 — Data Acquisition, Cleaning, and Preprocessing

Week 1 focused on acquiring the public dataset, understanding its structure, identifying data-quality problems, cleaning missing and inconsistent values, handling potential outliers, and creating useful derived features.

The main Week 1 script is:

```text
src/titanic_cleaning.py
```

The main output is:

```text
outputs/titanic_cleaned.csv
```

---

## Week 1 Workflow

```text
Kaggle Titanic Dataset
        │
        ▼
Load Dataset with Pandas
        │
        ▼
Initial Dataset Inspection
        │
        ├── First rows
        ├── Shape
        ├── Column names
        ├── Data types
        ├── Dataset information
        └── Descriptive statistics
        │
        ▼
Data Quality Assessment
        │
        ├── Missing values
        ├── Missing percentages
        ├── Duplicate rows
        ├── Duplicate PassengerIds
        ├── Categorical consistency
        └── Numerical validity checks
        │
        ▼
Outlier Analysis
        │
        ├── Age
        ├── Fare
        ├── SibSp
        └── Parch
        │
        ▼
Data Cleaning
        │
        ├── Age imputation
        ├── Embarked imputation
        ├── Cabin transformation
        └── Fare outlier treatment
        │
        ▼
Feature Engineering
        │
        ├── Cabin_known
        ├── Fare_capped
        ├── FamilySize
        └── IsAlone
        │
        ▼
Final Validation
        │
        ├── Missing-value check
        ├── Duplicate check
        └── Dataset shape verification
        │
        ▼
Save Cleaned Dataset
        │
        ▼
outputs/titanic_cleaned.csv
```

---

## Initial Dataset Exploration

The dataset was first inspected using Pandas to understand its structure and contents.

The initial analysis included:

* first five rows
* dataset dimensions
* column names
* data types
* complete dataset information
* descriptive statistics
* categorical values
* missing-value counts
* duplicate checks

The original dataset contained:

```text
891 rows
12 columns
```

---

## Missing-Value Analysis

Three columns contained missing values:

| Feature    | Missing Values | Percentage |
| ---------- | -------------: | ---------: |
| `Age`      |            177 |     19.87% |
| `Cabin`    |            687 |     77.10% |
| `Embarked` |              2 |      0.22% |

A missing-value visualization was also generated to make the scale of missing data easier to understand.

No other original columns contained missing values.

---

## Duplicate and Consistency Checks

The dataset was checked for:

* complete duplicate rows
* duplicate passenger identifiers
* unexpected categorical values
* invalid numerical values

Results:

* Duplicate rows: **0**
* Duplicate `PassengerId` values: **0**

Categorical variables were also inspected.

Observed values included:

```text
Sex:
male
female
```

```text
Embarked:
S
C
Q
```

```text
Pclass:
1
2
3
```

```text
Survived:
0
1
```

No unexpected categorical values were identified.

---

## Numerical Validation

The numerical variables were checked for invalid negative values.

The following variables were validated:

* `Age`
* `Fare`
* `SibSp`
* `Parch`

No invalid negative values were detected.

This helped distinguish genuine statistical outliers from clearly erroneous data.

---

## Outlier Analysis

Potential outliers were identified using the **1.5 × IQR rule**.

The variables examined were:

* `Age`
* `Fare`
* `SibSp`
* `Parch`

The number of observations flagged was:

| Feature | Flagged Observations |
| ------- | -------------------: |
| `Age`   |                   11 |
| `Fare`  |                  116 |
| `SibSp` |                   46 |
| `Parch` |                  213 |

Boxplots were generated to visualize these distributions.

The flagged observations were not automatically removed because a statistical outlier is not necessarily incorrect data.

This was particularly important for `Parch`, where both the first and third quartiles are zero. As a result, valid non-zero family relationships can be identified as statistical outliers.

---

## Missing-Value Treatment

### Age

Missing `Age` values were filled using the median within groups defined by:

* `Pclass`
* `Sex`

This was selected instead of one overall median because passenger age patterns differ across class and sex groups.

A global median fallback was also included in case any missing values remained.

After imputation:

```text
Missing Age values = 0
```

### Embarked

The two missing `Embarked` values were filled using the mode of the column.

After treatment:

```text
Missing Embarked values = 0
```

### Cabin

`Cabin` contained:

```text
687 missing values
77.10% missing
```

Because most cabin values were unavailable, filling them with artificial cabin numbers would introduce unreliable information.

Instead, a new binary variable called `Cabin_known` was created:

```text
1 = Cabin information is available
0 = Cabin information is unavailable
```

Final counts:

```text
Cabin unknown = 687
Cabin known   = 204
```

The original `Cabin` column was retained for transparency.

---

## Fare Outlier Treatment

`Fare` contained several extreme values.

Using the IQR rule, the upper limit was calculated as:

```text
65.6344
```

Instead of deleting high-fare passengers, the original `Fare` column was preserved and a new feature called `Fare_capped` was created.

Original maximum fare:

```text
512.3292
```

Maximum capped fare:

```text
65.6344
```

This approach reduces the influence of extreme observations while retaining the original data.

A before-and-after boxplot was generated to visualize the effect of capping.

---

## Feature Engineering

Several additional variables were created during preprocessing.

### Cabin_known

Indicates whether cabin information is available.

```text
1 = Known
0 = Unknown
```

### Fare_capped

Contains the fare after limiting values above the upper IQR threshold.

The original `Fare` column remains unchanged.

### FamilySize

Family size was calculated as:

```text
FamilySize = SibSp + Parch + 1
```

The additional `1` represents the passenger.

Summary:

* Minimum family size: **1**
* Maximum family size: **11**
* Mean family size: approximately **1.90**

### IsAlone

A binary variable was created using `FamilySize`.

```text
1 = Passenger travelled alone
0 = Passenger travelled with family
```

Final distribution:

| Travel Status | Count |
| ------------- | ----: |
| Alone         |   537 |
| With family   |   354 |

---

## Week 1 Final Validation

After the cleaning and feature-engineering process, the final dataset was validated.

### Dataset Dimensions

Original dataset:

```text
(891, 12)
```

Cleaned dataset:

```text
(891, 16)
```

### Final Results

* Rows removed: **0**
* Duplicate rows: **0**
* Missing values in analysis-ready features: **0**
* Original source columns retained where useful
* Four useful engineered features created

The cleaned dataset is saved as:

```text
outputs/titanic_cleaned.csv
```

---

## Week 1 Visualizations

Week 1 figures are stored in:

```text
outputs/figures/
```

The cleaning script generates figures including:

```text
missing_values.png
age_boxplot.png
fare_boxplot.png
sibsp_boxplot.png
parch_boxplot.png
fare_before_after.png
survival_by_sex.png
survival_by_class.png
age_distribution.png
confusion_matrix.png
```

Some survival and validation plots were included in the original Week 1 workflow, while the dedicated exploratory analysis was later expanded substantially in Week 2.

---

## Week 1 Key Decisions

Important preprocessing decisions included:

* using grouped medians instead of one overall value for missing Age
* using the mode for the two missing Embarked entries
* avoiding artificial cabin-number imputation
* retaining statistical outliers when they represented plausible passenger data
* preserving the original Fare column
* creating a capped Fare variable instead of deleting high-fare records
* retaining all 891 passengers
* creating family-related and cabin-related features for later analysis

---

## Impact of Week 1 Preprocessing

The cleaning process improved the dataset's suitability for further analysis.

However, the transformations also affect the statistical properties of the data.

For example:

* Age imputation replaces unknown ages with estimates.
* Fare capping reduces the magnitude of extreme fares.
* `Cabin_known` simplifies cabin information into an availability indicator.
* `FamilySize` combines two family-related features into a more interpretable measurement.

The original columns were retained where useful so that preprocessing decisions remain transparent and auditable.

---

# Week 2 — Exploratory Data Analysis and Visualization

Week 2 builds directly on the cleaned dataset generated during Week 1.

The objective is to identify meaningful distributions, trends, associations, and anomalies using Pandas, Matplotlib, and Seaborn.

The Week 2 script is:

```text
src/titanic_eda.py
```

The input is:

```text
outputs/titanic_cleaned.csv
```

The figures are stored in:

```text
outputs/eda_figures/
```

---

## Week 2 Workflow

```text
Cleaned Titanic Dataset
        │
        ▼
Dataset Overview
        │
        ├── Shape
        ├── Columns
        ├── Data types
        ├── Statistics
        ├── Missing values
        └── Duplicate check
        │
        ▼
Univariate Analysis
        │
        ├── Survival
        ├── Passenger class
        ├── Sex
        ├── Age
        ├── Fare
        ├── Embarkation
        ├── Family size
        ├── Travelling status
        └── Cabin availability
        │
        ▼
Bivariate Analysis
        │
        ├── Survival vs Sex
        ├── Survival vs Class
        ├── Survival vs Embarkation
        ├── Survival vs IsAlone
        ├── Survival vs Cabin_known
        ├── Survival vs Age
        ├── Survival vs Fare
        └── Survival vs FamilySize
        │
        ▼
Multivariate Analysis
        │
        ├── Sex + Class + Survival
        ├── Age + Fare + Class + Survival
        └── Class + Cabin + Survival
        │
        ▼
Correlation Analysis
        │
        ▼
Interpretation and Key Insights
```

---

# Week 2 Dataset Overview

The cleaned dataset contains:

```text
891 rows
16 columns
```

The EDA script confirms:

* dataset dimensions
* column names
* data types
* descriptive statistics
* missing values
* duplicate records

The original `Cabin` variable still contains missing values because it is intentionally preserved.

The engineered `Cabin_known` feature is used when analyzing cabin-information availability.

---

# Univariate Analysis

Univariate analysis examines one feature at a time.

Features analyzed include:

* `Survived`
* `Pclass`
* `Sex`
* `Age`
* `Fare_capped`
* `Embarked`
* `FamilySize`
* `IsAlone`
* `Cabin_known`

---

## Survival Distribution

| Survival Status | Passengers | Percentage |
| --------------- | ---------: | ---------: |
| Did not survive |        549 |     61.62% |
| Survived        |        342 |     38.38% |

The dataset contains more non-survivors than survivors.

---

## Passenger Class Distribution

| Class     | Passengers |
| --------- | ---------: |
| 1st Class |        216 |
| 2nd Class |        184 |
| 3rd Class |        491 |

Third-class passengers form the largest passenger group.

---

## Sex Distribution

| Sex    | Passengers |
| ------ | ---------: |
| Male   |        577 |
| Female |        314 |

Male passengers represent the majority of the dataset.

---

## Age Distribution

After Week 1 preprocessing:

* Mean age: **29.11 years**
* Median age: **26 years**
* Minimum age: **0.42 years**
* Maximum age: **80 years**

Most passengers are concentrated among younger and middle-aged adults.

---

## Fare Distribution

Using `Fare_capped`:

* Mean: **24.05**
* Median: **14.45**
* Maximum: **65.6344**

The capped version helps prevent a small number of very large fares from dominating the visualization.

---

## Embarkation Port

| Port        | Passengers | Percentage |
| ----------- | ---------: | ---------: |
| Southampton |        646 |     72.50% |
| Cherbourg   |        168 |     18.86% |
| Queenstown  |         77 |      8.64% |

Southampton was the most common embarkation port.

---

## Family Size

Family size is concentrated at smaller values.

The most common family size is:

```text
1
```

This corresponds to passengers travelling alone.

---

## Travelling Alone

| Travel Status | Passengers | Percentage |
| ------------- | ---------: | ---------: |
| With family   |        354 |     39.73% |
| Alone         |        537 |     60.27% |

Most passengers travelled alone.

---

## Cabin Information Availability

| Cabin Information | Passengers | Percentage |
| ----------------- | ---------: | ---------: |
| Unknown           |        687 |     77.10% |
| Known             |        204 |     22.90% |

Only around one-quarter of passengers had recorded cabin information.

---

# Bivariate Analysis

Bivariate analysis examines the relationship between two variables, with survival used as the main comparison variable.

---

## Survival by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |        74.20% |
| Male   |        18.89% |

Female passengers had a substantially higher observed survival rate.

---

## Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| 1st Class       |        62.96% |
| 2nd Class       |        47.28% |
| 3rd Class       |        24.24% |

Observed survival decreases substantially from 1st class to 3rd class.

---

## Survival by Embarkation Port

| Port        | Survival Rate |
| ----------- | ------------: |
| Cherbourg   |        55.36% |
| Queenstown  |        38.96% |
| Southampton |        33.90% |

Cherbourg passengers had the highest observed survival rate.

However, embarkation should not be interpreted as a direct cause because passenger composition differs across ports.

---

## Survival by Travelling Status

| Travel Status | Survival Rate |
| ------------- | ------------: |
| With family   |        50.56% |
| Alone         |        30.35% |

Passengers travelling with family had a higher observed survival rate.

---

## Survival by Cabin Information

| Cabin Information | Survival Rate |
| ----------------- | ------------: |
| Unknown           |        29.99% |
| Known             |        66.67% |

Passengers with known cabin information showed substantially higher observed survival.

Cabin availability is also strongly related to passenger class.

---

## Age and Survival

| Survival Status | Mean Age | Median Age |
| --------------- | -------: | ---------: |
| Did not survive |    29.74 |         25 |
| Survived        |    28.11 |         27 |

Age distributions overlap considerably.

Age therefore has only a weak overall relationship with survival when analyzed independently.

---

## Fare and Survival

| Survival Status | Mean Fare | Median Fare |
| --------------- | --------: | ----------: |
| Did not survive |     18.92 |       10.50 |
| Survived        |     32.28 |       26.00 |

Surviving passengers generally paid higher fares.

Fare is also related to passenger class.

---

## Survival by Family Size

| Family Size | Count | Survival Rate |
| ----------: | ----: | ------------: |
|           1 |   537 |        30.35% |
|           2 |   161 |        55.28% |
|           3 |   102 |        57.84% |
|           4 |    29 |        72.41% |
|           5 |    15 |        20.00% |
|           6 |    22 |        13.64% |
|           7 |    12 |        33.33% |
|           8 |     6 |         0.00% |
|          11 |     7 |         0.00% |

Moderate family sizes show relatively high observed survival.

Very large family-size groups contain few passengers, so their percentages should be interpreted cautiously.

---

# Multivariate Analysis

Multivariate analysis examines several passenger characteristics simultaneously.

---

## Survival by Passenger Class and Sex

| Passenger Class | Sex    | Count | Survival Rate |
| --------------: | ------ | ----: | ------------: |
|               1 | Female |    94 |        96.81% |
|               1 | Male   |   122 |        36.89% |
|               2 | Female |    76 |        92.11% |
|               2 | Male   |   108 |        15.74% |
|               3 | Female |   144 |        50.00% |
|               3 | Male   |   347 |        13.54% |

Female passengers show higher survival in every passenger class.

Passenger class also remains important within each sex group.

---

## Passenger Class and Cabin Availability

| Class | Cabin Known | Count | Survival Rate |
| ----: | ----------: | ----: | ------------: |
|     1 |          No |    40 |        47.50% |
|     1 |         Yes |   176 |        66.48% |
|     2 |          No |   168 |        44.05% |
|     2 |         Yes |    16 |        81.25% |
|     3 |          No |   479 |        23.59% |
|     3 |         Yes |    12 |        50.00% |

Known-cabin passengers show higher observed survival within each passenger class.

The 2nd- and 3rd-class known-cabin groups contain relatively few passengers, so those percentages should be interpreted carefully.

---

## Age, Fare, Survival, and Passenger Class

A multivariate scatter plot examines:

* Age
* `Fare_capped`
* Survival status
* Passenger class

The visualization shows substantial overlap between groups while also demonstrating the relationship between fare and passenger class.

---

# Correlation Analysis

Pearson correlation was calculated for numerical and engineered features.

## Correlation with Survival

| Feature       | Correlation |
| ------------- | ----------: |
| `Pclass`      |       -0.34 |
| `Fare_capped` |       +0.32 |
| `Cabin_known` |       +0.32 |
| `IsAlone`     |       -0.20 |
| `Parch`       |       +0.08 |
| `Age`         |       -0.06 |
| `SibSp`       |       -0.04 |
| `FamilySize`  |       +0.02 |

Passenger class, fare, and cabin-information availability show some of the strongest linear relationships with survival among the numerical variables.

---

## Relationships Between Features

Several stronger correlations also exist between predictor variables:

```text
Pclass vs Fare_capped       = -0.72
Pclass vs Cabin_known       = -0.73
Fare_capped vs Cabin_known  = +0.62
FamilySize vs SibSp         = +0.89
FamilySize vs Parch         = +0.78
FamilySize vs IsAlone       = -0.69
```

Some of these relationships are expected.

`FamilySize` is directly constructed from `SibSp` and `Parch`, so strong correlations with those features are natural.

Passenger class, fare, and cabin availability also represent related socioeconomic and travel characteristics.

Correlation measures linear association and should not be interpreted as evidence of causation.

---

# Week 2 Visualizations

All Week 2 EDA figures are stored in:

```text
outputs/eda_figures/
```

The EDA script generates **22 visualizations**.

## Univariate Visualizations

* Survival distribution
* Passenger class distribution
* Sex distribution
* Age distribution
* Fare distribution
* Embarkation distribution
* Family-size distribution
* Travelling-alone distribution
* Cabin-information distribution

## Bivariate Visualizations

* Survival by sex
* Survival by passenger class
* Survival by embarkation port
* Survival by travelling status
* Survival by cabin availability
* Age by survival status
* Age distribution by survival status
* Fare by survival status
* Survival by family size

## Multivariate and Correlation Visualizations

* Survival by class and sex
* Age vs fare by survival status and class
* Survival by class and cabin availability
* Correlation heatmap

---

# Key EDA Findings

The Week 2 analysis identified several important patterns:

* **38.38%** of passengers survived.
* Female passengers had substantially higher survival than male passengers.
* 1st-class passengers had much higher survival than 3rd-class passengers.
* Survivors generally paid higher fares.
* Passengers travelling with family had higher survival than solo travellers.
* Moderate family sizes generally had better observed outcomes.
* Passengers with known cabin information showed higher observed survival.
* Age alone had only a weak linear relationship with survival.
* Fare, passenger class, and cabin availability were strongly related.
* Sex and passenger class together produced some of the clearest survival differences.

---

# Interpretation Considerations

The EDA identifies **associations**, not causal relationships.

Several variables are related to one another.

For example:

* Fare is strongly associated with passenger class.
* Cabin availability is associated with passenger class.
* Embarkation groups may contain different passenger-class and sex distributions.
* `FamilySize`, `SibSp`, and `Parch` are mathematically related.

Small subgroups should also be interpreted cautiously.

For example, family sizes of 8 and 11 contain very few passengers, so their survival percentages are less stable than results based on hundreds of observations.

---

# Technologies Used

| Week 1             | Week 2             |
| ------------------ | ------------------ |
| Python             | Python             |
| Pandas             | Pandas             |
| Matplotlib         | Matplotlib         |
| Seaborn            | Seaborn            |
| Scikit-learn       | pathlib            |
| pathlib            | uv                 |
| uv                 | Visual Studio Code |
| Visual Studio Code | Git                |
| Git                | GitHub             |
| GitHub             |                    |

---

# Reports

The final reports are stored in:

```text
reports/
├── Week_1_Data_Cleaning_Report.docx
└── Week_2_Titanic_EDA_Report.docx
```

---

# Project Structure

```text
Titanic_Data_Science_Project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── titanic_cleaning.py
│   └── titanic_eda.py
│
├── outputs/
│   ├── titanic_cleaned.csv
│   │
│   ├── figures/
│   │   └── Week 1 cleaning and preprocessing figures
│   │
│   └── eda_figures/
│       └── Week 2 EDA figures
│
├── reports/
│   ├── Week_1_Data_Cleaning_Report.docx
│   └── Week_2_Titanic_EDA_Report.docx
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/ARraj14/Titanic_Data_Science_Project.git
cd Titanic_Data_Science_Project
```

## 2. Install Dependencies

```bash
uv sync
```

## 3. Run Week 1

```bash
uv run python src/titanic_cleaning.py
```

This generates:

```text
outputs/titanic_cleaned.csv
```

Week 1 figures are saved in:

```text
outputs/figures/
```

## 4. Run Week 2

After generating the cleaned dataset:

```bash
uv run python src/titanic_eda.py
```

Week 2 figures are saved in:

```text
outputs/eda_figures/
```

---

# Challenges and Solutions

## Missing Age Values

Instead of using a single overall median, missing Age values were filled using medians calculated within passenger-class and sex groups.

## Missing Cabin Information

Because most Cabin values were unavailable, cabin numbers were not artificially imputed.

The `Cabin_known` indicator preserves useful information without fabricating cabin assignments.

## Statistical Outliers

The IQR rule was used to identify unusual values, but observations were interpreted in context instead of automatically being removed.

## Extreme Fare Values

The original Fare variable was preserved.

A separate capped Fare feature was created so that extreme fares would have less influence on analysis and visualization.

## Preserving Passenger Records

No passenger rows were removed during data cleaning.

## Interpreting EDA Results

Relationships discovered during EDA were treated as associations rather than evidence of causation.

Sample size was also considered when interpreting small passenger subgroups.

---

# Key Learnings

This project provided practical experience with:

* public dataset acquisition
* dataset exploration
* data-quality assessment
* missing-value analysis
* grouped imputation
* duplicate detection
* categorical consistency checking
* numerical validation
* IQR-based outlier detection
* contextual interpretation of outliers
* feature engineering
* data transformation
* data validation
* univariate analysis
* bivariate analysis
* multivariate analysis
* Pandas aggregation
* correlation analysis
* Matplotlib visualization
* Seaborn visualization
* interpretation of statistical relationships
* reproducible project organization
* dependency management with uv
* Git version control
* GitHub repository management

---

# Conclusion

The Titanic Data Science Project currently demonstrates a complete workflow from raw-data preparation through exploratory analysis.

During **Week 1**, the original Titanic dataset was inspected, cleaned, validated, and transformed. Missing values were handled using variable-specific strategies, statistical outliers were analyzed rather than blindly removed, and useful engineered features such as `Cabin_known`, `Fare_capped`, `FamilySize`, and `IsAlone` were created.

The resulting cleaned dataset retained all **891 passenger records** and increased from **12 original columns to 16 columns**.

During **Week 2**, the cleaned dataset was analyzed using univariate, bivariate, multivariate, and correlation-based techniques. A total of **22 visualizations** were generated to examine distributions and relationships within the dataset.

The analysis identified particularly strong associations between survival and passenger sex, passenger class, fare, travelling status, and cabin-information availability.

Together, Week 1 and Week 2 demonstrate how careful data preparation and exploratory analysis can be combined into a structured, transparent, and reproducible data science workflow.

---

# Dataset Source

**Titanic - Machine Learning from Disaster**
Kaggle

## Project Repository

```text
https://github.com/ARraj14/Titanic_Data_Science_Project
```
