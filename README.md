# Titanic Data Cleaning and Preprocessing

## Project Overview

This project focuses on cleaning and preprocessing the Titanic dataset from Kaggle's **Titanic - Machine Learning from Disaster** competition.

The objective is to identify and handle missing values, detect potential outliers and inconsistencies, engineer useful features, and prepare the dataset for further analysis and machine learning.

The project demonstrates a complete workflow using Python, Pandas, Matplotlib, Seaborn, and Scikit-learn. A Logistic Regression model is used as a downstream validation step to confirm that the processed data can be used successfully in a machine-learning pipeline.

## Dataset Information

The main dataset used is `train.csv` from the **Titanic - Machine Learning from Disaster** Kaggle competition. The repository also retains `test.csv`, but the current cleaning script and model evaluation use only `train.csv` because it contains the `Survived` target.

### Original Dataset Shape

- Rows: 891
- Columns: 12

### Columns

- `PassengerId` - Unique passenger identifier
- `Survived` - Survival status where 0 = No and 1 = Yes
- `Pclass` - Passenger class
- `Name` - Passenger name
- `Sex` - Passenger sex
- `Age` - Passenger age
- `SibSp` - Number of siblings/spouses aboard
- `Parch` - Number of parents/children aboard
- `Ticket` - Ticket number
- `Fare` - Passenger fare
- `Cabin` - Cabin number
- `Embarked` - Port of embarkation

### Missing Values Identified

- `Age` - 177 missing values (19.87%)
- `Cabin` - 687 missing values (77.10%)
- `Embarked` - 2 missing values (0.22%)

No duplicate rows or duplicate `PassengerId` values were found.

## Data Cleaning and Preprocessing

### 1. Missing Values

#### Age
Missing `Age` values were filled using the median within groups defined by `Pclass` and `Sex`. A global median fallback is also present in the script in case any values remain missing.

#### Embarked
The 2 missing `Embarked` values were filled using the mode.

#### Cabin
Because `Cabin` contains a large amount of missing data, cabin numbers were not imputed. Instead, a binary feature named `Cabin_known` was created:

- `1` - cabin information is available
- `0` - cabin information is missing

### 2. Outlier Analysis and Treatment

Potential outliers were identified using the 1.5 x IQR rule for:

- `Age`
- `Fare`
- `SibSp`
- `Parch`

The number of values flagged was:

| Feature | Flagged observations |
|---|---:|
| Age | 11 |
| Fare | 116 |
| SibSp | 46 |
| Parch | 213 |

These values were not automatically treated as errors. For `Fare`, the original values were retained and a separate `Fare_capped` feature was created using the upper IQR limit of **65.6344**.

### 3. Feature Engineering

#### FamilySize

`FamilySize = SibSp + Parch + 1`

The additional 1 represents the passenger themself.

#### IsAlone

- `1` - passenger travelled alone
- `0` - passenger travelled with family

The final data contains 537 passengers with `IsAlone = 1` and 354 with `IsAlone = 0`.

### 4. Machine-Learning Preprocessing

The final model uses 10 features:

- `Pclass`
- `Age`
- `SibSp`
- `Parch`
- `Fare_capped`
- `Cabin_known`
- `FamilySize`
- `IsAlone`
- `Sex`
- `Embarked`

A Scikit-learn `ColumnTransformer` and `Pipeline` are used so that:

- numerical features receive defensive median imputation and `StandardScaler`
- categorical features receive defensive most-frequent imputation and `OneHotEncoder`
- Logistic Regression is trained after preprocessing

The scaling and encoding steps are fitted on the training split and then applied consistently to the test split.

> **Technical note:** grouped Age medians, the Embarked mode, and the Fare cap threshold are currently calculated before the train/test split. For strict predictive benchmarking, these learned statistics could be moved into train-fitted custom transformers in a future version.

## Exploratory Data Analysis and Visualizations

The script generates the following figures in `outputs/figures/`:

- `missing_values.png`
- `age_boxplot.png`
- `fare_boxplot.png`
- `sibsp_boxplot.png`
- `parch_boxplot.png`
- `fare_before_after.png`
- `survival_by_sex.png`
- `survival_by_class.png`
- `age_distribution.png`
- `confusion_matrix.png`

### Missing Values

![Missing Values](outputs/figures/missing_values.png)

### Fare Before and After Capping

![Fare Before and After Capping](outputs/figures/fare_before_after.png)

### Survival Rate by Sex

![Survival Rate by Sex](outputs/figures/survival_by_sex.png)

### Survival Rate by Passenger Class

![Survival Rate by Passenger Class](outputs/figures/survival_by_class.png)

### Age Distribution

![Age Distribution](outputs/figures/age_distribution.png)

## Machine Learning Validation

The dataset was split using an 80:20 stratified train/test split with `random_state=42`.

- Training samples: 712
- Testing samples: 179

A Logistic Regression classifier was used as a downstream validation model.

### Accuracy

**79.89%**

### Classification Report Summary

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 - Did not survive | 0.82 | 0.86 | 0.84 | 110 |
| 1 - Survived | 0.76 | 0.70 | 0.73 | 69 |
| Macro average | 0.79 | 0.78 | 0.78 | 179 |
| Weighted average | 0.80 | 0.80 | 0.80 | 179 |

### Confusion Matrix

```text
[[95 15]
 [21 48]]
```

![Confusion Matrix](outputs/figures/confusion_matrix.png)

The model is used mainly to confirm that the cleaned and preprocessed feature set can successfully pass through a complete machine-learning workflow. The primary objective of the project is data cleaning and preprocessing rather than maximizing prediction accuracy.

## Final Dataset Summary

- Original shape: `(891, 12)`
- Cleaned shape: `(891, 16)`
- Rows removed: 0
- Missing values in final model features: 0
- Duplicate rows in cleaned dataset: 0
- Final model feature count: 10
- Cleaned file: `outputs/titanic_cleaned.csv`

## Project Structure

```text
Titanic_Data_Cleaning_Project/
|
├── data/
│   ├── train.csv
│   └── test.csv
│
├── outputs/
│   ├── titanic_cleaned.csv
│   └── figures/
│       ├── missing_values.png
│       ├── age_boxplot.png
│       ├── fare_boxplot.png
│       ├── sibsp_boxplot.png
│       ├── parch_boxplot.png
│       ├── fare_before_after.png
│       ├── survival_by_sex.png
│       ├── survival_by_class.png
│       ├── age_distribution.png
│       └── confusion_matrix.png
│
├── src/
│   └── titanic_cleaning.py
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

The local `src/titanic_cleaning_backup.py` file is excluded from Git tracking through `.gitignore` and therefore is not part of the published repository.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- uv
- Visual Studio Code
- Git and GitHub

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ARraj14/Titanic_Data_Cleaning_Project.git
cd Titanic_Data_Cleaning_Project
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Project

```bash
uv run python src/titanic_cleaning.py
```

The cleaned dataset will be saved to:

```text
outputs/titanic_cleaned.csv
```

The generated visualizations will be saved to:

```text
outputs/figures/
```

## Challenges and Solutions

### Handling Missing Age Values

Rather than using one overall median, missing ages were filled using medians within passenger class and sex groups.

### Handling Missing Cabin Information

Because 687 cabin values were missing, direct cabin-number imputation was avoided. `Cabin_known` preserves whether cabin information was present without inventing values.

### Handling Statistical Outliers

The IQR rule was used for detection, but statistical flags were interpreted in context rather than automatically deleted. This is especially important for `Parch`, where Q1 and Q3 are both 0 and therefore every non-zero value is flagged.

### Handling Extreme Fare Values

Instead of deleting high-fare passenger records, the original `Fare` was retained and a separate capped feature was created.

### Consistent Model Preprocessing

`Pipeline` and `ColumnTransformer` ensure that scaling and categorical encoding learned from the training split are applied consistently to the test split.

## Impact of Preprocessing

The preprocessing steps remove missing values from the final machine-learning features and convert the data into a form suitable for Logistic Regression. At the same time, some transformations alter the statistical properties of the data. Age imputation replaces unknown values with estimates, and fare capping changes the magnitude of high fares. The original source columns are therefore retained where useful to keep the transformations transparent and auditable.

## Key Learnings

This project provided practical experience with:

- dataset exploration and quality assessment
- missing-value analysis and imputation
- IQR-based outlier detection
- distinguishing statistical outliers from invalid data
- feature engineering
- data visualization
- train/test splitting and stratification
- `ColumnTransformer` and Scikit-learn pipelines
- Logistic Regression validation
- accuracy, precision, recall, F1-score, and confusion matrices
- organizing a reproducible Python project with uv and GitHub

## Conclusion

This project demonstrates a complete data-cleaning and preprocessing workflow on the Titanic training dataset. Missing values were handled with variable-specific strategies, potential outliers were interpreted rather than blindly removed, useful features were engineered, and a structured Scikit-learn pipeline was used for downstream validation.

The final cleaned dataset contains 891 rows and 16 columns, with no missing values in the 10 selected model features and no duplicate rows. The Logistic Regression validation model achieved **79.89%** accuracy on the test split.

## Dataset Source

**Titanic - Machine Learning from Disaster** - Kaggle

Project repository: https://github.com/ARraj14/Titanic_Data_Cleaning_Project
