# Heart Disease Prediction using Machine Learning

## Overview

This project applies a complete machine learning pipeline to predict the presence of heart disease using clinical data. The workflow includes data preprocessing, exploratory analysis, feature encoding, and classification using two machine learning algorithms: **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)**.

The goal is to analyze the relationships between medical features and heart disease and evaluate the predictive performance of different models.

---

# Dataset

The dataset contains medical attributes related to heart disease diagnosis, including:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG results
* Maximum heart rate
* Exercise-induced angina
* ST depression
* ST slope
* Heart disease label (target variable)

Target variable:

```
HeartDisease
0 = No heart disease
1 = Presence of heart disease
```

---

# Project Workflow

## 1. Data Loading

The dataset is loaded from Google Drive and converted into a Pandas DataFrame.

```python
df = pd.read_csv('/content/drive/MyDrive/heart.csv')
```

---

# 2. Data Exploration

Initial exploration includes:

* Displaying the dataset
* Inspecting data types
* Visualizing the distribution of heart disease cases

Example visualization:

* Count plot of heart disease cases using **Seaborn**

---

# 3. Missing Value Handling

Missing values are handled using **SimpleImputer**:

### Numeric Features

Missing values replaced with the **mean**.

### Categorical Features

Missing values replaced with the **most frequent category**.

The cleaned dataset is saved as:

```
heart_cleaned.csv
```

---

# 4. Correlation Analysis

A full correlation matrix is computed using different techniques depending on variable types:

| Feature Type               | Method Used            |
| -------------------------- | ---------------------- |
| Numeric vs Numeric         | Pearson Correlation    |
| Categorical vs Categorical | Cramér's V             |
| Numeric vs Categorical     | Correlation Ratio (η²) |

The results are visualized using a **heatmap**.

---

# 5. Feature Encoding

Different encoding strategies are applied depending on feature type.

### Standardization

Numeric features are normalized using **StandardScaler**.

### One-Hot Encoding

Applied to nominal categorical variables:

* Sex
* ChestPainType
* RestingECG
* ExerciseAngina

### Ordinal Encoding

Applied to ordered variable:

```
ST_Slope
Up < Flat < Down
```

---

# 6. Machine Learning Models

Two classification models are implemented.

## Support Vector Machine (SVM)

Hyperparameters optimized using **GridSearchCV**.

Parameters tested:

```
C = [0.1, 1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1]
kernel = RBF
```

Evaluation metrics:

* Accuracy
* Confusion Matrix
* ROC Curve
* AUC Score

---

## K-Nearest Neighbors (KNN)

Hyperparameters optimized using **GridSearchCV**.

Parameters tested:

```
n_neighbors = [3,5,7,9,11]
weights = ['uniform','distance']
metric = ['euclidean','manhattan']
```

Evaluation metrics:

* Accuracy
* Confusion Matrix
* ROC Curve
* AUC Score

---

# Model Evaluation

Models are evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **ROC Curve**
* **AUC Score**

These metrics help assess classification performance and compare models.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

---

# Results

Both models demonstrate the ability to classify heart disease cases based on medical features. Hyperparameter tuning using **GridSearchCV** improves model performance and ensures better generalization.

---

# Future Improvements

Possible future improvements include:

* Testing additional models (Random Forest, XGBoost)
* Performing feature selection
* Applying cross-validation with larger folds
* Improving class imbalance handling
* Building a deployment interface


