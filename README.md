# House Prices Prediction using TensorFlow Decision Forests

## Kaggle score: 0.13661
## Project Overview

This project predicts house prices using the **Kaggle House Prices: Advanced Regression Techniques** dataset. Multiple tree-based models from **TensorFlow Decision Forests (TF-DF)** were trained and evaluated to estimate house sale prices.

The project includes:

* Data preprocessing
* Missing value handling
* Feature engineering
* Model training using TensorFlow Decision Forests
* Performance comparison
* Final Kaggle submission generation

---

## Repository Structure

```text
├── README.md
├── housing_prices (1).ipynb
├── train.csv
├── test.csv
├── output.csv
└── requirements.txt
```

### Files Description

| File                       | Description                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `housing_prices (1).ipynb` | Main notebook containing preprocessing, training, evaluation, and prediction pipeline |
| `train.csv`                | Training dataset containing features and target (`SalePrice`)                         |
| `test.csv`                 | Test dataset used for Kaggle prediction                                               |
| `output.csv`               | Final submission file generated for Kaggle                                            |
| `requirements.txt`         | Required dependencies                                                                 |
| `README.md`                | Project documentation                                                                 |

---

## Dataset

Dataset used:

**Kaggle House Prices: Advanced Regression Techniques**

The dataset contains **79 explanatory variables** describing residential homes.

### Target Variable

`SalePrice`

The target was transformed using logarithmic scaling:

```python
np.log1p(SalePrice)
```

This helps reduce skewness and improves regression performance.

---

## Data Preprocessing

### 1. Removed Unnecessary Columns

The following columns were removed due to extremely high missing values or low usefulness:

* `Utilities`
* `MiscFeature`

Additionally:

* `Id` column removed from training data
* `Id` preserved separately for Kaggle submission

---

### 2. Feature Engineering

A new feature was created:

#### `HasPool`

Indicates whether a house has a pool.

```python
x['HasPool'] = x['PoolQC'].notna().astype(int)
```
*`PoolQC` was removed after that
---

### 3. Train-Validation Split

Dataset was split into:

* **80% Training**
* **20% Validation**

```python
train_test_split(
    x,
    test_size=0.20,
    random_state=44
)
```

---

## Models Used

Three regression models from **TensorFlow Decision Forests** were trained and compared.

### 1. CART Model

A single decision tree regressor.

### 2. Random Forest

Ensemble of multiple decision trees to improve generalization.

### 3. Gradient Boosted Trees

Sequential boosting of trees for better prediction accuracy.

---

## Model Performance

| Model                  | Training RMSE | Validation RMSE |
| ---------------------- | ------------- | --------------- |
| CART Model             | 0.1386        | 0.1957          |
| Random Forest          | 0.0735        | 0.1447          |
| Gradient Boosted Trees | **0.0425**    | **0.1391**      |

### Best Model

**Gradient Boosted Trees** achieved the lowest validation RMSE and was selected for final predictions.

---


```python
np.expm1(predictions)
```
## Export Kaggle submission file

---

## Output

The final prediction file generated:

```text
output.csv
```

Format:

| Id   | SalePrice |
| ---- | --------- |
| 1461 | 118301.89 |
| 1462 | 163050.82 |

This file is ready for **Kaggle submission**.

---

## License

This project is licensed under the terms specified in the `LICENSE` file.
