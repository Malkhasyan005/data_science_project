
# Data Science Project – Demand Forecasting

![Python](https://img.shields.io/badge/python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-green)

A **demand forecasting project** demonstrating end-to-end machine learning pipelines with linear models, tree-based models, and XGBoost.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project aims to forecast product demand using various machine learning algorithms. It includes:

- Data preprocessing (handling missing values, scaling, feature selection)
- Model training and evaluation
- Feature importance analysis for interpretability
- Support for multiple models:
  - Linear models (Linear Regression, Ridge, Lasso, ElasticNet)
  - Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
  - XGBoost

---

## Dataset
- Source: Add your dataset source here (e.g., Kaggle, internal dataset)  
- Number of samples: `Insert here`  
- Features: `List key features`  
- Target variable: `sale_amount` (or your target)  

> ⚠️ Note: Some evaluation metrics like MAPE may be unreliable if the target contains zeros.

---

## Installation
Clone the repository:

```bash
git clone https://github.com/Malkhasyan005/data_science_project.git
cd data_science_project
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Usage
### Train a model
Example using XGBoost:

```bash
python scripts/trainer.py --model_type xgboost --config configs/xgboost_config.yaml
```

### Predict on new data

```python
from src.models.xgboost_model import XGBoostForecastingModel

# Initialize model with config
model = XGBoostForecastingModel(config=config)

# Train model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)
```

### Evaluate model

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
```

---

## Models
| Model Type           | Description |
|---------------------|-------------|
| Linear Regression    | Baseline linear model |
| Ridge, Lasso, ElasticNet | Regularized linear models for feature selection and multicollinearity |
| Decision Tree        | Single tree for interpretability |
| Random Forest        | Ensemble of trees for robust performance |
| Gradient Boosting    | Sequential tree ensemble |
| XGBoost              | High-performance gradient boosting |

---

## Evaluation
Models are evaluated using:

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **MAPE** – Mean Absolute Percentage Error  
- **Bias** – Average prediction bias  

Example XGBoost metrics:

```
MAE: 0.1146
RMSE: 0.7242
MAPE: 12,075,085.89%  # Highly sensitive if target has zeros
Bias: -0.0577
```

---

## Feature Importance
Tree-based models and XGBoost provide feature importance for insights:

```python
importance = model.get_feature_importance_detailed()
print(importance['top_10_features'])
```

Use this to identify which features most influence predictions.

---

## Contributing
Contributions are welcome:

1. Fork the repository  
2. Create a branch: `git checkout -b feature-name`  
3. Commit your changes: `git commit -m "Add feature"`  
4. Push to the branch: `git push origin feature-name`  
5. Open a pull request  

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.