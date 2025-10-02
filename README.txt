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
