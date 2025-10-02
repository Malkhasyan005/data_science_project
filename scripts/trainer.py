import sys
import yaml
import logging
import numpy as np
from pathlib import Path
import ruamel.yaml
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
import xgboost as xgb


from src.data_processing.data_loader import FreshRetailDataLoader
from src.data_processing.feature_engineering import FeatureEngineer

from src.models.baseline.linear_models import LinearForecastingModel
from src.models.baseline.tree_models import TreeForecastingModel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main(config_path="config/config.yaml", selected_model=None):
    # ------------------------------
    # Load config
    # ------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------
    # Load and prepare data
    # ------------------------------
    data_loader = FreshRetailDataLoader(config_path)
    train_data, eval_data = data_loader.load_data()

    target_col = config["data"]["target_column"]
    metadata_cols = ["store_id", "product_id", "city_id", "dt", target_col, 'hours_sale', 'hours_stock_status','day_part']

    feature_engineer = FeatureEngineer(config["preprocessing"])
    train_data = feature_engineer.engineer_all_features(train_data)

    X = train_data.drop(columns=metadata_cols)
    X = X.drop(columns=X.columns[X.isna().all()])
    y = train_data[target_col]


    # ------------------------------
    # Define models + param grids
    # ------------------------------
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=5000),
        "random_forest": RandomForestRegressor(random_state=config["training"]["random_state"]),
        "gradient_boosting": GradientBoostingRegressor(random_state=config["training"]["random_state"]),
        "xg_boost": xgb.XGBRegressor(random_state=config["training"]["random_state"], n_jobs=-1)
    }

    param_grids = {
        "linear": {"model__fit_intercept": [True, False]},
        "ridge": {"model__alpha": [0.1, 1.0, 10.0], "model__fit_intercept": [True, False]},
        "lasso": {"model__alpha": [0.01, 0.1, 1.0], "model__fit_intercept": [True, False]},
        "random_forest": {"model__n_estimators": [100, 300], "model__max_depth": [10, 20, None]},
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5],
        },
        'xg_boost': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
    }

    # Filter for selected model if provided
    if selected_model:
        if selected_model not in models:
            logger.error(f"Model '{selected_model}' not recognized. Available models: {list(models.keys())}")
            return
        model = models[selected_model]
        param_grid = param_grids[selected_model]

    # ------------------------------
    # Run grid search for each model
    # ------------------------------

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Fill NaNs
        ('scaler', StandardScaler()),                  # Optional scaling
        ('model', model)
    ])

    logger.info(f"Running GridSearchCV for {selected_model}...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=config["training"]["cv_folds"],
        n_jobs=-1,
    )
    grid.fit(X, y)

    mean_score = -grid.best_score_  # MAE
    logger.info(f"{selected_model} best MAE: {mean_score:.4f} with params {grid.best_params_}")

    best_params = grid.best_params_

    # ------------------------------
    # Save all best params into config
    # ------------------------------
    # Ensure keys exist
    config.setdefault("models", {})
    config["models"].setdefault("baseline", {})
    config["models"]["baseline"].setdefault(selected_model, {})

    # Remove 'model__' prefix from GridSearchCV keys
    best_params_clean = {key.replace("model__", ""): value for key, value in grid.best_params_.items()}
    
    yaml_loader = ruamel.yaml.YAML()
    with open(config_path, "r") as f:
        config = yaml_loader.load(f)

    for key, value in best_params_clean.items():
        config["models"]["baseline"][selected_model][key] = value

    # Write back preserving formatting
    with open(config_path, "w") as f:
        yaml_loader.dump(config, f)

    logger.info(f"Updated {config_path} with best parameters for selected model(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retail models with GridSearchCV")
    parser.add_argument("--model", type=str, help="Name of the model to run (run all if not provided)")
    args = parser.parse_args()

    main(selected_model=args.model)
