import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from csv_loader import CSVDataLoader
from utils import detect_outliers_z_score

df = CSVDataLoader("./weatherkit_plus_load.csv")
X, y = df.prepare_features()

outliers = detect_outliers_z_score(y)
y = y[~y.index.isin(outliers.index)]
X = X.loc[y.index]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "n_estimators": 300,  # Fixed, but can be tuned further
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
    )

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred)

    return mae + rmse + mape  # minimize MAE, RMSE, MAPE combined


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)


print("Best parameters found: ", study.best_params)
print("Best RMSE: ", study.best_value)


best_params = study.best_params
best_params["objective"] = "regression"
best_params["metric"] = "rmse"


best_model = lgb.train(best_params, lgb.Dataset(X_train, label=y_train))


final_predictions = best_model.predict(X_val, num_iteration=best_model.best_iteration)


print("\n Best Parameters Found:")

for k, v in best_params.items():
    print(f"{k}: {v}")


final_rmse = mean_squared_error(y_val, final_predictions) ** 0.5
print(f"Final Model RMSE: {final_rmse:.4f}")

final_mae = mean_absolute_error(y_val, final_predictions)
print(f"Final Model MAE: {final_mae:.4f}")


final_mape = mean_absolute_percentage_error(y_val, final_predictions)
print(f"Final Model MAPE: {final_mape:.4f}")
