import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from cfg_loader import ConfigLoader
from csv_loader import CSVDataLoader
from utils import detect_outliers_quantile, detect_outliers_z_score


def main(config_path, config_name):
    cfg = ConfigLoader(config_path, config_name).load()
    data_loader_cfg = cfg.get("data_loader")

    weather_data = CSVDataLoader(data_loader_cfg["path"])

    X, y = weather_data.prepare_features()

    if data_loader_cfg["remove_outliers"]:

        outliers = detect_outliers_z_score(y)
        y = y[~y.index.isin(outliers.index)]
        X = X.loc[y.index]

    X_train, X_test, y_train, y_test = weather_data.split(X, y)

    if data_loader_cfg["normalize"]:
        X_train, X_test = weather_data.normalize(
            X_train, "standard"
        ), weather_data.normalize(X_test, "standard")

    tscv = TimeSeriesSplit(n_splits=5)

    ## LIGHTGBM

    model_cfg = dict(cfg.get("model").get("lightgbm"))
    maes, rmses, mapes = [], [], []

    best_model_path = "lightgbm_model.txt"
    best_rmse = float("inf")

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        model = lgb.train(
            model_cfg,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
        )

        # Predict and evaluate
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        rmses.append(rmse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mapes.append(mape)

        if rmse < best_rmse:
            best_rmse = rmse
            model.save_model(best_model_path)

    worst_rmse = np.max(rmses)
    print("Worst RMSE:", worst_rmse)
    std_rmse = np.std(rmses)
    print("Standard Deviation of RMSE:", std_rmse)

    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    print(
        f"LightGBM Performance - MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}, MAPE: {avg_mape:.2f}"
    )


if __name__ == "__main__":

    main("configs", "config")
