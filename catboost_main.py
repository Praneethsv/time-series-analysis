import numpy as np
from catboost import CatBoostRegressor
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
        # remove outliers
        outliers = detect_outliers_z_score(y)
        y = y[~y.index.isin(outliers.index)]
        X = X.loc[y.index]

    X_train, X_val, y_train, y_val = weather_data.split(X, y)

    if data_loader_cfg["normalize"]:
        X_train, X_test = weather_data.normalize(
            X_train, "standard"
        ), weather_data.normalize(X_test, "standard")

    params = dict(cfg.get("model").get("catboost"))

    catboost_model = CatBoostRegressor(**params)

    catboost_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    model_path = "catboost_model.cbm"
    catboost_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    y_pred = catboost_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    print(f"CatBoost Performance - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")


if __name__ == "__main__":

    main("configs", "config")
