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

    ## CATBOOST
    # catboost_model = CatBoostRegressor(
    #     iterations=1000,
    #     learning_rate=0.13786576194363528,
    #     depth=7,
    #     loss_function="RMSE",
    #     l2_leaf_reg=7.76873502580487e-08,
    #     bagging_temperature=0.7329476073761131,
    #     random_strength=0.0627338404379398,
    # )

    catboost_model = CatBoostRegressor(**params)

    # Fit the model
    catboost_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    # Predictions
    y_pred = catboost_model.predict(X_val)

    # Compute metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    print(f"CatBoost Performance - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")


if __name__ == "__main__":

    main("configs", "config")
