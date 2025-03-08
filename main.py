import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from cfg_loader import ConfigLoader
from csv_loader import CSVDataLoader
from model import EnergyLoadPredictor
from utils import detect_outliers_quantile, detect_outliers_z_score


def main(config_path, config_name):
    cfg = ConfigLoader(config_path, config_name).load()
    data_loader_cfg = cfg.get("data_loader")

    weather_data = CSVDataLoader(data_loader_cfg["path"])
    cols = weather_data.get_cols()
    X, y = weather_data.prepare_features()

    if data_loader_cfg["remove_outliers"]:
        # remove outliers
        outliers = detect_outliers_z_score(y)
        y = y[~y.index.isin(outliers.index)]
        X = X.loc[y.index]

    X_train, X_test, y_train, y_test = weather_data.split(X, y)

    if data_loader_cfg["normalize"]:
        X_train, X_test = weather_data.normalize(
            X_train, "standard"
        ), weather_data.normalize(X_test, "standard")

    energy_load_model = EnergyLoadPredictor(
        objective="reg:squarederror",  # "reg:pseudohubererror",  # "reg:squarederror"
        n_estimators=700,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    tscv = TimeSeriesSplit(n_splits=5)

    # Use cross-validation to evaluate the model
    cv_scores = cross_val_score(
        energy_load_model.model,
        X_train,
        y_train,
        cv=tscv,
        scoring="neg_mean_absolute_error",
    )
    print(f"Cross-Validation MAE Scores: {cv_scores}")
    print(f"Mean CV MAE: {-cv_scores.mean()}")

    energy_load_model.fit(X_train, y_train)
    energy_load_model.predict(X_test, y_test)

    ## LIGHTGBM

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 142,
        "learning_rate": 0.03272,
        "min_data_in_leaf": 43,
        "lambda_l1": 1.26633,
        "lambda_l2": 0.008,
        "feature_fraction": 0.9798,
        "bagging_fraction": 0.9815,
        "bagging_freq": 5,
    }

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=500,
        )

        # Predict and evaluate
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(
            f"LightGBM Performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}"
        )


if __name__ == "__main__":

    main("configs", "config")
