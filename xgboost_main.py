import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from cfg_loader import ConfigLoader
from csv_loader import CSVDataLoader
from energy_load_predictor import EnergyLoadPredictor
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

    model_cfg = dict(cfg.get("model").get("xgboost"))

    energy_load_model = EnergyLoadPredictor(**model_cfg)

    # other param configurations found through hyper parameter tuning; one can experiment
    # energy_load_model = xgb.XGBRegressor(
    #     objective="reg:squarederror",
    #     learning_rate=0.10881423369683225,
    #     max_depth=9,
    #     eval_metric="rmse",
    #     min_child_weight=1,
    #     alpha=2.6445390308358276e-07,
    #     subsample=0.9444859273350957,
    #     colsample_bytree=0.85611557093695,
    #     random_state=42,
    #     n_estimators=300,
    # )

    # energy_load_model = xgb.XGBRegressor(
    #     objective="reg:squarederror",
    #     learning_rate=0.044902247165282884,
    #     max_depth=10,
    #     eval_metric="rmse",
    #     min_child_weight=1,
    #     alpha=2.1368030534944994e-05,
    #     subsample=0.8212183597751991,
    #     colsample_bytree=0.9764746706925298,
    #     random_state=42,
    #     n_estimators=300,
    #     tree_method="hist",
    # )

    tscv = TimeSeriesSplit(n_splits=5)

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

    model_path = "xgboost_model.json"
    energy_load_model.save(model_path)
    print(f"Model saved to {model_path}")

    energy_load_model.predict(X_test, y_test) # here y_test is being used to evaluate the performance


if __name__ == "__main__":

    main("configs", "config")
