import xgboost as xgb

from csv_loader import DataLoader
from hyper_parameter_tuning import HyperParameterTuner

if __name__ == "__main__":
    weather_data = DataLoader("./weatherkit_plus_load.csv")
    cols = weather_data.get_cols()
    X, y = weather_data.prepare_features()
    X_train, X_test, y_train, y_test = weather_data.split(X, y)

    tuner = HyperParameterTuner(xgb.XGBRegressor(objective="reg:squarederror"))
    param_grid = {
        "n_estimators": [100, 300, 500, 700],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    best_model = tuner.tune(X_train, y_train, param_grid=param_grid)
