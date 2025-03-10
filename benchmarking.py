import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from cfg_loader import ConfigLoader
from csv_loader import CSVDataLoader
from utils import detect_outliers_z_score, is_stationary

cfg = ConfigLoader("configs", "config").load()
data_loader_cfg = cfg.get("data_loader")

weather_data = CSVDataLoader(data_loader_cfg["path"])

X, y = weather_data.prepare_features()

if data_loader_cfg["remove_outliers"]:
    outliers = detect_outliers_z_score(y)
    y = y[~y.index.isin(outliers.index)]
    X = X.loc[y.index]

stationarity = is_stationary(y)


# def evaluate_model(y_true, y_pred, model_name):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mape = mean_absolute_percentage_error(y_true, y_pred)
#     print(f"{model_name} Performance:")
#     print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f} \n")


# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

metrics = {"ARIMA": [], "SARIMA": [], "Exponential Smoothing": []}


for train_idx, test_idx in tscv.split(y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ARIMA
    arima_model = ARIMA(y_train, order=(2, 0, 1))
    arima_fit = arima_model.fit()
    y_pred_arima = arima_fit.forecast(steps=len(y_test))

    metrics["ARIMA"].append(
        [
            mean_absolute_error(y_test, y_pred_arima),
            np.sqrt(mean_squared_error(y_test, y_pred_arima)),
            mean_absolute_percentage_error(y_test, y_pred_arima),
        ]
    )

    # auto ARIMA
    # auto_model = auto_arima(
    #     y_train, seasonal=False, trace=True, suppress_warnings=True, stepwise=True
    # )
    # print(auto_model.summary())

    # SARIMA
    sarima_model = SARIMAX(y_train, order=(2, 0, 1), seasonal_order=(1, 0, 1, 3))
    sarima_fit = sarima_model.fit()
    y_pred_sarima = sarima_fit.forecast(steps=len(y_test))

    metrics["SARIMA"].append(
        [
            mean_absolute_error(y_test, y_pred_sarima),
            np.sqrt(mean_squared_error(y_test, y_pred_sarima)),
            mean_absolute_percentage_error(y_test, y_pred_sarima),
        ]
    )

    # Exponential Smoothing
    es_model = ExponentialSmoothing(
        y_train, trend="mul", seasonal="mul", seasonal_periods=3
    )
    es_fit = es_model.fit()
    y_pred_es = es_fit.forecast(steps=len(y_test))

    metrics["Exponential Smoothing"].append(
        [
            mean_absolute_error(y_test, y_pred_es),
            np.sqrt(mean_squared_error(y_test, y_pred_es)),
            mean_absolute_percentage_error(y_test, y_pred_es),
        ]
    )

for model, values in metrics.items():
    avg_mae, avg_rmse, avg_mape = np.mean(values, axis=0)
    print(f"{model} Average Performance across folds:")
    print(f"MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}, MAPE: {avg_mape:.2f} \n")
