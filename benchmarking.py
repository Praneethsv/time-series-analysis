import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
data = pd.read_csv("./weatherkit_plus_load.csv")
data["event_timestamp"] = pd.to_datetime(data["event_timestamp"], errors="coerce")
data.set_index("event_timestamp", inplace=True)
data = data.dropna()
y = data["load_MW"]  # Assuming 'load' is the target variable
X = data.drop(columns=["load_MW"])


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")


# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ARIMA
    arima_model = ARIMA(y_train, order=(5, 0, 5))  # (p,d,q) -> change if needed
    arima_fit = arima_model.fit()
    y_pred_arima = arima_fit.forecast(steps=len(y_test))
    evaluate_model(y_test, y_pred_arima, "ARIMA")

    # auto ARIMA
    auto_model = auto_arima(
        y_train, seasonal=False, trace=True, suppress_warnings=True, stepwise=True
    )
    print(auto_model.summary())

    # SARIMA
    # sarima_model = SARIMAX(
    #     y_train, order=(2, 0, 1), seasonal_order=(1, 1, 1, 4), exog=X_train
    # )
    # sarima_fit = sarima_model.fit()
    # y_pred_sarima = sarima_fit.forecast(steps=len(y_test), exog=X_test)
    # evaluate_model(y_test, y_pred_sarima, "SARIMA")

    # # Exponential Smoothing
    # es_model = ExponentialSmoothing(
    #     y_train, trend="add", seasonal="add", seasonal_periods=12
    # )
    # es_fit = es_model.fit()
    # y_pred_es = es_fit.forecast(steps=len(y_test))
    # evaluate_model(y_test, y_pred_es, "Exponential Smoothing")

    break  # Only run for the first split to save time
