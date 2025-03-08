import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Load dataset
data = pd.read_csv("./weatherkit_plus_load.csv")
data["event_timestamp"] = pd.to_datetime(data["event_timestamp"], errors="coerce")
y = data["load_MW"]  # Assuming 'load_MW' is the target variable
X = data.drop(columns=["load_MW"])


# Feature engineering: Add lag and rolling statistics
def create_lag_features(X, y, lags=5):
    for lag in range(1, lags + 1):
        X[f"lag_{lag}"] = y.shift(lag)
    X["rolling_mean_3"] = y.rolling(window=3).mean()
    X["rolling_mean_7"] = y.rolling(window=7).mean()
    X["rolling_std_3"] = y.rolling(window=3).std()
    return X


X = create_lag_features(X, y)

# Drop missing values created by lag features
X = X.dropna()

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"LightGBM Performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
