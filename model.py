import json
import pickle
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


class EnergyLoadPredictor:
    def __init__(
        self,
        objective,
        n_estimators,
        learning_rate,
        max_depth,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
    ):
        self.model = xgb.XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )

    def fit(self, X_train, y_train) -> xgb.XGBRegressor:
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X_test, y_test) -> None:
        y_pred = self.model.predict(X_test)

        print("Sample y_test:", y_test[:5])
        print("Sample y_pred:", y_pred[:5])

        # evaluation metrics
        mean_abs_err = mean_absolute_error(y_test, y_pred)
        mean_square_err = mean_squared_error(y_test, y_pred)
        r_mean_square_err = mean_square_err**0.5
        mean_abs_perc_error = mean_absolute_percentage_error(y_test, y_pred)
        print("Model Performance Metrics:")
        print(f"MAE: {mean_abs_err:.2f}")
        print(f"RMSE: {r_mean_square_err:.2f}")
        print(f"MAPE: {mean_abs_perc_error:.2%}")

    def save(self, name):
        self.model.save_model(name)

    def load(self, model):
        if isinstance(model, dict):
            loaded_model = xgb.Booster()
            loaded_model.load_model(model)

        if isinstance(model, object):
            with open(model, "rb") as f:
                loaded_model = pickle.load(f)
        return loaded_model
