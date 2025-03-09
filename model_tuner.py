import catboost as cb
import lightgbm as lgb
import optuna
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from cfg_loader import ConfigLoader
from csv_loader import CSVDataLoader
from utils import detect_outliers_z_score


class ModelTuner:
    def __init__(self, config_path, config_name, model_name):

        self.cfg = ConfigLoader(config_path, config_name).load()
        data_loader_cfg = self.cfg.get("data_loader")

        weather_data = CSVDataLoader(data_loader_cfg["path"])

        X, y = weather_data.prepare_features()

        if data_loader_cfg["remove_outliers"]:
            # remove outliers
            outliers = detect_outliers_z_score(y)
            y = y[~y.index.isin(outliers.index)]
            X = X.loc[y.index]

        self.model_name = model_name

        self.X_train, self.X_val, self.y_train, self.y_val = weather_data.split(X, y)

    def get_model(self, trial):
        if self.model_name == "lightgbm":
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "n_estimators": 300,
            }
            return lgb.LGBMRegressor(**params)

        elif self.model_name == "catboost":
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
                "iterations": 300,
                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 10.0),
                "bagging_temperature": trial.suggest_uniform(
                    "bagging_temperature", 0.0, 1.0
                ),
                "random_strength": trial.suggest_uniform("random_strength", 0.0, 1.0),
                "verbose": 0,
            }
            return cb.CatBoostRegressor(**params)

        elif self.model_name == "xgboost":
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 10.0),
                "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
                "n_estimators": 300,
            }
            return xgb.XGBRegressor(**params)
        else:
            raise ValueError("Unsupported model")

    def objective(self, trial):
        """Objective function for Optuna optimization."""
        model = self.get_model(trial)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)

        rmse = mean_squared_error(self.y_val, y_pred) ** 0.5
        mae = mean_absolute_error(self.y_val, y_pred)
        mape = mean_absolute_percentage_error(self.y_val, y_pred)

        return mae + rmse + mape  # Minimize all three combined

    def tune_and_train(self, n_trials=500):
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)

        print("\nBest parameters found:", study.best_params)
        print("Best RMSE, MAE, MAPE:", study.best_value)

        # Train final model with best params
        best_model = self.get_model(optuna.trial.FixedTrial(study.best_params))
        best_model.fit(self.X_train, self.y_train)

        # print the best params

        for k, v in study.best_params.items():
            print(f"{k}: {v}")

        # Predictions
        final_predictions = best_model.predict(self.X_val)
        final_rmse = mean_squared_error(self.y_val, final_predictions) ** 0.5
        final_mae = mean_absolute_error(self.y_val, final_predictions)
        final_mape = mean_absolute_percentage_error(self.y_val, final_predictions)

        print("\nFinal Model Performance:")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"MAPE: {final_mape:.4f}")

        return best_model


if __name__ == "__main__":
    model_tuner = ModelTuner("configs", "config", "catboost")
    model_tuner.tune_and_train()
