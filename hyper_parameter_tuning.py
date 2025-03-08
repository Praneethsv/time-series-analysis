from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit


class HyperParameterTuner:
    def __init__(self, model, type: str = "grid"):
        self.model = model
        self.type = type
        self.tscv = TimeSeriesSplit(n_splits=5)

    def tune(
        self, X_train, y_train, param_grid, cv=5, scoring="neg_mean_absolute_error"
    ):

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=scoring,
            cv=self.tscv,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
        return grid_search.best_estimator_
