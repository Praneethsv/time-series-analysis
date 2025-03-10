from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

from utils import predict

forecast_columns = [
    "weatherkit_observed_temperature_C",
    "weatherkit_observed_humidity_pc",
    "weatherkit_observed_air_pressure_kPa",
    "weatherkit_observed_cloud_cover_pc",
    "weatherkit_observed_wind_direction_deg",
    "weatherkit_observed_wind_speed_km_h",
    "weatherkit_forecast_temp_C",
    "weatherkit_forecast_humidity_pc",
    "weatherkit_forecast_air_pressure_kPa",
    "weatherkit_forecast_cloud_cover_pc",
    "weatherkit_forecast_wind_direction_deg",
    "weatherkit_forecast_wind_speed_km_h",
    "hour",
    "day_of_week",
    "month",
    "load_lag1",
    "load_lag2",
    "load_lag3",
    "load_rolling_mean_3",
]

num_hours = 24
np.random.seed(42)

observed_dummy = np.random.uniform(low=10, high=35, size=(num_hours, 6))
forecasted_dummy = np.random.uniform(low=10, high=35, size=(num_hours, 6))


temporal_lag_features = np.random.uniform(low=1, high=24, size=(num_hours, 6))

load_rolling_mean_3 = np.random.uniform(low=0, high=100, size=(num_hours, 1))

combined_data = np.hstack(
    [observed_dummy, forecasted_dummy, temporal_lag_features, load_rolling_mean_3]
)


forecast_data = pd.DataFrame(combined_data, columns=forecast_columns)


forecast_dates = pd.date_range(
    start=datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1),
    periods=num_hours,
    freq="H",
)


loaded_model = xgb.XGBRegressor()
loaded_model.load_model("./xgboost_model.json")
res = predict(loaded_model, forecast_data)
print(res)
