import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class CSVDataLoader:
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(
            csv_file,
            parse_dates=["event_timestamp"],
            infer_datetime_format=True,
            low_memory=False,
        )

    def get_cols(self):
        return self.data.columns.to_list()

    def show(self, df=None, lb=0, ub=20, print_default=False):
        if df is not None:
            print(df[lb:ub])
        if print_default:
            print(self.data[lb:ub])

    def pre_process(self, df: pd.DataFrame):
        # for pre-processing the raw data. Eg. Filling the missing values with rolling mean or dropping NaN values and so on..
        df.dropna(inplace=True)

    def prepare_features(self):
        columns = self.data.columns.to_list()

        self.data["hour"] = self.data["event_timestamp"].dt.hour
        self.data["day_of_week"] = self.data["event_timestamp"].dt.dayofweek
        self.data["month"] = self.data["event_timestamp"].dt.month

        # prepare lag features
        # TODO: Perhaps, try k lags

        self.data["load_lag1"] = self.data["load_MW"].shift(1)
        self.data["load_lag2"] = self.data["load_MW"].shift(2)
        self.data["load_lag3"] = self.data["load_MW"].shift(3)

        # Rolling window features
        # TODO: Try k window size
        self.data["load_rolling_mean_3"] = self.data["load_MW"].rolling(window=3).mean()
        self.data["load_rolling_std_3"] = self.data["load_MW"].rolling(window=3).std()

        self.pre_process(self.data)

        features = columns[1:-1] + [
            "hour",
            "day_of_week",
            "month",
            "load_lag1",
            "load_lag2",
            "load_lag3",
            "load_rolling_mean_3",
        ]

        X = self.data[features]
        y = self.data["load_MW"]
        return X, y

    def normalize(self, data, type):
        scaler = StandardScaler() if type == "standard" else MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def split(self, X, y, test_size=0.2, shuffle=False):
        # TODO: Can use time-based splitting techniques
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )
        return X_train, X_test, y_train, y_test
