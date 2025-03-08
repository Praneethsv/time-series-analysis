import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from csv_loader import CSVDataLoader


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def visualize(self):
        # self.box_plot_load()
        # self.histogram()
        # self.time_series_load_plot()
        self.weather_overlay_load_plot()

    def box_plot_load(self):
        plt.figure(figsize=(8, 4))
        sns.boxplot(self.df["load_MW"])
        plt.title("Boxplot of Target Variable")
        plt.show()

    def histogram(self):
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df["load_MW"], bins=50, kde=True)
        plt.title("Histogram of Target Variable")
        plt.show()

    def time_series_load_plot(self):
        self.df["event_timestamp"] = pd.to_datetime(self.df["event_timestamp"])
        # Plot the time series of load_MW
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.df["event_timestamp"],
            self.df["load_MW"],
            label="Load (MW)",
            color="blue",
            linewidth=1,
        )
        plt.xlabel("Time")
        plt.ylabel("Load (MW)")
        plt.title("Time Series Plot of Load (MW)")
        plt.legend()
        plt.grid()
        plt.show()

    def weather_overlay_load_plot(self):
        df = self.df.sort_values(by="event_timestamp")

        # Set timestamp as index
        df.set_index("event_timestamp", inplace=True)

        # Plot Load MW along with key weather features
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df["load_MW"], label="Load (MW)", color="blue", alpha=0.7)
        plt.plot(
            df.index,
            df["weatherkit_observed_temperature_C"],
            label="Temperature (°C)",
            color="red",
            alpha=0.7,
        )
        plt.plot(
            df.index,
            df["weatherkit_observed_humidity_pc"],
            label="Humidity (%)",
            color="green",
            alpha=0.7,
        )

        plt.plot(
            df.index,
            df["weatherkit_observed_air_pressure_kPa"],
            label="Air Pressure (Kilo-Pascals)",
            color="brown",
            alpha=0.7,
        )

        plt.plot(
            df.index,
            df["weatherkit_observed_wind_speed_km_h"],
            label="Wind Speed (m/s)",
            color="purple",
            alpha=0.7,
        )

        plt.plot(
            df.index,
            df["weatherkit_observed_wind_direction_deg"],
            label="Wind Direction (degrees)",
            color="pink",
            alpha=0.7,
        )

        plt.plot(
            df.index,
            df["weatherkit_observed_cloud_cover_pc"],
            label="Cloud Cover (%)",
            color="yellow",
            alpha=0.7,
        )

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Time Series Plot of Load and Weather Features")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    weather_data = CSVDataLoader("./weatherkit_plus_load.csv")
    vis_dom = DataVisualizer(weather_data.data)
    vis_dom.visualize()
