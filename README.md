# Time Series Analysis

This repository contains code for performing time series analysis. It includes various models and methods for forecasting and analyzing time series data, such as ARIMA, XGBoost, and LightGBM.

## Installation

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/Praneethsv/time-series-analysis.git

cd time-series-analysis

```

### 2. Set up & Configuration

1. Create a conda environment using the following command:
    `conda create --name myenv python=3.9`

2. Activate the conda environment using the following command:
    `conda activate myenv`

3. Install the dependencies using: 
    `pip install -r requirements.txt`

4. Open the `config.yaml` file and modify the path variable under data_loader to point out the time series data (csv format).

5. Give the parameters of your choice for any of the available models under model in `config.yaml` or just run the main files as described below with already tuned parameters set in the config. 


### 3. Verify Installation

Verify the installation by running the following commands:

To fit xgboost model run the following command:
```bash
python xgboost_main.py
```
To fit lightgbm model run the following command:
```bash
python lightgbm_main.py
```

To fit catboost model run the following command:
```bash
python catboost_main.py
```

## Note:  

All the trained models are saved in the current working directory when you run main files of xgboost, lightgbm, and catboost.

