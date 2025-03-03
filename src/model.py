import lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from plots_raw import scale_data, get_merged_data


# Feature Engineering - Create Lag Features
def create_lag_features(data, lags=None):
    if lags is None:
        lags = [1, 7, 30]
    for lag in lags:
        data[f"Mean GWE {lag} Day Lag"] = data["Mean GWE"].shift(lag)
    data.dropna(inplace=True)
    return data

# Machine Learning Pipeline
def train_ml_pipeline(data: pd.DataFrame):
    data = create_lag_features(data)
    tickers = ["FPI", "CWT", "AVO", "ZC=F", "CDZI"] # ETPs we are targeti
    date = ["Date"]
    features = [col for col in data.columns if col not in tickers]

    X = data[features]
    y = data[tickers]
    print(X.head(), y.head())

    tscv = TimeSeriesSplit(n_splits = 5)
    reg = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)

    # Full pipeline with model
    pipeline = Pipeline([
        ('preprocessor', scale_data(data, MinMaxScaler())),
        ('model', reg)
    ])

    # Train & Evaluate
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    print(f"Cross-Validation Scores: {scores.mean():.4f}")

train_ml_pipeline(get_merged_data())