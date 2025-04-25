import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from plots_raw import get_merged_data
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("TS-Model")

# ------------------------------ Feature Engineering ------------------------------ #

def create_essential_features(df: pd.DataFrame, target_cols, lags=[1, 7, 30]) -> pd.DataFrame:
   """
   Creates a minimal but effective set of features for time series forecasting.
   
   Args:
       df: Input dataframe with time series data
       target_cols: List of target column names to create features for
       lags: List of lag periods to use
       
   Returns:
       DataFrame with added features and NaN values removed
   """
   df = df.copy()
   
   # Create lag features for each target column
   for col in target_cols:
       for lag in lags:
           df[f"{col}_lag_{lag}"] = df[col].shift(lag)
   
   # Add a rolling mean feature to capture trend
   df['Mean GWE_rolling_mean_14'] = df['Mean GWE'].rolling(window=14).mean()
   
   # Add calendar features to capture seasonality
   df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
   df['month'] = pd.to_datetime(df['Date']).dt.month
   
   # Remove rows with missing values from feature creation
   df.dropna(inplace=True)
   return df

# ------------------------------ Scaling Functions ------------------------------ #

def scale_data(X_train, X_test, y_train, y_test):
   """
   Scales both features and target variables using RobustScaler.
   
   Args:
       X_train: Training features
       X_test: Testing features
       y_train: Training targets
       y_test: Testing targets
       
   Returns:
       Scaled data and the fitted scalers for inverse transformation
   """
   # Scale feature variables
   feature_scaler = RobustScaler()
   X_train_scaled = pd.DataFrame(
       feature_scaler.fit_transform(X_train),
       columns=X_train.columns,
       index=X_train.index
   )
   X_test_scaled = pd.DataFrame(
       feature_scaler.transform(X_test),
       columns=X_test.columns,
       index=X_test.index
   )
   
   # Scale target variables with separate scalers for each column
   y_train_scaled = pd.DataFrame(index=y_train.index)
   y_test_scaled = pd.DataFrame(index=y_test.index)
   target_scalers = {}
   
   for col in y_train.columns:
       scaler = RobustScaler()
       y_train_scaled[col] = scaler.fit_transform(y_train[[col]]).flatten()
       y_test_scaled[col] = scaler.transform(y_test[[col]]).flatten()
       target_scalers[col] = scaler
   
   return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scalers

# ------------------------------ Model Pipeline ------------------------------ #

def model(data: pd.DataFrame, target_cols, n_splits=5):
   """
   Time series cross-validation pipeline with proper scaling and evaluation.
   
   Args:
       data: Input dataframe containing time series data
       target_cols: List of target columns to predict
       n_splits: Number of cross-validation folds
       
   Returns:
       Dictionary of results with performance metrics
   """
   # Prepare data
   df = data.sort_values("Date").reset_index(drop=True)
   df = create_essential_features(df, target_cols)
   
   # Define features and targets
   X = df.drop(columns=target_cols + ["Date"])
   y = df[target_cols]
   
   # Set up cross-validation
   tscv = TimeSeriesSplit(n_splits=n_splits)
   results = {col: {"rmse": [], "mae": []} for col in target_cols}
   
   # Define hyperparameter search space
   param_grid = {
       'n_estimators': [100, 200],
       'learning_rate': [0.01, 0.05],
       'max_depth': [5, 7],
       'min_child_samples': [20]
   }
   
   # Perform cross-validation
   for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
       log.info(f"\nFold {fold + 1}/{n_splits}")
       X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
       y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
       
       # Scale both features and targets
       X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scalers = scale_data(
           X_train, X_test, y_train, y_test
       )
       dates = pd.to_datetime(df['Date'])
       # Train models for each target
       for col in target_cols:
           log.info(f"  → Training model for {col}")
           
           # Train with grid search for hyperparameter optimization
           model = LGBMRegressor(objective='regression', verbose=-1)
           grid = GridSearchCV(
               estimator=model,
               param_grid=param_grid,
               cv=TimeSeriesSplit(n_splits=3),
               scoring='neg_mean_squared_error',
               n_jobs=-1
           )
           grid.fit(X_train_scaled, y_train_scaled[col])
           best_model = grid.best_estimator_
           
           # Make predictions in scaled space
           preds_scaled = best_model.predict(X_test_scaled)
           
           # Convert predictions back to original scale
           preds = target_scalers[col].inverse_transform(
               preds_scaled.reshape(-1, 1)
           ).flatten()
           
           # Calculate performance metrics in original scale
           rmse = np.sqrt(mean_squared_error(y_test[col], preds))
           mae = np.mean(np.abs(y_test[col] - preds))
           r2 = r2_score(y_test[col], preds)
           
           # Store results
           results[col]["rmse"].append(rmse)
           results[col]['mae'].append(mae)
           
           log.info(f"    Best Params: {grid.best_params_}")
           log.info(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}")
           
           test_dates = dates.iloc[test_idx]

           if fold == n_splits - 1:
                plt.figure(figsize=(10, 5))
                plt.plot(test_dates, y_test[col].values, label='Actual', linewidth=2)
                plt.plot(test_dates, preds, label='Predicted', linewidth=2, linestyle='--')
                plt.title(f'{col}: RMSE={rmse:.4f}, MAE={mae:.4f}')
                plt.xlabel('Date')
                plt.ylabel(col)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.gcf().autofmt_xdate()  # Auto-format the date labels
                plt.tight_layout()
                plt.savefig(f'{col}_predictions.png')  # Save instead of show to avoid blocking
                plt.close()
   
   # Display average performance across folds
   log.info("\n Average Metrics Across Folds")
   for col in target_cols:
       avg_rmse = np.mean(results[col]["rmse"])
       avg_mae = np.mean(results[col]["mae"])
       log.info(f"{col}: Avg RMSE={avg_rmse:.4f}, Avg MAE={avg_mae:.4f}")
   
   return results

# Run the time series cross-validation pipeline
results = model(
   get_merged_data(), 
   target_cols=["FPI", "CWT", "AVO", "ZC=F", "CDZI"], 
   n_splits=5
)