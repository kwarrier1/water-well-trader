import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from plots_raw import get_merged_data
import logging
from scipy import signal
from statsmodels.tsa.seasonal import STL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("TS-Model")

# ------------------------------ Advanced Feature Engineering ------------------------------ #

def create_enhanced_features(df: pd.DataFrame, target_cols, lags=[1, 7, 30]) -> pd.DataFrame:
    """
    Creates an enhanced set of features for time series forecasting using advanced techniques.
    
    Args:
        df: Input dataframe with time series data
        target_cols: List of target column names to create features for
        lags: List of lag periods to use
        
    Returns:
        DataFrame with added features and NaN values removed
    """
    df = df.copy()
    
    # Create basic lag features for each target column
    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    # Add advanced rolling window statistics to capture complex patterns
    rolling_windows = [7, 14, 30, 60]
    stats_funcs = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'skew': lambda x: pd.Series(x).skew(),
        'kurt': lambda x: pd.Series(x).kurt()
    }
    
    for col in target_cols:
        for window in rolling_windows:
            for stat_name, stat_func in stats_funcs.items():
                # Skip some combinations for efficiency
                if window > 30 and stat_name in ['skew', 'kurt']:
                    continue
                df[f"{col}_{stat_name}_{window}"] = df[col].rolling(window=window).apply(stat_func)
    
    # Add rate of change features
    for col in target_cols:
        for window in [7, 14, 30]:
            df[f"{col}_roc_{window}"] = df[col].pct_change(periods=window, fill_method=None)
    
    # Add Fourier features to capture cyclical patterns
    add_fourier_features(df, 'Mean GWE', [7, 30, 90, 365])
    
    # Add seasonal decomposition features
    add_seasonal_decomposition(df, 'Mean GWE', period=365)
    
    # Add calendar features to capture seasonality
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['Date']).dt.month
    df['quarter'] = pd.to_datetime(df['Date']).dt.quarter
    df['day_of_year'] = pd.to_datetime(df['Date']).dt.dayofyear
    df['is_month_start'] = pd.to_datetime(df['Date']).dt.is_month_start.astype(int)
    df['is_month_end'] = pd.to_datetime(df['Date']).dt.is_month_end.astype(int)
    
    # Add groundwater specific features
    df['GWE_above_avg'] = (df['Mean GWE'] > df['Mean GWE'].mean()).astype(int)
    df['GWE_trend'] = df['Mean GWE'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # For FPI and CDZI specifically, add interaction features
    for stock in ['FPI', 'CDZI']:
        if stock in df.columns:
            df[f'{stock}_GWE_ratio'] = df[stock] / df['Mean GWE'].replace(0, np.nan)
            df[f'{stock}_GWE_diff'] = df[stock] - df['Mean GWE']
    
    # Remove rows with missing values from feature creation
    df.dropna(inplace=True)
    return df

def add_fourier_features(df, column, periods):
    """
    Add Fourier features to capture cyclical patterns in the data.
    
    Args:
        df: DataFrame containing the time series
        column: Column to transform
        periods: List of periods to extract (e.g., [7, 30, 365] for weekly, monthly, yearly)
    """
    for period in periods:
        for n in range(1, 3):  # Use first 2 Fourier terms
            df[f'{column}_sin_{period}_{n}'] = np.sin(2 * n * np.pi * df.index / period)
            df[f'{column}_cos_{period}_{n}'] = np.cos(2 * n * np.pi * df.index / period)

def add_seasonal_decomposition(df, column, period=365):
    """
    Add seasonal decomposition features using STL (Seasonal-Trend-Loess).
    
    Args:
        df: DataFrame containing the time series
        column: Column to decompose
        period: Period for the seasonal component
    """
    if len(df) >= 2 * period:  # Ensure enough data points
        try:
            # Fill any missing values for decomposition
            series = df[column].copy()
            series = series.interpolate(method='linear')
            
            # Apply STL decomposition
            stl = STL(series, period=period)
            result = stl.fit()
            
            # Add components as features
            df[f'{column}_trend'] = result.trend
            df[f'{column}_seasonal'] = result.seasonal
            df[f'{column}_residual'] = result.resid
            
            # Add volatility of residuals
            df[f'{column}_residual_volatility'] = result.resid.rolling(window=30).std()
        except Exception as e:
            log.warning(f"Could not perform seasonal decomposition: {e}")

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

# ------------------------------ Improved Cross-Validation ------------------------------ #

def walk_forward_validation(data, target_cols, initial_train_size=0.5, step_size=0.1, max_window_size=None):
    """
    Performs walk-forward validation, which is more appropriate for time series than standard cross-validation.
    
    Args:
        data: DataFrame containing the time series data
        target_cols: List of target columns to predict
        initial_train_size: Initial proportion of data to use for training
        step_size: Proportion of data to move forward in each iteration
        max_window_size: Maximum window size for training (if None, use expanding window)
        
    Returns:
        Dictionary of results with performance metrics
        Best models for each target column
        Final df used for training
    """
    df = data.sort_values("Date").reset_index(drop=True)
    df = create_enhanced_features(df, target_cols)
    
    # Define features and targets
    X = df.drop(columns=target_cols + ["Date"])
    y = df[target_cols]
    
    n_samples = len(X)
    initial_train_end = int(n_samples * initial_train_size)
    step = int(n_samples * step_size)
    results = {col: {"rmse": [], "mae": [], "r2": []} for col in target_cols}
    
    # Hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'min_child_samples': [20, 40],
        'colsample_bytree': [0.8, 1.0],
    }
    
    # Store best model for each target
    best_models = {}
    
    # Track test indices for plotting
    all_test_indices = []
    all_predictions = {col: [] for col in target_cols}
    all_actuals = {col: [] for col in target_cols}
    
    # Perform walk-forward validation
    for i in range(initial_train_end, n_samples - step, step):
        log.info(f"\nValidation Window: Training from 0 to {i}, Testing from {i} to {i+step}")
        
        # Define train/test split for this iteration
        if max_window_size is not None:
            train_start = max(0, i - max_window_size)
        else:
            train_start = 0
            
        train_indices = range(train_start, i)
        test_indices = range(i, min(i + step, n_samples))
        all_test_indices.extend(test_indices)
        
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        # Scale data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scalers = scale_data(
            X_train, X_test, y_train, y_test
        )
        
        # Train models for each target
        for col in target_cols:
            log.info(f"  → Training model for {col}")
            
            # Initialize model
            if col in best_models:
                # Use previous best parameters as a starting point
                init_params = best_models[col]['params']
                log.info(f"    Using previous best parameters as starting point")
                
                # Adjust the grid to search around previous best params
                adjusted_grid = {}
                for param, value in init_params.items():
                    if param in param_grid:
                        adjusted_grid[param] = [value]
                        # Add nearby values if applicable
                        if isinstance(value, (int, float)) and value > 0:
                            if param == 'learning_rate':
                                adjusted_grid[param] = [value*0.5, value, value*1.5]
                            elif param in ['n_estimators', 'max_depth', 'min_child_samples']:
                                adjusted_grid[param] = [max(1, int(value*0.8)), value, int(value*1.2)]
                
                # Merge with original grid
                search_grid = {**param_grid, **adjusted_grid}
            else:
                search_grid = param_grid
            
            # Train with grid search for hyperparameter optimization
            tscv = TimeSeriesSplit(n_splits=3)
            model = LGBMRegressor(objective='regression', verbose=-1)
            grid = GridSearchCV(
                estimator=model,
                param_grid=search_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X_train_scaled, y_train_scaled[col])
            
            # Update best model parameters
            best_models[col] = {
                'params': grid.best_params_,
                'model': grid.best_estimator_
            }
            
            # Make predictions in scaled space
            preds_scaled = grid.best_estimator_.predict(X_test_scaled)
            
            # Convert predictions back to original scale
            preds = target_scalers[col].inverse_transform(
                preds_scaled.reshape(-1, 1)
            ).flatten()
            
            # Store predictions and actuals for later plotting
            all_predictions[col].extend(preds)
            all_actuals[col].extend(y_test[col].values)
            
            # Calculate performance metrics in original scale
            rmse = np.sqrt(mean_squared_error(y_test[col], preds))
            mae = np.mean(np.abs(y_test[col] - preds))
            r2 = r2_score(y_test[col], preds)
            
            # Store results
            results[col]["rmse"].append(rmse)
            results[col]['mae'].append(mae)
            results[col]['r2'].append(r2)
            
            log.info(f"    Best Params: {grid.best_params_}")
            log.info(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Plot full test results
    dates = pd.to_datetime(df['Date'])
    test_dates = dates.iloc[all_test_indices]
    
    for col in target_cols:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, all_actuals[col], label='Actual', linewidth=2)
        plt.plot(test_dates, all_predictions[col], label='Predicted', linewidth=2, linestyle='--')
        
        avg_rmse = np.mean(results[col]["rmse"])
        avg_mae = np.mean(results[col]['mae'])
        avg_r2 = np.mean(results[col]['r2'])
        
        plt.title(f'{col} Walk-Forward Validation Results: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()  
        plt.tight_layout()
        plt.savefig(f'{col}_walk_forward_predictions.png')
        plt.close()
    
    # Display average performance across validation windows
    log.info("\nAverage Metrics Across Validation Windows")
    for col in target_cols:
        avg_rmse = np.mean(results[col]["rmse"])
        avg_mae = np.mean(results[col]["mae"])
        avg_r2 = np.mean(results[col]["r2"])
        log.info(f"{col}: Avg RMSE={avg_rmse:.4f}, Avg MAE={avg_mae:.4f}, Avg R²={avg_r2:.4f}")
    
    structured_results = {
        'metrics': results,
        'test_indices': all_test_indices,
        'predictions': all_predictions,
        'actuals': all_actuals,
        'dates': test_dates
    }
    save_path = '/Users/keshavwarrier/water-well-trader/model-runs'
    df.to_csv(f"{save_path}/engineered_data.csv", index=False)
    log.info(f"Engineered data saved to {save_path}/engineered_data.csv")
    all_predictions_df = pd.DataFrame(all_predictions, index=test_dates)
    all_predictions_df.to_csv(f"{save_path}/predictions.csv", index=True)   
    log.info(f"Predictions saved to {save_path}/predictions.csv")
    return structured_results, best_models, df

# ------------------------------ Main Function ------------------------------ #

def main():
    """
    Main function to run the enhanced time series cross-validation pipeline.
    """
    data = get_merged_data()
    target_cols = ["FPI", "CDZI"]  # Focus on these stocks as specified
    
    # Run the enhanced walk-forward validation
    results, best_models, engineered_df = walk_forward_validation(
        data, 
        target_cols=target_cols, 
        initial_train_size=0.5,
        step_size=0.1,
        max_window_size=365  # Use at most the last year of data for training
    )
    
    return results, best_models

# Run the pipeline if executed directly
if __name__ == "__main__":
    main()