import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("TS-Model")

def load_model_data(folder_path):
    """
    Load model prediction data from CSVs.
    """
    predictions = pd.read_csv(os.path.join(folder_path, 'predictions.csv'), index_col=0, parse_dates=True)
    actuals = pd.read_csv(os.path.join(folder_path, 'engineered_data.csv'), index_col=0, parse_dates=True)

    actuals = actuals[['FPI', 'CDZI']].loc[predictions.index]
    return predictions, actuals

def generate_signals(predictions, actuals, threshold_pct=0.02, smoothing_window=3):
    """
    Generate buy/sell/hold signals based on smoothed predictions and momentum confirmation.
    """
    signals = pd.DataFrame(index=predictions.index)
    
    # Smooth predictions
    smoothed_preds = predictions.rolling(window=smoothing_window, min_periods=1).mean()

    for col in predictions.columns:
        prev_actual = actuals[col].shift(1)
        pred_return = (smoothed_preds[col] - prev_actual) / prev_actual
        actual_return = actuals[col].pct_change().shift(1)

        signals[col] = 0  # Hold

        # Buy only if predicted return > threshold and actual return is positive (momentum confirmation)
        signals.loc[(pred_return > threshold_pct) & (actual_return > 0), col] = 1
        
        # Short only if predicted return < -threshold and actual return is negative
        signals.loc[(pred_return < -threshold_pct) & (actual_return < 0), col] = -1
        
    return signals


def backtest_strategy(signals, prices, initial_capital=10000, commission=0.001):
    """
    Backtest strategy allowing both long and short positions.
    """
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['cash'] = initial_capital
    portfolio['holdings'] = 0
    portfolio['total'] = initial_capital
    portfolio['returns'] = 0
    
    position_shares = 0  # Positive for long, negative for short

    for i in range(1, len(signals)):
        signal = signals.iloc[i]
        price_today = prices.iloc[i]
        price_yesterday = prices.iloc[i-1]

        for col in signals.columns:
            cash = portfolio['cash'].iloc[i-1]
            holding_value = portfolio['holdings'].iloc[i-1]

            if signal[col] == 1 and position_shares == 0:
                # Enter long
                shares = (cash * 0.95) / price_today[col]
                cost = shares * price_today[col] * (1 + commission)
                position_shares = shares
                portfolio.at[portfolio.index[i], 'cash'] = cash - cost
                portfolio.at[portfolio.index[i], 'holdings'] = shares * price_today[col]
                
            elif signal[col] == -1 and position_shares == 0:
                # Enter short
                shares = (cash * 0.95) / price_today[col]
                proceeds = shares * price_today[col] * (1 - commission)
                position_shares = -shares
                portfolio.at[portfolio.index[i], 'cash'] = cash + proceeds
                portfolio.at[portfolio.index[i], 'holdings'] = -shares * price_today[col]
                
            elif (signal[col] == 0) and (position_shares != 0):
                # Close any open position (long or short)
                if position_shares > 0:
                    # Closing long
                    proceeds = position_shares * price_today[col] * (1 - commission)
                else:
                    # Closing short
                    proceeds = abs(position_shares) * price_today[col] * (1 - commission)
                    
                portfolio.at[portfolio.index[i], 'cash'] = cash + proceeds
                portfolio.at[portfolio.index[i], 'holdings'] = 0
                position_shares = 0
            else:
                # Hold position
                portfolio.at[portfolio.index[i], 'cash'] = cash
                portfolio.at[portfolio.index[i], 'holdings'] = position_shares * price_today[col]
        
        portfolio.at[portfolio.index[i], 'total'] = portfolio.at[portfolio.index[i], 'cash'] + portfolio.at[portfolio.index[i], 'holdings']
        portfolio.at[portfolio.index[i], 'returns'] = portfolio['total'].iloc[i] / portfolio['total'].iloc[i-1] - 1

    return portfolio


def calculate_performance(portfolio, risk_free_rate=0.02/252):
    """
    Calculate performance metrics for the portfolio.
    """
    returns = portfolio['returns'].dropna()
    
    total_return = portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    max_drawdown = (portfolio['total'] / portfolio['total'].cummax() - 1).min()
    win_rate = (returns > 0).mean()
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if abs(returns[returns < 0].sum()) > 0 else np.inf
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor
    }

def plot_results(portfolio, title='Portfolio Performance', save_path=None):
    """
    Plot portfolio value and drawdown.
    """
    fig, ax = plt.subplots(2, 1, figsize=(14,10), gridspec_kw={'height_ratios': [2,1]})
    
    # Portfolio value
    ax[0].plot(portfolio.index, portfolio['total'])
    ax[0].set_title(title)
    ax[0].set_ylabel('Portfolio Value ($)')
    ax[0].grid(True, alpha=0.3)
    
    # Drawdown
    peak = portfolio['total'].cummax()
    drawdown = portfolio['total'] / peak - 1
    ax[1].plot(portfolio.index, drawdown, color='red')
    ax[1].set_title('Drawdown')
    ax[1].set_ylabel('Drawdown (%)')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_trading_system(csv_folder=r'/Users/keshavwarrier/water-well-trader/model-runs', threshold_pct=0.02, smoothing_window=3):
    """
    Run the full trading system: load data, generate signals, backtest, evaluate, plot.
    """
    predictions, actuals = load_model_data(csv_folder)
    
    signals = generate_signals(predictions, actuals, threshold_pct=threshold_pct, smoothing_window=smoothing_window)
    portfolio = backtest_strategy(signals, actuals)
    metrics = calculate_performance(portfolio)
    
    print('Performance Metrics:')
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    plot_results(portfolio, title='Trading System Results')
    return signals, portfolio, metrics

if __name__ == "__main__":
    run_trading_system(threshold_pct=0.02, smoothing_window=3)
