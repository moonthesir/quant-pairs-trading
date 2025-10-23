# backtest.py (skeleton, expand as you go)
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def fit_beta(x, y):
    # regress x on y to get beta for spread = x - beta*y
    y_const = add_constant(y)
    model = OLS(x, y_const).fit()
    beta = model.params[1]
    intercept = model.params[0]
    return beta, intercept

def compute_spread(x, y, beta, intercept=0.0):
    return x - (beta * y + intercept)

def compute_zscore(spread, lookback):
    roll_mean = spread.rolling(window=lookback).mean()
    roll_std = spread.rolling(window=lookback).std()
    return (spread - roll_mean) / roll_std

def backtest_pairs(data, params):
    """
    data: DataFrame with columns ['AAPL','MSFT'] indexed by date
    params: dict with keys: lookback, entry_z, exit_z, max_hold, cost_per_trade, slippage_pct
    """
    x = data['AAPL']
    y = data['MSFT']

    beta, intercept = fit_beta(x, y)
    spread = compute_spread(x, y, beta, intercept)
    z = compute_zscore(spread, params['lookback'])

    positions = []   # +1 long spread (long A, short B), -1 short spread
    entry_dates = []
    pnl = []
    position = 0
    entry_price = 0.0
    hold = 0

    for t in range(len(z)):
        if np.isnan(z.iloc[t]):
            pnl.append(0)
            positions.append(position)
            continue

        current_z = z.iloc[t]
        priceA = x.iloc[t]
        priceB = y.iloc[t]

        # entry logic
        if position == 0:
            if current_z > params['entry_z']:
                # short spread: sell A, buy B
                position = -1
                entry_price = (priceA, priceB)
                hold = 0
                # account for cost on entry
            elif current_z < -params['entry_z']:
                position = 1
                entry_price = (priceA, priceB)
                hold = 0

        # exit logic
        elif position != 0:
            hold += 1
            if (abs(current_z) < params['exit_z']) or (hold >= params['max_hold']):
                # close position -> compute PnL
                exit_price = (priceA, priceB)
                # compute pnl: simple PnL calc for dollar-neutral unit notional; adjust for slippage/costs
                # add to pnl list
                position = 0
                entry_price = 0
                hold = 0

        positions.append(position)
        pnl.append(0)  # replace with computed daily pnl/dummy for now

    result_df = pd.DataFrame({
        'spread': spread,
        'z': z,
        'position': positions,
        'pnl': pnl
    }, index=data.index)

    # compute cumulative returns, metrics (add later)
    return result_df
