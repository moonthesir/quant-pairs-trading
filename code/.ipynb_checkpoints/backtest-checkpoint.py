# code/backtest.py
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def fit_beta_ols(x, y):
    """Fit single beta coefficient by OLS: x ~ a + beta*y"""
    # drop NaNs and align
    df = pd.concat([x, y], axis=1).dropna()
    X = add_constant(df.iloc[:,1])  # y
    model = OLS(df.iloc[:,0], X).fit()
    intercept = model.params[0]
    beta = model.params[1]
    return beta, intercept

def compute_spread(x, y, beta, intercept=0.0):
    """Compute spread series: x - (beta*y + intercept)"""
    return x - (beta * y + intercept)

def compute_zscore(spread, lookback):
    roll_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    roll_std = spread.rolling(window=lookback, min_periods=lookback).std()
    return (spread - roll_mean) / roll_std

def compute_trade_pnl(entry_price_A, entry_price_B, exit_price_A, exit_price_B, beta,
                      position_sign=1, units=1.0, cost_per_trade=1.0, slippage_pct=0.0001):
    """
    Simplified PnL for a single 'unit' of spread with beta from OLS.
    Position_sign: +1 means LONG spread (long A, short beta*B)
                   -1 means SHORT spread (short A, long beta*B)
    units: number of spread units (1.0 default)
    PnL formula: units * ( (exitA - entryA) - beta*(exitB - entryB) ) * position_sign
    Costs: we subtract cost_per_trade twice (entry+exit) and slippage estimated as
           slippage_pct * (abs(traded_dollar_amount)) for entry+exit (approx).
    """
    # raw pnl per unit
    raw_pnl = (exit_price_A - entry_price_A) - beta * (exit_price_B - entry_price_B)
    pnl = units * raw_pnl * position_sign

    # approximate slippage cost: sum of dollar volumes traded * slippage_pct
    # approximate traded dollar amount per leg on entry and exit
    traded_dollars_entry = abs(entry_price_A) + abs(beta * entry_price_B)
    traded_dollars_exit = abs(exit_price_A) + abs(beta * exit_price_B)
    slippage_cost = slippage_pct * (traded_dollars_entry + traded_dollars_exit)

    total_costs = 2 * cost_per_trade + slippage_cost
    pnl_after_costs = pnl - total_costs
    return pnl_after_costs

def backtest_pairs(data, params, verbose=False):
    """
    data: DataFrame with columns ['AAPL','MSFT'] (prices)
    params: dict with keys:
       'lookback' (int), 'entry_z' (float), 'exit_z' (float),
       'max_hold' (int), 'cost_per_trade' (float), 'slippage_pct' (float)
    Returns: result_df (daily), trades_df (trade-level)
    """
    # copy input
    x = data.iloc[:,0].copy()  # A
    y = data.iloc[:,1].copy()  # B

    # fit beta on full sample (MVP). Note: rolling-beta is a later improvement.
    beta, intercept = fit_beta_ols(x, y)
    spread = compute_spread(x, y, beta, intercept)
    z = compute_zscore(spread, params['lookback'])

    # prepare daily arrays
    dates = data.index
    daily_position = np.zeros(len(dates), dtype=int)
    daily_pnl = np.zeros(len(dates), dtype=float)

    trades = []  # list of trade dicts

    position = 0
    entry_index = None
    entry_price = (None, None)
    hold = 0
    units = 1.0

    for t in range(len(dates)):
        if np.isnan(z.iloc[t]):
            daily_position[t] = position
            continue

        current_z = z.iloc[t]
        priceA = x.iloc[t]
        priceB = y.iloc[t]

        # ENTRY
        if position == 0:
            if current_z > params['entry_z']:
                # SHORT spread: sell A, buy beta*B
                position = -1
                entry_index = t
                entry_price = (priceA, priceB)
                hold = 0
                if verbose: print(f"Entry SHORT at {dates[t]} z={current_z:.2f}")
            elif current_z < -params['entry_z']:
                # LONG spread: buy A, sell beta*B
                position = 1
                entry_index = t
                entry_price = (priceA, priceB)
                hold = 0
                if verbose: print(f"Entry LONG at {dates[t]} z={current_z:.2f}")

        # MANAGE/EXIT
        elif position != 0:
            hold += 1
            exit_trade = False
            # exit if z crosses threshold toward zero or forced by max_hold
            if abs(current_z) < params['exit_z']:
                exit_trade = True
                reason = 'z_cross'
            if hold >= params['max_hold']:
                exit_trade = True
                reason = 'max_hold'

            if exit_trade:
                exit_index = t
                exit_price = (priceA, priceB)
                pnl_trade = compute_trade_pnl(
                    entry_price_A=entry_price[0],
                    entry_price_B=entry_price[1],
                    exit_price_A=exit_price[0],
                    exit_price_B=exit_price[1],
                    beta=beta,
                    position_sign=position,
                    units=units,
                    cost_per_trade=params.get('cost_per_trade',1.0),
                    slippage_pct=params.get('slippage_pct',0.0001)
                )
                # record trade
                trade = {
                    'entry_date': dates[entry_index],
                    'exit_date': dates[exit_index],
                    'entry_price_A': entry_price[0],
                    'entry_price_B': entry_price[1],
                    'exit_price_A': exit_price[0],
                    'exit_price_B': exit_price[1],
                    'position': position,
                    'pnl': pnl_trade,
                    'hold_days': hold,
                    'reason': reason
                }
                trades.append(trade)
                # apply realized pnl on exit date (simple approximation)
                daily_pnl[exit_index] += pnl_trade

                # reset position
                position = 0
                entry_index = None
                entry_price = (None, None)
                hold = 0

        daily_position[t] = position

    result_df = pd.DataFrame({
        'price_A': x,
        'price_B': y,
        'spread': spread,
        'z': z,
        'position': daily_position,
        'daily_pnl': daily_pnl
    }, index=dates)

    trades_df = pd.DataFrame(trades)
    return result_df, trades_df

# small helper to run a single parameter set conveniently
def run_single(data, params):
    res, trades = backtest_pairs(data, params, verbose=False)
    # equity curve from daily pnl
    res['cum_pnl'] = res['daily_pnl'].cumsum()
    return res, trades