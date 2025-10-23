import numpy as np
import pandas as pd

def compute_metrics_from_daily_pnl(daily_pnl_series, initial_capital=10000.0):
    equity = initial_capital + daily_pnl_series.cumsum()
    daily_returns = equity.pct_change().fillna(0)

    total_return = (equity.iloc[-1] / initial_capital) - 1.0
    years = len(daily_returns)/252.0
    cagr = (equity.iloc[-1]/initial_capital)**(1.0/years) - 1 if years>0 else np.nan
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean()*252)/ann_vol if ann_vol>0 else np.nan
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()

    return {
        'initial_capital': initial_capital,
        'ending_capital': equity.iloc[-1],
        'total_return': total_return,
        'cagr': cagr,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }