# Project 1: Pairs Trading Backtest (AAPL–MSFT)

## Overview
This repository contains a Python backtesting pipeline for a **mean-reversion pairs trading strategy**.  
The project computes the OLS beta between two assets, constructs the spread, generates z-score signals, simulates trades with realistic transaction costs, performs parameter search, and optionally includes walk-forward validation.

## Contents
- `code/`
  - `backtest.py` — backtest engine (compute beta, spread, trades, PnL).  
  - `metrics.py` — helper functions to compute daily returns, Sharpe, CAGR, drawdown, etc.  
  - `PairsTrading_Data.ipynb` — main notebook: single-run backtest, grid search, plots.  
  - `optimize_notebook.ipynb` — optional notebook for extended parameter search/analysis.  
- `data/` — CSV file with price data (`pairs_data.csv`).  
- `results/` — plots and parameter search results.  
- `docs/Project1_Brief.md` — project write-up: Abstract, Method, Results, Robustness, Limitations.  

## Installation
```bash
# 1. Clone the repo
git clone https://github.com/moonthesir/quant-pairs-trading.git
cd quant-pairs-trading

# 2. Activate your conda environment
conda activate pow_proj1

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch Jupyter notebook
jupyter notebook
