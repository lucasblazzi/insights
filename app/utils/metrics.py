import pandas as pd
import numpy as np


def returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change().fillna(0)


def cumulative_returns(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.add(1).cumprod().sub(1)


def total_returns(cum_rets: pd.DataFrame) -> pd.DataFrame:
    return cum_rets.tail(1)


def annualized_returns(cum_rets: pd.DataFrame, periods_in_year: int = 252) -> pd.DataFrame:
    total_rets = total_returns(cum_rets)
    return (1 + total_rets) ** (periods_in_year/cum_rets.shape[0]) - 1


def volatility(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.std()


def annualized_volatility(rets: pd.DataFrame, periods_in_year: int = 252) -> pd.DataFrame:
    vol = volatility(rets)
    return vol * np.sqrt(periods_in_year)


def sharpe(rets: pd.DataFrame) -> pd.DataFrame:
    riskfree_rate = 0.03
    excess_return = annualized_returns(rets) - riskfree_rate
    annualized_vol = annualized_volatility(rets)
    return excess_return / annualized_vol


def drawdown(cum_rets: pd.DataFrame):
    previous_peaks = cum_rets.cummax()
    return (cum_rets - previous_peaks) / previous_peaks


def portfolio_returns(rets: pd.DataFrame, weights: list) -> pd.DataFrame:
    portfolio_rets = rets.copy()
    portfolio_rets["Portfolio"] = (rets * weights).sum(axis=1)
    return portfolio_rets