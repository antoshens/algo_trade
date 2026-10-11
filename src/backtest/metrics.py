"""
Performance and risk metrics for portfolio backtesting.

This module provides financial metric calculation functions used to assess portfolio
performance during backtesting, including CAGR, Maximum Drawdown (MDD), Sharpe ratio,
rolling Sharpe ratio, Sortino ratio, rolling Sortino ratio, and Calmar ratio.
"""

from datetime import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import Settings
from src.portfolio import get_risk_free_rate


def calculate_cagr(
    port_value: np.ndarray,
    testing_period: float,
    settings: Settings | None = None,
) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) for a portfolio.

    Parameters
    ----------
    port_value : np.ndarray
        Array of portfolio equity values over time.
    testing_period : float
        Total duration of the backtest period in trading days.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    float
        Annualized compound growth rate as a decimal.
    """
    if settings is None:
        settings = Settings.from_toml()

    trading_days_per_year = settings.portfolio.trading_days_per_year

    cagr = (port_value[-1] / port_value[0]) ** (
        trading_days_per_year / (testing_period - 1)
    ) - 1

    return cagr


def calculate_mdd(port_value: np.ndarray) -> float:
    """
    Calculate the Maximum Drawdown (MDD) of a portfolio equity curve.

    Maximum drawdown measures the largest peak-to-trough decline observed
    before a new peak is achieved.

    Parameters
    ----------
    port_value : np.ndarray
        Array of portfolio equity values over time.

    Returns
    -------
    float
        Maximum drawdown as a positive decimal value.
    """
    port_drawdown = (
        port_value - np.maximum.accumulate(port_value)
    ) / np.maximum.accumulate(port_value)

    return np.abs(np.min(port_drawdown))


def calculate_rolling_sharpe(
    daily_returns: np.ndarray,
    start_date: datetime,
    end_date: datetime,
    window: int,
    settings: Settings | None = None,
) -> np.ndarray:
    """
    Calculate the rolling annualized Sharpe ratio from daily returns over sliding windows.

    Parameters
    ----------
    daily_returns : np.ndarray
        1D array of daily portfolio returns.
    start_date : datetime
        Start date of the evaluation period to determine the risk-free rate.
    end_date : datetime
        End date of the evaluation period to determine the risk-free rate.
    window : int
        Size of the rolling window in trading days.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    np.ndarray
        1D array of rolling annualized Sharpe ratios matching the length of
        `daily_returns`, with initial positions prior to a full window populated
        with NaN.
    """
    if settings is None:
        settings = Settings.from_toml()

    trading_days_per_year = settings.portfolio.trading_days_per_year

    n = daily_returns.shape[0]
    rolling_sharpe = np.full(n, np.nan, dtype=float)

    if n < window:
        return rolling_sharpe

    daily_rf = (
        get_risk_free_rate("T_BILLS", start_date, end_date) / trading_days_per_year
    )

    excess_ret = daily_returns - daily_rf
    windows_excess = sliding_window_view(excess_ret, window_shape=window)

    rolling_sharpe[window - 1 :] = (
        (windows_excess.mean(axis=1) - daily_rf)
        / (windows_excess.std(axis=1) + 1e-8)
        * np.sqrt(trading_days_per_year)
    )

    return rolling_sharpe


def calculate_sharpe(
    daily_returns: np.ndarray,
    start_date: datetime,
    end_date: datetime,
    settings: Settings | None = None,
) -> float:
    """
    Calculate the annualized Sharpe ratio for a series of daily returns.

    Measures risk-adjusted performance by penalizing total return variability
    relative to a risk-free benchmark.

    Parameters
    ----------
    daily_returns : np.ndarray
        Array of daily portfolio returns.
    start_date : datetime
        Start date of the evaluation period to determine the risk-free rate.
    end_date : datetime
        End date of the evaluation period to determine the risk-free rate.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if settings is None:
        settings = Settings.from_toml()

    trading_days_per_year = settings.portfolio.trading_days_per_year

    daily_rf = (
        get_risk_free_rate("T_BILLS", start_date, end_date) / trading_days_per_year
    )

    sharpe = (
        (daily_returns - daily_rf).mean() / (daily_returns.std() + 1e-8)
    ) * np.sqrt(trading_days_per_year)

    return sharpe


def calculate_rolling_sortino(
    daily_returns: np.ndarray,
    start_date: datetime,
    end_date: datetime,
    window: int,
    settings: Settings | None = None,
) -> np.ndarray:
    """
    Calculate the rolling annualized Sortino ratio over sliding windows.

    Unlike the Sharpe ratio, Sortino penalizes only downside volatility (returns below
    the risk-free rate).

    Parameters
    ----------
    daily_returns : np.ndarray
        1D array of daily portfolio returns.
    start_date : datetime
        Start date of the evaluation period to determine the risk-free rate.
    end_date : datetime
        End date of the evaluation period to determine the risk-free rate.
    window : int
        Size of the rolling window in trading days.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    np.ndarray
        Array of rolling annualized Sortino ratios matching the length of `daily_returns`,
        with initial positions prior to a full window populated with NaN.
    """
    if settings is None:
        settings = Settings.from_toml()

    trading_days_per_year = settings.portfolio.trading_days_per_year

    daily_rf = (
        get_risk_free_rate("T_BILLS", start_date, end_date) / trading_days_per_year
    )

    n = daily_returns.shape[0]
    rolling_sortino = np.full(n, np.nan, dtype=float)

    if n < window:
        return rolling_sortino

    excess_ret = daily_returns - daily_rf
    downside_sq = np.minimum(0.0, excess_ret) ** 2

    windows_excess = sliding_window_view(excess_ret, window_shape=window)
    windows_downside_sq = sliding_window_view(downside_sq, window_shape=window)

    rolling_excess_mean = windows_excess.mean(axis=1)
    rolling_downside_vol = np.sqrt(windows_downside_sq.mean(axis=1)) + 1e-8

    rolling_sortino[window - 1 :] = (
        rolling_excess_mean / rolling_downside_vol
    ) * np.sqrt(trading_days_per_year)

    return rolling_sortino


def calculate_sortino(
    daily_returns: np.ndarray,
    start_date: datetime,
    end_date: datetime,
    settings: Settings | None = None,
) -> float:
    """
    Calculate the annualized Sortino ratio for a series of daily returns.

    Measures risk-adjusted performance by penalizing only downside volatility
    relative to the risk-free benchmark.

    Parameters
    ----------
    daily_returns : np.ndarray
        Array of daily portfolio returns.
    start_date : datetime
        Start date of the evaluation period to determine the risk-free rate.
    end_date : datetime
        End date of the evaluation period to determine the risk-free rate.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    if settings is None:
        settings = Settings.from_toml()

    trading_days_per_year = settings.portfolio.trading_days_per_year

    daily_rf = (
        get_risk_free_rate("T_BILLS", start_date, end_date) / trading_days_per_year
    )

    negative_deviation = np.minimum(0.0, daily_returns - daily_rf) ** 2
    downside_vol = np.sqrt((negative_deviation).mean(axis=0)) + 1e-8
    sortino = ((daily_returns - daily_rf).mean() / downside_vol) * np.sqrt(
        trading_days_per_year
    )

    return sortino


def calculate_calmar(cagr: float, mdd: float) -> float:
    """
    Calculate the Calmar ratio.

    Measures return relative to drawdown risk by taking the ratio of
    Compound Annual Growth Rate (CAGR) to Maximum Drawdown (MDD).

    Parameters
    ----------
    cagr : float
        Compound Annual Growth Rate as a decimal.
    mdd : float
        Maximum Drawdown as a decimal.

    Returns
    -------
    float
        Calmar ratio.
    """
    return cagr / (np.abs(mdd) + 1e-8)
