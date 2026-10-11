from typing import TypeVar

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import Settings

PandasData = TypeVar("PandasData", pd.DataFrame, pd.Series)


def log_returns(stocks: PandasData) -> PandasData:
    """Calculate logarithmic daily returns for stock price data.

    Parameters
    ----------
    stocks : pd.DataFrame | pd.Series
        Price data containing 'Close' prices or MultiIndex ticker history.

    Returns
    -------
    pd.DataFrame | pd.Series
        Daily log returns indexed by Date.

    Raises
    ------
    ValueError
        If stocks input is None.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    stocks = stocks.dropna()
    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and "Close" in stocks.columns:
        close_prices = stocks["Close"]

    else:
        close_prices = stocks

    log_returns = np.log(close_prices).diff().dropna()  # type: ignore

    return log_returns


def overnight_gaps_prc(stocks: pd.DataFrame) -> pd.DataFrame:
    """Calculate the overnight price gap percentage between previous close and current open prices.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'Open' and 'Close' prices.

    Returns
    -------
    pd.DataFrame
        Overnight price gaps in percentage (%).

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'Open' and 'Close' columns.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    stocks = stocks.dropna()
    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        open_prices = stocks.xs("Open", axis=1, level=level_name)
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Open", "Close"}.issubset(
        stocks.columns
    ):
        close_prices = stocks["Close"]
        open_prices = stocks["Open"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing both 'Open' and 'Close' prices."
        )

    overnight_gap = np.log(open_prices / close_prices.shift(1)) * 100

    return overnight_gap.dropna()  # type: ignore


def rolling_overnight_gaps_std(stocks: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate the rolling standard deviation of overnight price gaps over a specified window.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'Open' and 'Close' prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.DataFrame
        Rolling standard deviation of overnight gaps.
    """
    overnight_gap = overnight_gaps_prc(stocks)

    return overnight_gap.rolling(window).std()


def intraday_returns_prc(stocks: pd.DataFrame) -> pd.DataFrame:
    """Calculate the intraday percentage log return between open and close prices.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'Open' and 'Close' prices.

    Returns
    -------
    pd.DataFrame
        Intraday percentage log returns (%).

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'Open' and 'Close' columns.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    stocks = stocks.dropna()
    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        open_prices = stocks.xs("Open", axis=1, level=level_name)
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Open", "Close"}.issubset(
        stocks.columns
    ):
        close_prices = stocks["Close"]
        open_prices = stocks["Open"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing both 'Open' and 'Close' prices."
        )

    intraday_return = np.log(close_prices / open_prices) * 100

    return intraday_return.dropna()  # type: ignore


def daily_spread_pct(stocks: pd.DataFrame) -> pd.DataFrame:
    """Calculate the relative daily high-low price spread as a percentage of the close price.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'High', 'Low', and 'Close' prices.

    Returns
    -------
    pd.DataFrame
        Relative daily high-low price spread (%).

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'High', 'Low', and 'Close' columns.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    stocks = stocks.dropna()
    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        close_prices = stocks.xs("Close", axis=1, level=level_name)
        low_prices = stocks.xs("Low", axis=1, level=level_name)
        high_prices = stocks.xs("High", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Low", "High", "Close"}.issubset(
        stocks.columns
    ):
        close_prices = stocks["Close"]
        low_prices = stocks["Low"]
        high_prices = stocks["High"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing 'Low', 'High' and 'Close' prices."
        )

    intraday_spread = ((high_prices - low_prices) / close_prices) * 100

    return intraday_spread.dropna()  # type: ignore


def rolling_daily_spreads_mean(stocks: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate the rolling mean of daily high-low price spreads over a specified window.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'High', 'Low', and 'Close' prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.DataFrame
        Rolling mean of high-low price spreads.
    """
    intraday_spread = daily_spread_pct(stocks).dropna()

    return intraday_spread.rolling(window).mean()


def rolling_vol_daily(stocks: PandasData, window: int) -> PandasData:
    """Calculate rolling daily volatility (standard deviation of log returns) over a specified window.

    Parameters
    ----------
    stocks : pd.DataFrame | pd.Series
        Price data containing stock prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.DataFrame | pd.Series
        Rolling daily volatility indexed by Date.
    """
    returns = log_returns(stocks)
    return returns.rolling(window).std().dropna()


def parkinson_rolling_vol_daily(stocks: pd.DataFrame, window: int) -> pd.Series:
    """Calculate the Parkinson daily volatility estimate over a specified rolling window.

    Uses high and low prices to estimate the latest daily volatility per asset based on
    Parkinson's (1980) extreme value volatility estimator.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'High' and 'Low' prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.Series
        Latest estimated daily volatility per asset.

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'Low' and 'High' prices.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    stocks = stocks.dropna()
    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        low_prices = stocks.xs("Low", axis=1, level=level_name)
        high_prices = stocks.xs("High", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Low", "High", "Close"}.issubset(
        stocks.columns
    ):
        low_prices = stocks["Low"]
        high_prices = stocks["High"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing 'Low', 'High' and 'Close' prices."
        )

    hl_ratio = pd.DataFrame(np.log(high_prices / low_prices) ** 2)
    parkinson_rolling_vol = np.sqrt(
        hl_ratio.rolling(window=window).mean().iloc[-1] / (4 * np.log(2))
    )

    return parkinson_rolling_vol


def get_portfolio_exp_vol(
    log_return: pd.DataFrame,
    weights: np.ndarray | None,
    settings: Settings | None = None,
) -> float:
    """Calculate the expected annualized portfolio volatility.

    If portfolio weights are provided, computes the historical realized annualized volatility
    of the weighted portfolio returns. If weights are None, computes the average annualized
    volatility across all assets using their covariance matrix.

    Parameters
    ----------
    log_return : pd.DataFrame
        Asset logarithmic returns.
    weights : np.ndarray | None
        Current portfolio weight allocation array, or None.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    float
        Expected annualized portfolio volatility.
    """
    if settings is None:
        settings = Settings.from_toml()

    if weights is not None:
        current_port_returns = log_return.values @ weights
        expected_annyal_vol = np.std(current_port_returns) * np.sqrt(
            settings.portfolio.trading_days_per_year
        )
    else:
        expected_annyal_vol = np.sqrt(
            np.mean(log_return.cov() * settings.portfolio.trading_days_per_year)
        )

    return expected_annyal_vol


def get_slippage(
    tickers_df: pd.DataFrame, delta_weight: float, portfolio_value: float
) -> pd.Series | float:
    """Calculate estimated trading slippage per asset based on market volume and Parkinson volatility.

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Price and volume data containing 'Close', 'Volume', 'High', and 'Low' prices.
    delta_weight : float
        Turnover / rebalancing weight delta.
    portfolio_value : float
        Total portfolio dollar value.

    Returns
    -------
    pd.Series | float
        Estimated slippage per asset (or 0.0 if delta_weight is negligible).
    """
    close = tickers_df.xs("Close", axis=1, level="Price")
    volume = tickers_df.xs("Volume", axis=1, level="Price")

    if abs(delta_weight) < 1e-6:
        return 0.0

    window = 21
    nu = 0.2
    order_dollar_volume = np.abs(delta_weight) * portfolio_value
    dollar_vol = close * volume
    avg_dollar_vol = dollar_vol.rolling(window).mean().iloc[-1]

    # Parkinson daily volatility
    daily_vol = parkinson_rolling_vol_daily(tickers_df, window)

    slippage = nu * daily_vol * np.sqrt(order_dollar_volume / avg_dollar_vol)

    return slippage


def corwin_shultz_half_spread(tickers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Corwin-Schultz (2012) bid-ask half-spread estimate from High and Low prices.

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Price data containing 'High' and 'Low' prices.

    Returns
    -------
    pd.DataFrame
        Rolling half-spread estimates.
    """
    high = tickers_df.xs("High", axis=1, level="Price")
    low = tickers_df.xs("Low", axis=1, level="Price")

    window = 21
    beta = np.log(high / low) ** 2 + np.log(high.shift(1) / low.shift(1)) ** 2
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    alpha_1 = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
    alpha_2 = np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    alpha = alpha_1 - alpha_2

    spread = (2 * (np.exp(alpha) - 1)) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0.0)

    half_spread = (spread / 2.0).rolling(window=window, min_periods=5).mean()

    return half_spread


def calcualte_structural_cost_coupling_value(
    data: pd.DataFrame,
    log_return: pd.DataFrame,
    turnover: float,
    portfolio_value: float,
    broker_commission: float,
    rebalances_per_year: int,
    weights: np.ndarray | None,
    settings: Settings | None = None,
) -> float:
    """Calculate the structural turnover penalty multiplier based on transaction friction and portfolio volatility.

    Aggregates broker commission, bid-ask spread, and slippage into total trading cost,
    and scales it relative to annual rebalance frequency, expected volatility, and risk aversion.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data containing OHLCV for all assets.
    log_return : pd.DataFrame
        Asset logarithmic returns.
    turnover : float
        Expected portfolio turnover.
    portfolio_value : float
        Current total portfolio equity value.
    broker_commission : float
        Proportional broker commission rate per unit turnover.
    rebalances_per_year : int
        Number of rebalancing periods per year.
    weights : np.ndarray | None
        Current portfolio weights vector, or None.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    float
        Turnover penalty coefficient (l1_coeff).
    """
    if settings is None:
        settings = Settings.from_toml()

    half_spread = corwin_shultz_half_spread(data)
    slippage = get_slippage(data, turnover, portfolio_value)

    c_trade = broker_commission + slippage + half_spread
    c_trade = np.mean(c_trade)

    if weights is not None:
        current_port_returns = log_return.values @ weights
        expected_annyal_vol = np.std(current_port_returns) * np.sqrt(
            settings.portfolio.trading_days_per_year
        )
    else:
        expected_annyal_vol = np.sqrt(
            np.mean(log_return.cov() * settings.portfolio.trading_days_per_year)
        )

    expected_annyal_vol = np.clip(expected_annyal_vol, 0.10, 0.50)

    penalty = (
        c_trade
        * (rebalances_per_year / expected_annyal_vol)
        * settings.portfolio.risk_unacceptance_value
    )

    return penalty


def get_market_weights(stocks: PandasData) -> pd.DataFrame:
    """Calculate historical market capitalization weights for the given assets.

    Extracts close prices from stock data, fetches the number of shares outstanding
    for each ticker via Yahoo Finance, computes the historical market capitalization,
    and normalizes it across all assets at each timestamp so weights sum to 1.0.

    Parameters
    ----------
    stocks : pd.DataFrame | pd.Series
        Historical price data containing 'Close' prices or MultiIndex columns per ticker.

    Returns
    -------
    pd.DataFrame
        Historical market capitalization weights per ticker indexed by Date.

    Raises
    ------
    ValueError
        If stocks input is None.
    """
    shares_dict = {}

    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    stocks = stocks.dropna()
    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and "Close" in stocks.columns:
        close_prices = stocks["Close"]

    else:
        close_prices = stocks

    tickers = list(close_prices.columns)
    for ticker in tickers:
        stock = yf.Ticker(ticker)

        # sharesOutstanding or floatShares
        shares_dict[ticker] = stock.info.get("sharesOutstanding", 1.0)

    shares = pd.Series(shares_dict)
    market_cap_df = close_prices.multiply(shares, axis=1)

    w_market_history = market_cap_df.divide(market_cap_df.sum(axis=1), axis=0)

    return w_market_history
