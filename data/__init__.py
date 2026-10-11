"""Market data acquisition, financial time-series preprocessing, and feature engineering."""

from .constants import RISK_UNACCEPTANCE_VALUE, TRADING_DAYS_PER_YEAR
from .download_data import download_tickers_history
from .news_provider import HistoricalNewsProvider
from .processors import (
    PandasData,
    calcualte_structural_cost_coupling_value,
    corwin_shultz_half_spread,
    daily_spread_pct,
    get_market_weights,
    get_portfolio_exp_vol,
    get_slippage,
    intraday_returns_prc,
    log_returns,
    overnight_gaps_prc,
    parkinson_rolling_vol_daily,
    rolling_daily_spreads_mean,
    rolling_overnight_gaps_std,
    rolling_vol_daily,
)

__all__ = [
    "RISK_UNACCEPTANCE_VALUE",
    "TRADING_DAYS_PER_YEAR",
    "HistoricalNewsProvider",
    "PandasData",
    "calcualte_structural_cost_coupling_value",
    "corwin_shultz_half_spread",
    "daily_spread_pct",
    "download_tickers_history",
    "get_market_weights",
    "get_portfolio_exp_vol",
    "get_slippage",
    "intraday_returns_prc",
    "log_returns",
    "overnight_gaps_prc",
    "parkinson_rolling_vol_daily",
    "rolling_daily_spreads_mean",
    "rolling_overnight_gaps_std",
    "rolling_vol_daily",
]
