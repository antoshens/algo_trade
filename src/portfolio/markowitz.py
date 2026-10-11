"""
Markowitz Portfolio Optimization Module.

Provides classical mean-variance optimization (efficient frontier), maximum Sharpe ratio
(tangency portfolio), and maximum Sortino ratio (downside risk) optimization routines.
Supports:
- Covariance models: Sample covariance ('CLASSIC'), Ledoit-Wolf shrinkage ('LEDOIT_WOLF'),
  and univariate (E)GARCH volatility modeling ('GARCH', 'EGARCH').
- Return models: Implied equilibrium / reverse optimization returns ('HISTORICAL') and Black-Litterman model ('BLACK_LITTERMAN').
- Risk-free rate fetching from FRED ('T_BILLS', 'TREASURY_NOTES', 'SOFR').
- L1 rebalancing turnover penalties to control portfolio drift and transaction costs.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from data.constants import RISK_UNACCEPTANCE_VALUE, TRADING_DAYS_PER_YEAR
from data.processors import get_market_weights, log_returns

from .black_litterman import black_litterman
from .garch import garch

RiskFreeRateBase = Literal["T_BILLS", "TREASURY_NOTES", "SOFR"]
"""
The benchmark asset used to calculate the risk-free rate.

Options:
- 'T_BILLS': 13-week Treasury Bill (^IRX).
- 'TREASURY_NOTES': 10-year Treasury Note (^TNX).
- 'SOFR': Secured Overnight Financing Rate (SR3=F).
"""

OptimizationType = Literal["BACKTEST", "LIVE"]
"""
Determines the date range for fetching the risk-free rate.

Options:
- 'BACKTEST': Uses the start and end dates of the provided historical dataset.
- 'LIVE': Uses only the most recent date available in the dataset.
"""

CovarianceModel = Literal["CLASSIC", "LEDOIT_WOLF", "GARCH", "EGARCH"]
"""
Covariance estimation model to use.

Options:
- 'CLASSIC': Sample covariance matrix scaled by trading days.
- 'LEDOIT_WOLF': Ledoit-Wolf shrinkage covariance estimator.
- 'GARCH': Univariate GARCH volatility modeling.
- 'EGARCH': An Exponential GARCH modeling.
"""

ReturnsModel = Literal["HISTORICAL", "BLACK_LITTERMAN"]
"""
Expected returns estimation model to use.

Options:
- 'HISTORICAL': Implied equilibrium returns derived from market weights and covariance (or GARCH return forecasts).
- 'BLACK_LITTERMAN': Black-Litterman model incorporating market equilibrium and subjective investor views.
"""


@dataclass
class SharpeRatio:
    """
    Data structure holding the results of the Maximum Sharpe Ratio (Tangency Portfolio) optimization.

    Attributes
    ----------
    max_sharpe : float
        The maximum annualized Sharpe ratio achieved: (Return - RiskFreeRate) / Volatility.
    tangency_return : float
        The expected annualized return of the tangency portfolio.
    tangency_vol : float
        The expected annualized total volatility (standard deviation) of the tangency portfolio.
    """

    max_sharpe: float
    tangency_return: float
    tangency_vol: float


@dataclass
class SortinoRatio:
    """
    Data structure holding the results of the Maximum Sortino Ratio optimization.

    Attributes
    ----------
    max_sortino : float
        The maximum annualized Sortino ratio achieved: (Return - RiskFreeRate) / DownsideVolatility.
    tangency_return : float
        The expected annualized return of the tangency portfolio.
    tangency_vol : float
        The expected annualized downside volatility (semi-deviation below daily risk-free rate).
    """

    max_sortino: float
    tangency_return: float
    tangency_vol: float


def get_risk_free_rate(
    base: RiskFreeRateBase,
    start_date: datetime,
    end_date: datetime,
    opt_type: OptimizationType = "BACKTEST",
) -> float:
    """
    Fetches and calculates the annualized risk-free rate for a given period and benchmark from FRED.

    Parameters
    ----------
    base : RiskFreeRateBase
        The benchmark asset used for the risk-free rate:
        - 'T_BILLS': 13-week Treasury Bill (FRED: 'DGS3MO').
        - 'TREASURY_NOTES': 10-year Treasury Note (FRED: 'DGS10').
        - 'SOFR': Secured Overnight Financing Rate (FRED: 'SOFR').
    start_date : datetime
        The start date for fetching the historical rate series.
    end_date : datetime
        The end date for fetching the historical rate series.
    opt_type : OptimizationType, optional
        Determines how the rate is sampled:
        - 'BACKTEST' or None: Computes the mean annualized rate over the [start_date, end_date] window.
        - 'LIVE': Uses the single most recent available rate observation.

    Returns
    -------
    float
        The annualized risk-free rate expressed as a decimal (e.g., 0.045 for 4.5%).

    Raises
    ------
    ValueError
        If an unknown `base` or `opt_type` literal is provided, or if FRED returns no data.
    """
    ticker: str
    match base:
        case "T_BILLS":
            ticker = "DGS3MO"
        case "TREASURY_NOTES":
            ticker = "DGS10"
        case "SOFR":
            ticker = "SOFR"
        case _:
            raise ValueError(f"Unknown literal for base {base}")

    # Download the FRED data from the open source
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
    risk_ticker_df = (
        pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
        .loc[start_date:end_date]
        .dropna()
    )
    if risk_ticker_df.empty:
        raise ValueError(
            f"No data returned from FRED for ticker '{ticker}' between {start_date.date()} and {end_date.date()}."
        )

    risk_free_rate = 0.0
    match opt_type:
        case "BACKTEST":
            risk_free_rate = float(risk_ticker_df.mean().iloc[0])
        case "LIVE":
            risk_free_rate = float(risk_ticker_df.iloc[-1, 0])  # type: ignore
        case _:
            raise ValueError(f"Unknown literal for opt_type {opt_type}")

    return risk_free_rate / 100


def optimize_portfolio(
    tickers_df: pd.DataFrame,
    rf_base: RiskFreeRateBase,
    cov_model: CovarianceModel = "CLASSIC",
    returns_model: ReturnsModel = "HISTORICAL",
    opt_type: OptimizationType = "BACKTEST",
    views: np.ndarray | None = None,
    views_transition: np.ndarray | None = None,
    bl_tau: float = 0.05,
    prediction_period: int = 1,
    l1_coeff: float = 0.0,
    w_market: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculates the efficient frontier by minimizing volatility across a spectrum of target returns.

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical price data for the assets, structured with multi-index columns per ticker.
    rf_base : RiskFreeRateBase
        The benchmark asset used for calculating the risk-free rate ('T_BILLS', 'TREASURY_NOTES', or 'SOFR').
    cov_model : CovarianceModel, optional
        Covariance estimation model to use ('CLASSIC', 'LEDOIT_WOLF', 'GARCH', or 'EGARCH'),
        by default 'CLASSIC'.
    returns_model : ReturnsModel, optional
        Expected returns estimation model to use ('HISTORICAL' or 'BLACK_LITTERMAN'),
        by default 'HISTORICAL'. When 'HISTORICAL', implied equilibrium returns (or GARCH returns)
        are used without Black-Litterman subjective views blending.
    opt_type : OptimizationType, optional
        Determines the date range for fetching the risk-free rate ('BACKTEST' or 'LIVE'),
        by default 'BACKTEST'.
    views : np.ndarray | None, optional
        Investor view vector Q for the Black-Litterman model, by default None.
    views_transition : np.ndarray | None, optional
        Pick/transition matrix P linking views to assets for Black-Litterman, by default None.
    bl_tau : float, optional
        A scalar representing the degree of uncertainty in the prior equilibrium vector in Black-Litterman,
        by default 0.05.
    prediction_period : int, optional
        Forecast horizon in trading days for (E)GARCH volatility modeling,
        by default 1.
    l1_coeff : float, optional
        L1 penalty coefficient applied to portfolio turnover (sum of absolute weight differences)
        to penalize excessive rebalancing, by default 0.0.
    w_market : np.ndarray | None, optional
        Market capitalization weights vector across assets used to compute CAPM implied
        equilibrium expected returns. If None, derived from `tickers_df` using `get_market_weights`
        at the latest date, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row represents a portfolio on the efficient frontier,
        containing columns for 'weights', 'return', 'vol', 'sharpe', and 'sortino'.

    Raises
    ------
    ValueError
        If an unrecognized `opt_type`, `cov_model`, or `returns_model` entry is provided.
    """
    # Calculate optimization params
    tickers_df = tickers_df.dropna()
    log_ret_df = log_returns(tickers_df)
    log_ret = np.array(log_ret_df)
    if w_market is None:
        w_market = get_market_weights(tickers_df).loc[tickers_df.index[-1]].values

    # Defining optimization type
    match opt_type:
        case "BACKTEST":
            rf_start_date = (
                pd.to_datetime(tickers_df.index[0]).tz_localize(None).to_pydatetime()
            )
            rf_end_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
        case "LIVE":
            rf_start_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
            rf_end_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
        case _:
            raise ValueError(f"Unrecognized opt_value param value: {opt_type}.")

    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date, opt_type)

    # Choose covariance estimation model
    match cov_model:
        case "CLASSIC":
            cov_matrix = np.array(
                log_ret_df.cov() * TRADING_DAYS_PER_YEAR
            )  # assets covariance matrix # type: ignore
            expected_returns = (
                RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
            )
        case "LEDOIT_WOLF":
            lw = LedoitWolf()
            lw.fit(log_ret)
            cov_matrix = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            expected_returns = (
                RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
            )
        case "GARCH" | "EGARCH":
            lw = LedoitWolf()
            lw.fit(log_ret)
            cov_matrix = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            (garch_ret, garch_vol) = garch(
                log_ret, prediction_period, arch_type=cov_model
            )
            expected_returns = garch_ret
            garch_vol_diag = np.diag(garch_vol)
            cov_matrix = garch_vol_diag @ cov_matrix @ garch_vol_diag

        case _:
            raise ValueError(f"Unrecognized cov_model param value: {cov_model}.")

    # Choose returns estimation model
    match returns_model:
        case "BLACK_LITTERMAN":
            (expected_returns, cov_matrix) = black_litterman(
                expected_returns,
                cov_matrix,
                risk_free_rate,
                views,
                views_transition,
                bl_tau,
            )
        case "HISTORICAL":
            pass  # keep the mean historical or GARCH returns
        case _:
            raise ValueError(
                f"Unrecognized returns_model param value: {returns_model}."
            )

    # Optimization
    optimum_results = _run_portfolio_optimization(
        cov_matrix,
        l1_coeff,
        expected_returns,
        log_ret,
        risk_free_rate,
    )

    return pd.DataFrame(optimum_results)


def _run_portfolio_optimization(
    cov_matrix: np.ndarray,
    l1_coeff: float,
    expected_returns: np.ndarray,
    log_returns: np.ndarray,
    risk_free_rate: float,
) -> list[dict[str, Any]]:
    """
    Internal helper that performs the SLSQP portfolio optimization loop across target returns.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Annualized covariance matrix of asset returns.
    l1_coeff : float
        L1 penalty coefficient applied to portfolio turnover (||w - w_last||_1).
    expected_returns : np.ndarray
        Expected annualized asset returns.
    log_returns : np.ndarray
        Daily log returns matrix of the assets.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    list[dict[str, Any]]
        List of optimization result dictionaries containing 'weights', 'return', 'vol',
        'sharpe', and 'sortino' for each convergence point.
    """
    num_assets = len(cov_matrix)
    init_weights = np.ones(num_assets) / num_assets
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]

    # Bounds: Long-only (0 <= w_i <= 1)
    bounds = tuple((0.05, 0.25) for _ in range(num_assets))

    no_target_opt = minimize(
        fun=lambda weights: 0.5 * (weights.T @ cov_matrix @ weights),
        x0=init_weights,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
    )

    # Constraints over the possible target_returns
    min_target = np.dot(no_target_opt.x, expected_returns)
    max_target = expected_returns.max()
    target_returns = np.linspace(min_target - 1e-5, max_target + 1e-5, 100)

    # Optimize with the target_return constraints
    optimum_results = []
    last_optimal_weights = init_weights
    for target_return in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, target_return=target_return: (
                    np.dot(w, expected_returns) - target_return
                ),
            },
        ]

        res = minimize(
            fun=lambda w, last_optimal_weights=last_optimal_weights: (
                _optimize_portfolio_objective(
                    w,
                    last_optimal_weights,
                    l1_coeff,
                    cov_matrix,
                )
            ),
            x0=last_optimal_weights,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )

        if res.success:
            last_optimal_weights = res.x
            ret = np.dot(last_optimal_weights, expected_returns)
            vol = np.sqrt(last_optimal_weights.T @ cov_matrix @ last_optimal_weights)
            sharpe = (ret - risk_free_rate) / vol

            daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
            square_negative_deviations = (
                np.minimum(0, last_optimal_weights @ log_returns.T - daily_rf) ** 2
            )
            downside_vol = np.sqrt(np.mean(square_negative_deviations)) * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
            sortino = (ret - risk_free_rate) / downside_vol

            optimum_results.append(
                {
                    "weights": last_optimal_weights,
                    "return": ret,
                    "vol": vol,
                    "sharpe": sharpe,
                    "sortino": sortino,
                }
            )

    return optimum_results


def _optimize_portfolio_objective(
    weights: np.ndarray,
    current_weights: np.ndarray,
    l1_coeff: float,
    cov_matrix: np.ndarray,
) -> float:
    """
    Objective function for portfolio variance minimization with an L1 turnover penalty.

    Parameters
    ----------
    weights : np.ndarray
        Candidate portfolio weights vector to evaluate.
    current_weights : np.ndarray
        Current / baseline portfolio weights vector used to calculate rebalancing turnover.
    l1_coeff : float
        L1 penalty multiplier applied to turnover (sum of absolute weight differences).
    cov_matrix : np.ndarray
        Annualized covariance matrix of asset returns.

    Returns
    -------
    float
        Penalized objective value: 0.5 * (w^T @ Cov @ w) - l1_coeff * turnover.
    """
    turnover = np.sum(np.abs(weights - current_weights))
    penalty = l1_coeff * turnover

    markowitz_fun = 0.5 * (weights.T @ cov_matrix @ weights)

    return markowitz_fun - penalty


def find_max_sharpe(
    tickers_df: pd.DataFrame,
    rf_base: RiskFreeRateBase,
    cov_model: CovarianceModel = "CLASSIC",
    returns_model: ReturnsModel = "HISTORICAL",
    opt_type: OptimizationType = "BACKTEST",
    views: np.ndarray | None = None,
    views_transition: np.ndarray | None = None,
    bl_tau: float = 0.05,
    prediction_period: int = 1,
    init_weights: np.ndarray | None = None,
    l1_coeff: float = 0.0,
    w_market: np.ndarray | None = None,
) -> tuple[SharpeRatio, pd.DataFrame]:
    """
    Optimizes portfolio weights to maximize the Sharpe ratio (find the tangency portfolio).

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical price data for the assets, structured with multi-index columns per ticker.
    rf_base : RiskFreeRateBase
        The benchmark asset used for calculating the risk-free rate ('T_BILLS', 'TREASURY_NOTES', or 'SOFR').
    cov_model : CovarianceModel, optional
        Covariance estimation model to use ('CLASSIC', 'LEDOIT_WOLF', 'GARCH', or 'EGARCH'),
        by default 'CLASSIC'.
    returns_model : ReturnsModel, optional
        Expected returns estimation model to use ('HISTORICAL' or 'BLACK_LITTERMAN'),
        by default 'HISTORICAL'. When 'HISTORICAL', implied equilibrium returns (or GARCH returns)
        are used without Black-Litterman subjective views blending.
    opt_type : OptimizationType, optional
        Determines the date range for fetching the risk-free rate ('BACKTEST' or 'LIVE'), by default BACKTEST.
    views : np.ndarray | None, optional
        Investor view vector Q for the Black-Litterman model, by default None.
    views_transition : np.ndarray | None, optional
        Pick/transition matrix P linking views to assets for Black-Litterman, by default None.
    bl_tau : float, optional
        A scalar representing the degree of uncertainty in the prior equilibrium vector in Black-Litterman,
        by default 0.05.
    prediction_period : int, optional
        Forecast horizon in trading days for (E)GARCH volatility modeling,
        by default 1.
    init_weights : np.ndarray | None, optional
        Initial / current portfolio weights vector used as optimizer starting point and for
        evaluating rebalancing turnover, by default None (initialized to equal weights).
        If provided, weights are rescaled from percentages if needed, clipped to [0.05, 0.25],
        and normalized to sum to 1.
    l1_coeff : float, optional
        L1 penalty coefficient applied to portfolio turnover (||w - w_init||_1)
        to penalize excessive rebalancing, by default 0.0.
    w_market : np.ndarray | None, optional
        Market capitalization weights vector across assets used to compute CAPM implied
        equilibrium expected returns. If None, derived from `tickers_df` using `get_market_weights`
        at the latest date, by default None.

    Returns
    -------
    tuple[SharpeRatio, pd.DataFrame]
        A tuple containing:
        - SharpeRatio object with tangency portfolio metrics (max_sharpe, tangency_return, tangency_vol).
        - DataFrame containing the percentage weights for each stock in the optimal portfolio.

    Raises
    ------
    ValueError
        If an unrecognized `opt_type`, `cov_model`, or `returns_model` entry is provided.

    Warns
    -----
    UserWarning
        If the scipy SLSQP optimizer fails to converge, falls back to `init_weights`.
    """
    # Calculate optimization params
    tickers_df = tickers_df.dropna()
    num_assets = tickers_df.columns.get_level_values(0).nunique()
    log_ret_df = log_returns(tickers_df)
    log_ret = np.array(log_ret_df)
    if w_market is None:
        w_market = get_market_weights(tickers_df).loc[tickers_df.index[-1]].values

    if init_weights is None:
        init_weights = np.ones(num_assets) / num_assets
    else:
        init_weights = np.array([w / 100 if w > 1 else w for w in init_weights])
        init_weights = np.clip(init_weights, 0.05, 0.25)
        init_weights = init_weights / sum(init_weights)

    match opt_type:
        case "BACKTEST":
            rf_start_date = (
                pd.to_datetime(tickers_df.index[0]).tz_localize(None).to_pydatetime()
            )
            rf_end_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
        case "LIVE":
            rf_start_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
            rf_end_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
        case _:
            raise ValueError(f"Unrecognized opt_value param value: {opt_type}.")

    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date, opt_type)

    # Choose covariance estimation model
    match cov_model:
        case "CLASSIC":
            cov_matrix = np.array(
                log_ret_df.cov() * TRADING_DAYS_PER_YEAR
            )  # assets covariance matrix # type: ignore
            expected_returns = (
                RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
            )
        case "LEDOIT_WOLF":
            lw = LedoitWolf()
            lw.fit(log_ret)
            cov_matrix = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            expected_returns = (
                RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
            )
        case "GARCH" | "EGARCH":
            lw = LedoitWolf()
            lw.fit(log_ret)
            cov_matrix = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            sample_vols = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(sample_vols, sample_vols)

            (garch_ret, garch_vol) = garch(
                log_ret, prediction_period, arch_type=cov_model
            )

            expected_returns = garch_ret
            garch_vol_diag = np.diag(garch_vol)
            cov_matrix = garch_vol_diag @ corr_matrix @ garch_vol_diag
        case _:
            raise ValueError(f"Unrecognized cov_model param value: {cov_model}.")

    # Choose returns estimation model
    match returns_model:
        case "BLACK_LITTERMAN":
            (expected_returns, cov_matrix) = black_litterman(
                expected_returns,
                cov_matrix,
                risk_free_rate,
                views,
                views_transition,
                bl_tau,
            )
        case "HISTORICAL":
            pass  # keep the mean historical or GARCH returns
        case _:
            raise ValueError(
                f"Unrecognized returns_model param value: {returns_model}."
            )

    # Find the Sharpe Ratio optimum
    return _maximize_sharpe_ratio(
        tickers_df,
        init_weights,
        l1_coeff,
        cov_matrix,
        expected_returns,
        risk_free_rate,
    )


def _maximize_sharpe_ratio(
    tickers_df: pd.DataFrame,
    init_weights: np.ndarray,
    l1_coeff: float,
    cov_matrix: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float,
) -> tuple[SharpeRatio, pd.DataFrame]:
    """
    Internal helper that performs the SLSQP Sharpe ratio optimization.

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical asset data used to extract ticker names.
    init_weights : np.ndarray
        Initial weight allocation array used as optimizer starting point and turnover baseline.
    l1_coeff : float
        L1 penalty coefficient applied to turnover (||w - init_weights||_1).
    cov_matrix : np.ndarray
        Annualized covariance matrix of asset returns.
    expected_returns : np.ndarray
        Expected annualized returns of the assets.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    tuple[SharpeRatio, pd.DataFrame]
        Optimal SharpeRatio metrics and DataFrame of percentage weights per asset.

    Warns
    -----
    UserWarning
        If the SLSQP optimizer fails to converge, falls back to `init_weights`.
    """
    num_assets = len(cov_matrix)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.05, 0.25) for _ in range(num_assets))

    res_max_sharpe = minimize(
        fun=lambda w: _max_sharpe_objective(
            w,
            init_weights,
            l1_coeff,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate,
            cov_matrix=cov_matrix,
        ),
        x0=init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if res_max_sharpe.success:
        optimal_weights = res_max_sharpe.x
    else:
        optimal_weights = init_weights

        warnings.warn("The optimizator finished work with an error or was aborted.")

    exact_max_ret = np.dot(optimal_weights, expected_returns)
    exact_max_vol = float(np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights))
    exact_max_sharpe = (exact_max_ret - risk_free_rate) / exact_max_vol

    max_sharpe_stocks_weights = pd.DataFrame()
    tickers = tickers_df.columns.get_level_values(0).unique()
    for i, (name) in enumerate(tickers):
        max_sharpe_stocks_weights[name] = [round(optimal_weights[i], 2) * 100]

    max_sharpe = SharpeRatio(
        max_sharpe=exact_max_sharpe,
        tangency_return=exact_max_ret,
        tangency_vol=exact_max_vol,
    )

    return (max_sharpe, max_sharpe_stocks_weights)


def _max_sharpe_objective(
    weights: np.ndarray,
    current_weights: np.ndarray,
    l1_coeff: float,
    expected_returns: np.ndarray,
    risk_free_rate: float,
    cov_matrix: np.ndarray,
) -> float:
    """
    Objective function to find the maximum Sharpe ratio (returns negative Sharpe ratio for minimization).

    Parameters
    ----------
    weights : np.ndarray
        Array of candidate portfolio weights to optimize.
    current_weights : np.ndarray
        Current / baseline portfolio weights vector used to calculate rebalancing turnover.
    l1_coeff : float
        L1 penalty coefficient applied to turnover (||w - current_weights||_1).
    expected_returns : np.ndarray
        Array of expected annualized returns for the assets.
    risk_free_rate : float
        The annualized risk-free rate.
    cov_matrix : np.ndarray
        The annualized covariance matrix of the assets.

    Returns
    -------
    float
        Penalized negative Sharpe ratio: -((Return - RiskFreeRate) / Volatility - penalty).
    """
    ret = np.dot(weights, expected_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)

    turnover = np.sum(np.abs(weights - current_weights))
    penalty = l1_coeff * turnover

    # negative Sharpe ratio so minimize() finds the maximum
    return -((ret - risk_free_rate) / vol - penalty)


def find_max_sortino(
    tickers_df: pd.DataFrame,
    rf_base: RiskFreeRateBase,
    cov_model: CovarianceModel = "CLASSIC",
    returns_model: ReturnsModel = "HISTORICAL",
    opt_type: OptimizationType = "BACKTEST",
    views: np.ndarray | None = None,
    views_transition: np.ndarray | None = None,
    bl_tau: float = 0.05,
    prediction_period: int = 1,
    init_weights: np.ndarray | None = None,
    l1_coeff: float = 0.0,
    w_market: np.ndarray | None = None,
) -> tuple[SortinoRatio, pd.DataFrame]:
    """
    Optimizes portfolio weights to maximize the Sortino ratio (minimizing downside volatility).

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical price data for the assets, structured with multi-index columns per ticker.
    rf_base : RiskFreeRateBase
        The benchmark asset used for calculating the risk-free rate ('T_BILLS', 'TREASURY_NOTES', or 'SOFR').
    cov_model : CovarianceModel, optional
        Covariance estimation model to use ('CLASSIC', 'LEDOIT_WOLF', 'GARCH', or 'EGARCH'),
        by default 'CLASSIC'.
    returns_model : ReturnsModel, optional
        Expected returns estimation model to use ('HISTORICAL' or 'BLACK_LITTERMAN'),
        by default 'HISTORICAL'. When 'HISTORICAL', implied equilibrium returns (or GARCH returns)
        are used without Black-Litterman subjective views blending.
    opt_type : OptimizationType, optional
        Determines the date range for fetching the risk-free rate ('BACKTEST' or 'LIVE'),
        by default 'BACKTEST'.
    views : np.ndarray | None, optional
        Investor view vector Q for the Black-Litterman model, by default None.
    views_transition : np.ndarray | None, optional
        Pick/transition matrix P linking views to assets for Black-Litterman, by default None.
    bl_tau : float, optional
        A scalar representing the degree of uncertainty in the prior equilibrium vector in Black-Litterman,
        by default 0.05.
    prediction_period : int, optional
        Forecast horizon in trading days for (E)GARCH volatility modeling,
        by default 1.
    init_weights : np.ndarray | None, optional
        Initial / current portfolio weights vector used as optimizer starting point and for
        evaluating rebalancing turnover, by default None (initialized to equal weights).
        If provided, weights are rescaled from percentages if needed, clipped to [0.05, 0.25],
        and normalized to sum to 1.
    l1_coeff : float, optional
        L1 penalty coefficient applied to portfolio turnover (||w - w_init||_1)
        to penalize excessive rebalancing, by default 0.0.
    w_market : np.ndarray | None, optional
        Market capitalization weights vector across assets used to compute CAPM implied
        equilibrium expected returns. If None, derived from `tickers_df` using `get_market_weights`
        at the latest date, by default None.

    Returns
    -------
    tuple[SortinoRatio, pd.DataFrame]
        A tuple containing:
        - SortinoRatio object with metrics (max_sortino, tangency_return, tangency_vol).
        - DataFrame containing the percentage weights for each stock in the optimal portfolio.

    Raises
    ------
    ValueError
        If an unrecognized `opt_type`, `cov_model`, or `returns_model` entry is provided.

    Warns
    -----
    UserWarning
        If the scipy SLSQP optimizer fails to converge, falls back to `init_weights`.
    """
    # Calculate optimization params
    tickers_df = tickers_df.dropna()
    num_assets = tickers_df.columns.get_level_values(0).nunique()
    log_ret_df = log_returns(tickers_df)
    log_ret = np.array(log_ret_df)
    if w_market is None:
        w_market = get_market_weights(tickers_df).loc[tickers_df.index[-1]].values

    if init_weights is None:
        init_weights = np.ones(num_assets) / num_assets
    else:
        init_weights = np.array([w / 100 if w > 1 else w for w in init_weights])
        init_weights = np.clip(init_weights, 0.05, 0.25)
        init_weights = init_weights / sum(init_weights)

    match opt_type:
        case "BACKTEST":
            rf_start_date = (
                pd.to_datetime(tickers_df.index[0]).tz_localize(None).to_pydatetime()
            )
            rf_end_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
        case "LIVE":
            rf_start_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
            rf_end_date = (
                pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            )
        case _:
            raise ValueError(f"Unrecognized opt_value param value: {opt_type}.")

    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date, opt_type)

    # Choose covariance estimation model
    match cov_model:
        case "CLASSIC":
            cov_matrix = np.array(
                log_ret_df.cov() * TRADING_DAYS_PER_YEAR
            )  # assets covariance matrix # type: ignore
            expected_returns = (
                RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
            )
        case "LEDOIT_WOLF":
            lw = LedoitWolf()
            lw.fit(log_ret)
            cov_matrix = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            expected_returns = (
                RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
            )
        case "GARCH" | "EGARCH":
            lw = LedoitWolf()
            lw.fit(log_ret)
            cov_matrix = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            (garch_ret, garch_vol) = garch(
                log_ret, prediction_period, arch_type=cov_model
            )
            expected_returns = garch_ret
            garch_vol_diag = np.diag(garch_vol)
            cov_matrix = garch_vol_diag @ cov_matrix @ garch_vol_diag
        case _:
            raise ValueError(f"Unrecognized cov_model param value: {cov_model}.")

    # Choose returns estimation model
    match returns_model:
        case "BLACK_LITTERMAN":
            (expected_returns, _) = black_litterman(
                expected_returns,
                cov_matrix,
                risk_free_rate,
                views,
                views_transition,
                bl_tau,
            )
        case "HISTORICAL":
            pass  # keep the mean historical or GARCH returns
        case _:
            raise ValueError(
                f"Unrecognized returns_model param value: {returns_model}."
            )

    # Find the Sortino Ratio optimum
    return _maximize_sortino_ratio(
        tickers_df,
        init_weights,
        l1_coeff,
        log_ret,
        expected_returns,
        risk_free_rate,
    )


def _maximize_sortino_ratio(
    tickers_df: pd.DataFrame,
    init_weights: np.ndarray,
    l1_coeff: float,
    log_returns: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float,
) -> tuple[SortinoRatio, pd.DataFrame]:
    """
    Internal helper that performs the SLSQP Sortino ratio optimization.

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical asset data used to extract ticker names.
    init_weights : np.ndarray
        Initial weight allocation array used as optimizer starting point and turnover baseline.
    l1_coeff : float
        L1 penalty coefficient applied to turnover (||w - init_weights||_1).
    log_returns : np.ndarray
        Matrix of daily logarithmic asset returns.
    expected_returns : np.ndarray
        Expected annualized returns of the assets.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    tuple[SortinoRatio, pd.DataFrame]
        Optimal SortinoRatio metrics and DataFrame of percentage weights per asset.

    Warns
    -----
    UserWarning
        If the SLSQP optimizer fails to converge, falls back to `init_weights`.
    """
    num_assets = tickers_df.columns.get_level_values(0).nunique()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.05, 0.25) for _ in range(num_assets))

    res_max_sortino = minimize(
        fun=lambda w: _max_sortino_objective(
            w,
            init_weights,
            l1_coeff,
            log_returns,
            expected_returns,
            risk_free_rate,
        ),
        x0=init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    square_negative_deviations = np.minimum(0, log_returns - daily_rf) ** 2

    if res_max_sortino.success:
        optimal_weights = res_max_sortino.x
    else:
        optimal_weights = init_weights

        warnings.warn("The optimizator finished work with an error or was aborted.")

    exact_max_ret = np.dot(optimal_weights, expected_returns)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    square_negative_deviations = (
        np.minimum(0, optimal_weights @ log_returns.T - daily_rf) ** 2
    )
    exact_max_negative_vol = np.sqrt(
        (square_negative_deviations).mean(axis=0)
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)

    exact_max_sortino = (exact_max_ret - risk_free_rate) / exact_max_negative_vol

    max_sortino_stocks_weights = pd.DataFrame()
    tickers = tickers_df.columns.get_level_values(0).unique()
    for i, (name) in enumerate(tickers):
        max_sortino_stocks_weights[name] = [round(optimal_weights[i], 2) * 100]

    max_sortino = SortinoRatio(
        max_sortino=exact_max_sortino,
        tangency_return=exact_max_ret,
        tangency_vol=exact_max_negative_vol,
    )

    return (max_sortino, max_sortino_stocks_weights)


def _max_sortino_objective(
    weights: np.ndarray,
    current_weights: np.ndarray,
    l1_coeff: float,
    log_returns: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Objective function to find the maximum Sortino ratio (returns negative Sortino ratio for minimization).

    Parameters
    ----------
    weights : np.ndarray
        Array of candidate portfolio weights to optimize.
    current_weights : np.ndarray
        Current / baseline portfolio weights vector used to calculate rebalancing turnover.
    l1_coeff : float
        L1 penalty coefficient applied to turnover (||w - current_weights||_1).
    log_returns : np.ndarray
        Matrix of daily logarithmic asset returns.
    expected_returns : np.ndarray
        Array of expected annualized returns for the assets.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    float
        Penalized negative Sortino ratio: -((Return - RiskFreeRate) / DownsideVolatility - penalty).
    """
    ret = np.dot(weights, expected_returns)

    turnover = np.sum(np.abs(weights - current_weights))
    penalty = l1_coeff * turnover

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    square_negative_deviations = np.minimum(0, weights @ log_returns.T - daily_rf) ** 2
    downside_vol = np.sqrt((square_negative_deviations).mean(axis=0)) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    # negative Sortino ratio so minimize() finds the maximum
    return -((ret - risk_free_rate) / downside_vol - penalty)
