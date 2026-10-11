"""Portfolio backtesting engine with rolling out-of-sample evaluation.

Provides backtesting capabilities for portfolio optimization strategies
(e.g., GARCH Sharpe, GARCH Sortino, EGARCH Sortino) against an equal-weighted benchmark portfolio
and the S&P 500 (^GSPC), factoring in transaction costs, rebalancing cycles,
and daily asset weight drift.
"""

from dataclasses import astuple, dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from data import (
    HistoricalNewsProvider,
    calcualte_structural_cost_coupling_value,
    get_market_weights,
    get_portfolio_exp_vol,
    log_returns,
)
from src.agents import FinancialViewsAgent, translate_views
from src.config.settings import Settings
from src.portfolio import (
    ReturnsModel,
    SharpeRatio,
    SortinoRatio,
    find_max_sharpe,
    find_max_sortino,
)

L1PenaltyType = Literal["FIXED", "STRUCTURAL_COST"]
"""
Type of L1 turnover regularization penalty applied during portfolio optimization.

Options:
- 'FIXED': Fixed penalty inversely proportional to portfolio expected annual volatility.
- 'STRUCTURAL_COST': Dynamic structural penalty calibrated against transaction costs, turnover, portfolio value, and rebalance frequency.
"""

OptimizationMetric = Literal["GARCH_SHARPE", "GARCH_SORTINO", "EGARCH_SORTINO"]
"""
Objective metric and volatility model combination for portfolio optimization.

Options:
- 'GARCH_SHARPE': Maximizes Sharpe ratio using GARCH(t,t) conditional volatility and covariance.
- 'GARCH_SORTINO': Maximizes Sortino ratio using GARCH(t,t) conditional volatility and downside variance.
- 'EGARCH_SORTINO': Maximizes Sortino ratio using EGARCH(1,1) asymmetric conditional volatility and downside variance.
"""


@dataclass
class Turnover:
    """Represents the post-rebalance portfolio state and turnover costs.

    Attributes
    ----------
    turnover : float
        Total portfolio turnover fraction (sum of absolute weight changes).
    portfolio_value : float
        Updated portfolio capital after deducting turnover transaction costs.
    weights_drift : np.ndarray
        New target asset weights applied upon rebalance.
    turnover_cost : float
        Total transaction cost incurred for the rebalancing turnover.
    """

    turnover: float
    portfolio_value: float
    weights_drift: np.ndarray
    turnover_cost: float


@dataclass
class DailyAsset:
    """Represents daily portfolio return and drifted weights for an asset basket.

    Attributes
    ----------
    effective_return : float
        Net daily portfolio return, accounting for transaction fees on rebalance day.
    portfolio_value : float
        Updated portfolio value after applying daily asset returns.
    weights_drift : np.ndarray
        Updated asset weights after price drift over the trading day.
    """

    effective_return: float
    portfolio_value: float
    weights_drift: np.ndarray


class BacktestEngine:
    def __init__(self, settings: Settings | None = None):
        """Initialize the backtesting engine with configuration settings.

        Parameters
        ----------
        settings : Settings | None, optional
            Application settings containing portfolio and agent configuration.
            If None, loaded from TOML via Settings.from_toml().
        """
        if settings is None:
            settings = Settings.from_toml()

        self.settings = settings
        self.news_provider = HistoricalNewsProvider(settings=settings)

        self.llm = settings.agent.llm_name
        self.agent_base_url = settings.agent.agent_url
        self.trading_days_per_day = settings.portfolio.trading_days_per_year
        self.trading_days_per_year = self.trading_days_per_day
        self.risk_unacceptance_value = settings.portfolio.risk_unacceptance_value

    def _calculate_turnover(
        self,
        opt_weights: np.ndarray,
        w_drift: np.ndarray | None,
        broker_commission: float,
        portfolio_value: float,
        turnover: float = 1.0,
    ) -> Turnover:
        """Calculate portfolio rebalancing turnover, commission costs, and new capital.

        Parameters
        ----------
        opt_weights : np.ndarray
            New target asset weights from optimization.
        w_drift : np.ndarray | None
            Current drifted asset weights before rebalancing, or None for initial period.
        broker_commission : float
            Broker fee rate per unit of turnover (e.g., 0.0005 for 0.05%).
        portfolio_value : float
            Current portfolio equity before transaction cost deduction.
        turnover : float, default=1.0
            Fallback turnover value when w_drift is None (initial allocation).

        Returns
        -------
        Turnover
            Dataclass containing turnover, updated portfolio_value, weights_drift, and turnover_cost.
        """
        turnover = (
            np.abs(opt_weights - w_drift).sum() if w_drift is not None else turnover
        )
        t_cost = turnover * broker_commission
        portfolio_value *= 1 - t_cost
        w_drift = opt_weights

        return Turnover(
            turnover=turnover,
            portfolio_value=portfolio_value,
            weights_drift=w_drift,
            turnover_cost=t_cost,
        )

    def _calculate_daily_asset_returns(
        self,
        daily_asset_returns: np.ndarray,
        weights_drift: np.ndarray,
        turnover_cost: float,
        out_of_sample_start: bool,
        portfolio_value: float,
    ) -> DailyAsset:
        """Calculate daily portfolio return, apply transaction costs if at start, and drift weights.

        Parameters
        ----------
        daily_asset_returns : np.ndarray
            1D array of daily returns across all assets for the current day.
        weights_drift : np.ndarray
            1D array of asset weights entering the trading day.
        turnover_cost : float
            Transaction fee deduction to apply if this is the start of the rebalance period.
        out_of_sample_start : bool
            Whether current day is the first day of the out-of-sample rebalancing period.
        portfolio_value : float
            Current portfolio value before today's market movement.

        Returns
        -------
        DailyAsset
            Dataclass containing effective_return, updated portfolio_value, and drifted weights.
        """
        portfolio_returns = np.dot(weights_drift, daily_asset_returns)
        if out_of_sample_start:
            effective_port_ret = (1 - turnover_cost) * (1 + portfolio_returns) - 1
        else:
            effective_port_ret = portfolio_returns

        portfolio_value *= 1 + portfolio_returns
        weights_drift = weights_drift * (
            (1 + daily_asset_returns) / (1 + portfolio_returns)
        )
        weights_drift = weights_drift / np.sum(weights_drift)

        return DailyAsset(
            effective_return=effective_port_ret,
            portfolio_value=portfolio_value,
            weights_drift=weights_drift,
        )

    def _agent_view_analysis(
        self, tickers: list[str], as_of_date: str, log_ret_df: pd.DataFrame
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.llm is None:
            return (None, None)

        news_ns = self.news_provider.get_news_for_rebalance(as_of_date, tickers)

        agent = FinancialViewsAgent(
            model=self.llm,
            base_url=self.agent_base_url,
        )
        assets_index = pd.Index(tickers)
        report = agent.analyze_market_and_generate_views(
            as_of_date=as_of_date,  # type: ignore
            asset_universe=assets_index,
            news_payload=news_ns,
        )

        P = Q = None
        if report is not None:
            cov_matrix = np.array(log_ret_df.cov() * self.trading_days_per_day)

            translated = translate_views(report, assets_index, cov_matrix)
            P = translated.P if translated is not None else None
            Q = translated.Q if translated is not None else None

        return (P, Q)

    def perform_backtesting(
        self,
        tickers_df: pd.DataFrame,
        init_portfolio_value: float,
        optimization_metric: OptimizationMetric,
        returns_model: ReturnsModel,
        sp500_history: pd.DataFrame | None = None,
        lookback_window: int = 504,
        rebalancing_period: int = 21,  # 21-trading day
        broker_commission: float = 0.0005,  # 0.05%
        l1_penalty_type: L1PenaltyType = "FIXED",
    ) -> pd.DataFrame:
        """Perform rolling-window walk-forward backtesting for a portfolio optimization strategy.

        Evaluates the optimized portfolio out-of-sample against an equal-weighted
        benchmark portfolio and the S&P 500 (^GSPC). Accounts for transaction fees,
        regular rebalancing cycles, and daily asset weight drift.

        Parameters
        ----------
        tickers_df : pd.DataFrame
            Historical price data for tickers, structured as a pandas MultiIndex DataFrame
            with ('Ticker', 'Price') columns (containing Open, High, Low, Close, Volume).
        init_portfolio_value : float
            Initial investment capital (e.g., 10,000.0).
        optimization_metric : OptimizationMetric
            Objective metric for optimization. Options:
            - "GARCH_SHARPE": Maximize Sharpe ratio with GARCH(t,t) covariance.
            - "GARCH_SORTINO": Maximize Sortino ratio with GARCH(t,t) downside variance.
            - "EGARCH_SORTINO": Maximize Sortino ratio with EGARCH(1,1) asymmetric downside variance.
        returns_model : ReturnsModel
            Expected returns estimation model. Options:
            - "HISTORICAL": Mean historical returns scaled by trading days.
            - "BLACK_LITTERMAN": Black-Litterman model incorporating market capitalization and investor views.
        sp500_history : pd.DataFrame | None, default=None
            Historical price data for the S&P 500 (^GSPC) benchmark, structured as a pandas
            MultiIndex DataFrame with ('Ticker', 'Price') columns containing 'Close'.
        lookback_window : int, default=504
            Number of historical trading days in the in-sample training window (~2 years).
        rebalancing_period : int, default=21
            Frequency of portfolio rebalancing in trading days (~1 month).
        broker_commission : float, default=0.0005
            Broker fee applied to portfolio turnover on rebalance (0.0005 = 0.05%).
        l1_penalty_type : L1PenaltyType, default="FIXED"
            Type of L1 turnover regularization penalty. Options:
            - "FIXED": Fixed penalty inversely proportional to portfolio expected volatility.
            - "STRUCTURAL_COST": Dynamic penalty calibrated against annual turnover cost and portfolio capital.

        Returns
        -------
        pd.DataFrame
            Backtesting results indexed by date, containing:
            - 'sharpe' or 'sortino': In-sample optimized metric value at each rebalance.
            - 'weights': Target asset allocation weights determined by optimization.
            - 'daily_return': Daily return of the optimized portfolio.
            - 'equal_portfolio_daily_return': Daily return of the equal-weighted portfolio.
            - 'portfolio_value': Cumulative equity curve of the optimized portfolio.
            - 'equal_portfolio_value': Cumulative equity curve of the equal-weighted portfolio.
            - 'sp500_value': Cumulative equity curve of the S&P 500 benchmark (present only if sp500_history is provided).

        Raises
        ------
        ValueError
            If unrecognized l1_penalty_type or optimization_metric is provided.
        """
        backtest_res = []
        metric: SharpeRatio | SortinoRatio | None = None
        rebalances_per_year = int(self.trading_days_per_day / rebalancing_period)

        # Trader's portfolio
        weights_drift = None
        close = tickers_df.xs("Close", axis=1, level="Price")
        history_days = tickers_df.index.shape[0]
        num_assets = tickers_df.columns.get_level_values(0).nunique()
        log_ret = log_returns(tickers_df)
        portfolio_value = init_portfolio_value
        all_history_weights = get_market_weights(tickers_df)

        # S&P 500
        sp500_close = (
            sp500_history.xs("Close", axis=1, level="Price")
            if sp500_history is not None
            else None
        )

        # Equal weights
        equal_w_drift = None
        equal_portfolio_value = init_portfolio_value

        start_date = datetime.strftime(tickers_df.index[0], "%Y-%m-%d")
        end_date = datetime.strftime(tickers_df.index[-1], "%Y-%m-%d")
        tickers = tickers_df.columns.get_level_values(0).unique().to_list()
        if returns_model == "BLACK_LITTERMAN":
            self.news_provider.fetch_historical_news_fmp(tickers, start_date, end_date)

        # Main loop
        for day in range(lookback_window, history_days, rebalancing_period):
            # In-Sample
            training_sample = tickers_df.iloc[day - lookback_window : day]
            turnover = 1.0

            w_market = np.array(all_history_weights.iloc[day].values)
            log_ret_slice = log_ret.iloc[day - lookback_window : day]
            match l1_penalty_type:
                case "STRUCTURAL_COST":
                    penalty = calcualte_structural_cost_coupling_value(
                        training_sample,
                        log_ret_slice,
                        turnover,
                        portfolio_value,
                        broker_commission,
                        rebalances_per_year,
                        weights_drift,
                    )
                case "FIXED":
                    expected_annyal_vol = get_portfolio_exp_vol(
                        log_ret_slice, weights_drift
                    )
                    penalty = (self.risk_unacceptance_value * 0.2) / expected_annyal_vol
                case _:
                    raise ValueError(
                        f"Unrecognized l1_penalty_type param value: {l1_penalty_type}."
                    )

            # Agent views analysis for Black-Litterman model
            P = Q = None
            if returns_model == "BLACK_LITTERMAN" and self.llm is not None:
                try:
                    (P, Q) = self._agent_view_analysis(
                        tickers, tickers_df.index[day], log_ret_slice
                    )
                except ValueError as e:
                    print(f"An error occurred: {e}")

            match optimization_metric:
                case "GARCH_SHARPE":
                    (metric, opt_weights) = find_max_sharpe(
                        training_sample,
                        "T_BILLS",
                        "GARCH",
                        returns_model,
                        "BACKTEST",
                        prediction_period=rebalancing_period,
                        init_weights=weights_drift,
                        l1_coeff=penalty,
                        w_market=w_market,
                        views=Q,
                        views_transition=P,
                    )
                case "GARCH_SORTINO":
                    (metric, opt_weights) = find_max_sortino(
                        training_sample,
                        "T_BILLS",
                        "GARCH",
                        returns_model,
                        "BACKTEST",
                        prediction_period=rebalancing_period,
                        init_weights=weights_drift,
                        l1_coeff=penalty,
                        w_market=w_market,
                        views=Q,
                        views_transition=P,
                    )
                case "EGARCH_SORTINO":
                    (metric, opt_weights) = find_max_sortino(
                        training_sample,
                        "T_BILLS",
                        "EGARCH",
                        returns_model,
                        "BACKTEST",
                        prediction_period=1,
                        init_weights=weights_drift,
                        l1_coeff=penalty,
                        w_market=w_market,
                        views=Q,
                        views_transition=P,
                    )
                case _:
                    raise ValueError(
                        f"Unrecognized optimization_metric param value: {optimization_metric}."
                    )

            # Out-of-Sample evaluation
            out_of_sample_start = day  # out-of-sample starting point
            out_of_sample_end = min(
                day + rebalancing_period, history_days
            )  # out-of-sample ending point

            # Ensure weights are a flat 1D array
            opt_weights = np.squeeze(opt_weights).astype(float) / 100

            # Calculate trader's portfolio commission
            opt_weights = opt_weights / np.sum(opt_weights)  # Normalize weights vector
            portfolio_turnover = self._calculate_turnover(
                opt_weights, weights_drift, broker_commission, portfolio_value, turnover
            )
            (turnover, portfolio_value, weights_drift, turnover_cost) = astuple(
                portfolio_turnover
            )

            # Calculate equal portfolio commission
            eq_weights = np.ones(num_assets) / num_assets
            equals_turnover = self._calculate_turnover(
                eq_weights,
                equal_w_drift,
                broker_commission,
                equal_portfolio_value,
            )
            (_, equal_portfolio_value, equal_w_drift, eq_turnover_cost) = astuple(
                equals_turnover
            )

            # Out-of-sample loop
            for p in range(out_of_sample_start, out_of_sample_end):
                # Trader's portfolio
                daily_asset_returns = np.array(
                    (close.iloc[p] - close.iloc[p - 1]) / close.iloc[p - 1]
                )

                port_daily_asset = self._calculate_daily_asset_returns(
                    daily_asset_returns,
                    weights_drift,
                    turnover_cost,
                    p == out_of_sample_start,
                    portfolio_value,
                )
                (effective_port_ret, portfolio_value, weights_drift) = astuple(
                    port_daily_asset
                )

                # Equal weights portfolio
                equal_daily_asset_returns = (
                    close.iloc[p].values - close.iloc[p - 1].values
                ) / close.iloc[p - 1].values  # type: ignore

                eq_port_daily_asset = self._calculate_daily_asset_returns(
                    equal_daily_asset_returns,
                    equal_w_drift,
                    eq_turnover_cost,
                    p == out_of_sample_start,
                    equal_portfolio_value,
                )
                (equal_port_ret, equal_portfolio_value, equal_w_drift) = astuple(
                    eq_port_daily_asset
                )

                # Period results
                backtest_res.append(
                    {
                        "date": close.index[p],
                        "sharpe": metric.max_sharpe
                        if isinstance(metric, SharpeRatio)
                        else None,
                        "sortino": metric.max_sortino
                        if isinstance(metric, SortinoRatio)
                        else None,
                        "weights": opt_weights.copy(),
                        "daily_return": effective_port_ret,
                        "equal_portfolio_daily_return": equal_port_ret,
                        "portfolio_value": portfolio_value,
                        "equal_portfolio_value": equal_portfolio_value,
                    }
                )

        backtest_res_df = pd.DataFrame(backtest_res).set_index("date")

        if sp500_close is not None:
            sp500_oos_prices = sp500_close.reindex(backtest_res_df.index).ffill()
            sp500_equity = init_portfolio_value * (
                sp500_oos_prices / sp500_oos_prices.iloc[0]
            )

            backtest_res_df["sp500_value"] = sp500_equity

        return backtest_res_df
