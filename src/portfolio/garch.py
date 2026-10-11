from typing import Literal

import numpy as np
from arch import arch_model

from src.config import Settings

ArchType = Literal["GARCH", "EGARCH"]


def garch(
    log_returns: np.ndarray,
    t: int,
    arch_type: ArchType = "GARCH",
    rescale: bool = True,
    settings: Settings | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits univariate (E)GARCH(1,1) models to each asset and forecasts
    conditional expected returns and conditional volatility.

    Parameters
    ----------
    log_returns : np.ndarray
        Array of historical log returns of shape (T, N) or (T,) in decimal form.
    t : int
        Forecast horizon in days.
    arch_type : ArchType, optional
        Volatility model specification ('GARCH' or 'EGARCH'),
        by default 'GARCH'.
    rescale : bool, optional
        Whether to multiply returns by 100 internally for optimization stability
        and scale back to decimals on output,
        by default True.
    settings : Settings | None, optional
        Application settings instance. If None, loaded from TOML.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - next_period_returns : np.ndarray of shape (N,)
            Conditional expected annualized return for each asset (in decimals).
        - next_period_vol : np.ndarray of shape (N,)
            Conditional expected annualized standard deviation (volatility) for each asset (in decimals).
    """

    if settings is None:
        settings = Settings.from_toml()

    trading_days_per_year = settings.portfolio.trading_days_per_year

    scale = 100.0 if rescale else 1.0
    num_assets = log_returns.shape[1] if log_returns.ndim == 2 else 1
    scaled_log_returns = (log_returns * scale).reshape(-1, num_assets)
    next_period_returns: np.ndarray = np.zeros(num_assets)
    next_period_vol: np.ndarray = np.zeros(num_assets)
    dist = (
        "normal" if (arch_type == "EGARCH") else "studentst"
    )  # Student's t-dist instead of Gauss for fat-tails

    for i, tr in enumerate(scaled_log_returns.T):
        model = arch_model(
            tr,
            mean="Constant",  # r_t = mu + e_t
            vol=arch_type,
            p=1,  # Lagged Conditional Variance
            q=1,  # Lagged Squared Shock
            dist=dist,
        )

        # Model training
        arch_res = model.fit(disp="off", options={"maxiter": 300, "ftol": 1e-4})

        # Forecast volatility for next period (Current + t)
        forecast_method = (
            "simulation" if (arch_type == "EGARCH" and t > 1) else "analytic"
        )
        forecasts = arch_res.forecast(
            horizon=t, method=forecast_method
        )  # forecast for t days

        # Volatility for (Current + t) (converted back to decimal)
        next_period_variance = (
            forecasts.variance.iloc[-1].sum() * trading_days_per_year
        ) / t
        next_period_vol[i] = np.sqrt(next_period_variance) / scale

        # Expected returns for (Current + t) (converted back to decimal)
        next_period_returns[i] = (
            forecasts.mean.iloc[-1, 0] * trading_days_per_year
        ) / scale  # type: ignore

    return (next_period_returns, next_period_vol)
