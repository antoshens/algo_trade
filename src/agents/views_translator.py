"""
Views Translator Module.

Translates structured market views (from an AI analyst or expert report)
into numerical matrix formulations (P, Q, Omega) required by the Black-Litterman
portfolio optimization model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_contract import AssetView, MarketViewsReport


@dataclass
class TranslatedViews:
    """
    Container for Black-Litterman view matrices and associated metadata.

    Attributes
    ----------
    P : np.ndarray
        Pick/transition matrix of shape (K, N) mapping K investor views
        to the N assets in the portfolio universe.
    Q : np.ndarray
        Expected outperformance/return vector of shape (K, 1) corresponding
        to each view.
    Omega : np.ndarray
        Diagonal covariance/uncertainty matrix of shape (K, K) representing
        the variance associated with each view.
    valid_views : list[AssetView]
        List of view objects that were successfully validated and mapped to
        the asset universe.
    aggregation_date : str
        Effective date (`as_of_date`) of the source market views report.
    """

    P: np.ndarray  # shape (K, N)
    Q: np.ndarray  # shape (K, 1)
    Omega: np.ndarray  # shape (K, K)
    valid_views: list[AssetView]
    aggregation_date: str


def translate_views(
    report: MarketViewsReport,
    assets: pd.Index,
    cov_matrix: np.ndarray,
    tau: float = 0.05,
) -> TranslatedViews | None:
    """
    Translate a market views report into Black-Litterman matrices P, Q, and Omega.

    Filters incoming views against the universe of assets and constructs:
    - Matrix P (pick matrix): rows with 1 for absolute views, or 1 and -1
      for relative pairs.
    - Vector Q (expected returns): expected outperformance for each valid view.
    - Matrix Omega (view uncertainty): diagonal matrix with variance computed
      using the view's confidence level and prior covariance scaled by tau:
      Omega_k = (p_k @ (tau * cov_matrix) @ p_k.T) * ((1 - confidence) / confidence).

    Parameters
    ----------
    report : MarketViewsReport
        The market views report containing the date and list of asset views.
    assets : pd.Index
        Index of asset tickers defining the portfolio universe (dimension N).
    cov_matrix : np.ndarray
        Asset return covariance matrix of shape (N, N).
    tau : float, optional
        Uncertainty scalar for the prior equilibrium distribution, by default 0.05.

    Returns
    -------
    TranslatedViews | None
        An instance of TranslatedViews containing matrices P, Q, Omega,
        valid views, and aggregation date, or None if no valid views were found.
    """
    views = report.views
    N = assets.shape[0]  # num of assets

    p_rows = []
    omega_rows = []
    valid_views = []
    asset_to_idx = {ticker: i for i, ticker in enumerate(assets)}
    for view in views:
        if view.base_asset not in assets:
            continue

        row = np.zeros(N)
        # Matrix P
        if view.view_type == "ABSOLUTE":
            asset_ind = asset_to_idx[view.base_asset]
            row[asset_ind] = 1
        elif view.view_type == "RELATIVE":
            if (
                view.target_asset is None
                or view.target_asset not in assets
                or view.base_asset == view.target_asset
            ):
                continue

            base_asset_ind = asset_to_idx[view.base_asset]
            target_asset_ind = asset_to_idx[view.target_asset]
            row[base_asset_ind] = 1
            row[target_asset_ind] = -1

        # Matrix Omega
        # View base variance
        base_variance = float(row @ (tau * cov_matrix) @ row.T)

        confidence = min(
            max(view.confidence, 0.001), 0.999
        )  # safeguard against confidence >= 1, which lead to a division by zero
        omega_rows.append(base_variance * ((1 - confidence) / confidence))
        p_rows.append(row)

        valid_views.append(view)

    if len(valid_views) == 0:
        return None

    P = np.array(p_rows)
    Omega = np.diag(omega_rows)

    # Vector Q
    Q = np.array([view.expected_outperformance for view in valid_views]).reshape(-1, 1)

    return TranslatedViews(
        P=P,
        Q=Q,
        Omega=Omega,
        valid_views=valid_views,
        aggregation_date=report.as_of_date,
    )
