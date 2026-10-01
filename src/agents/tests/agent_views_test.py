import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.constants import RISK_UNACCEPTANCE_VALUE
from src.agents import FinancialViewsAgent, MarketViewsReport, translate_views
from src.portfolio import black_litterman, find_max_sharpe_mocked


@dataclass
class AgentTestResult:
    report: MarketViewsReport | None
    test_result_df: pd.DataFrame


class AgentUnitTest:
    def __init__(
        self,
    ):
        project_root = Path(__file__).parent.parent
        mocks_path = project_root / "mocks"

        self.mock_tickers_path = mocks_path / "mock_tickers.json"
        self.mock_news_path = mocks_path / "mock_news.json"

    def run_agent_views_test(
        self, model: str, base_url: str, scenario: str
    ) -> AgentTestResult:
        mock_tickers_ns = json.loads(
            self.mock_tickers_path.read_text(encoding="utf-8"),
            object_hook=lambda d: SimpleNamespace(**d),
        )
        tickers = mock_tickers_ns.tickers

        report = self.__download_and_analyse_views(model, base_url, scenario, tickers)

        result_df = self.__translate_and_retrieve_calculated_data(
            tickers,
            report,
        )

        return AgentTestResult(report, result_df)

    def __download_and_analyse_views(
        self,
        model: str,
        base_url: str,
        scenario_name: str,
        tickers: list[str],
    ) -> MarketViewsReport | None:
        mock_news_ns = json.loads(
            self.mock_news_path.read_text(encoding="utf-8"),
            object_hook=lambda d: SimpleNamespace(**d),
        )

        as_of_date = mock_news_ns.neutral_noise.as_of_date
        print("=" * 70)
        print(f"|> RUN TEST SCENARIO: {scenario_name.upper()} (As of: {as_of_date})")
        print("=" * 70)

        agent = FinancialViewsAgent(model=model, base_url=base_url)
        print("\n[1/4] Agent is analysing news and preparing hypothesis...")

        assets_index = pd.Index(tickers)
        report = agent.analyze_market_and_generate_views(
            as_of_date=as_of_date,
            asset_universe=assets_index,
            news_payload=mock_news_ns,
        )

        return report

    def __translate_and_retrieve_calculated_data(
        self,
        tickers: list[str],
        report: MarketViewsReport | None,
    ) -> pd.DataFrame:
        if report is None:
            raise ValueError("There is no report generated")

        assets_index = pd.Index(tickers)
        N = len(tickers)

        # Mock base assets volatility
        base_vols = np.array([0.22, 0.25, 0.22, 0.42])

        corr = np.full((N, N), 0.55)
        np.fill_diagonal(corr, 1.0)
        cov_matrix = np.diag(base_vols) @ corr @ np.diag(base_vols)  # mocked cov_matrix

        # S&P 500 capitalization (assets market weights)
        w_market = np.array([0.28, 0.22, 0.30, 0.20])

        print("\n[2/4] Translate hypothesis into matrix P, Q, Omega...")
        translated = translate_views(report, assets_index, cov_matrix)

        if translated is None:
            raise ValueError("Couldn't translate the vies into BL matrix")

        print(
            "\n[3/4] Calculation of CAPM (Prior) equilibrium and posteriori distribution (Black-Litterman)..."
        )
        risk_free_rate = 0.04

        pi_equilibrium = (
            RISK_UNACCEPTANCE_VALUE * (cov_matrix @ w_market) + risk_free_rate
        )  # mocked

        (mu_pri, _) = black_litterman(pi_equilibrium, cov_matrix, 0.04, None, None)
        w_prior = find_max_sharpe_mocked(mu_pri, cov_matrix, risk_free_rate)

        (mu_post, cov_post) = black_litterman(
            pi_equilibrium, cov_matrix, 0.04, translated.Q, translated.P
        )
        w_post = find_max_sharpe_mocked(mu_post, cov_post, risk_free_rate)

        print("\n[4/4] FINAL COMPARISON OF PORTFOLIO WEIGHTS:")
        df_comparison = pd.DataFrame(
            {
                "Market (Cap)": w_market,
                "Prior Ret (Pi)": [f"{r:.1%}" for r in mu_pri],
                "Posterior Ret (BL)": [f"{r:.1%}" for r in mu_post],
                "Weight: Prior": [f"{w:.1%}" for w in w_prior],
                "Weight: BL + Agent": [f"{w:.1%}" for w in w_post],
                "Delta Shift": [
                    f"{(w_post - w_pri):+.1%}" for w_pri, w_post in zip(w_prior, w_post)
                ],
            },
            index=tickers,
        )

        return df_comparison
