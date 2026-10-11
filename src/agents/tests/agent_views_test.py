import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data import HistoricalNewsProvider
from src.agents import FinancialViewsAgent, MarketViewsReport, translate_views
from src.config import Settings
from src.portfolio import black_litterman, find_max_sharpe_mocked


@dataclass
class AgentTestResult:
    report: MarketViewsReport | None
    test_result_df: pd.DataFrame


class AgentUnitTest:
    def __init__(self, settings: Settings | None = None):
        """Initialize unit tests with configuration settings and mock asset paths.

        Parameters
        ----------
        settings : Settings | None, optional
            Application settings containing LLM agent configurations.
            If None, loaded from TOML via Settings.from_toml().
        """
        project_root = Path(__file__).parent.parent
        mocks_path = project_root / "mocks"

        self.settings = Settings.from_toml() if settings is None else settings

        self.mock_tickers_path = mocks_path / "mock_tickers.json"
        self.mock_news_path = mocks_path / "mock_news.json"

        self.model = self.settings.agent.llm_name
        self.base_url = self.settings.agent.agent_url
        self.gemini_api_key = self.settings.gemini_api_key

    def run_agent_views_test(
        self,
        scenario: str | None = None,
        mocked_news: bool = True,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> AgentTestResult:
        if mocked_news:
            mock_tickers_ns = json.loads(
                self.mock_tickers_path.read_text(encoding="utf-8"),
                object_hook=lambda d: SimpleNamespace(**d),
            )
            tickers = mock_tickers_ns.tickers
        else:
            if start_date is None or end_date is None or as_of_date is None:
                raise ValueError(
                    "Some 'start_date', 'end_date' and 'as_of_date' are required for the live news (mocked_news=False)."
                )

            if tickers is None:
                raise ValueError(
                    "The 'tickers' param is required for the live news (mocked_news=False)."
                )

        report = self._download_and_analyse_views(
            scenario,
            tickers,  # type: ignore
            mocked_news,
            start_date,
            end_date,
            as_of_date,
        )

        result_df = self._translate_and_retrieve_calculated_data(
            tickers,  # type: ignore
            report,
        )

        return AgentTestResult(report, result_df)

    def _download_and_analyse_views(
        self,
        scenario_name: str | None,
        tickers: list[str],
        mocked_news: bool,
        start_date: str | None,
        end_date: str | None,
        as_of_date: str | None,
    ) -> MarketViewsReport | None:
        if mocked_news:
            if scenario_name is None:
                raise ValueError(
                    "The 'scenario_name' param is required for the live news (mocked_news=True)."
                )

            news_ns = json.loads(
                self.mock_news_path.read_text(encoding="utf-8"),
                object_hook=lambda d: SimpleNamespace(**d),
            )
            as_of_date = news_ns.neutral_noise.as_of_date
        else:
            if start_date is None or end_date is None or as_of_date is None:
                raise ValueError(
                    "Some 'start_date', 'end_date' and 'as_of_date' are required for the live news (mocked_news=False)."
                )

            news_provider = HistoricalNewsProvider(settings=self.settings)
            news_provider.fetch_historical_news_fmp(tickers, start_date, end_date)
            news_ns = news_provider.get_news_for_rebalance(as_of_date, tickers)

        print("=" * 70)
        print(
            f"|> RUN TEST SCENARIO: {scenario_name.upper() if mocked_news else 'live news'} (As of: {as_of_date})"  # type: ignore
        )
        print("=" * 70)

        agent = FinancialViewsAgent(
            model=self.model, api_key=self.gemini_api_key, base_url=self.base_url
        )
        print("\n[1/4] Agent is analysing news and preparing hypothesis...")

        assets_index = pd.Index(tickers)
        report = agent.analyze_market_and_generate_views(
            as_of_date=as_of_date,  # type: ignore
            asset_universe=assets_index,
            news_payload=news_ns,
        )

        return report

    def _translate_and_retrieve_calculated_data(
        self,
        tickers: list[str],
        report: MarketViewsReport | None,
    ) -> pd.DataFrame:
        if report is None:
            raise ValueError("There is no report generated")

        assets_index = pd.Index(tickers)
        N = len(tickers)

        # Mock base assets volatility
        base_vols = np.array([random.randint(20, 80) / 100 for _ in range(N)])
        base_vols = base_vols / base_vols.sum()

        corr = np.full((N, N), 0.55)
        np.fill_diagonal(corr, 1.0)
        cov_matrix = np.diag(base_vols) @ corr @ np.diag(base_vols)  # mocked cov_matrix

        # S&P 500 capitalization (assets market weights)
        w_market = np.array([random.randint(20, 80) / 100 for _ in range(N)])
        w_market = w_market / w_market.sum()

        print("\n[2/4] Translate hypothesis into matrix P, Q, Omega...")
        translated = translate_views(report, assets_index, cov_matrix)

        if translated is None:
            raise ValueError("Couldn't translate the vies into BL matrix")

        print(
            "\n[3/4] Calculation of CAPM (Prior) equilibrium and posteriori distribution (Black-Litterman)..."
        )
        risk_free_rate = 0.04
        settings = Settings.from_toml()
        risk_unacceptance_value = settings.portfolio.risk_unacceptance_value

        pi_equilibrium = (
            risk_unacceptance_value * (cov_matrix @ w_market) + risk_free_rate
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
                "Market (Cap)": [f"{w:.2%}" for w in w_market],
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
