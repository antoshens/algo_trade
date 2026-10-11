"""Application settings and configuration management."""

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import SettingsConfigDict

load_dotenv()


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio optimization and market parameters.

    Attributes
    ----------
    initial_capital : float
        Starting portfolio capital in currency units.
    risk_unacceptance_value : float
        Degree of investor risk aversion / penalty coefficient (typically between 2 and 3).
    trading_days_per_year : int
        Annual number of active trading days (typically 252).
    """

    initial_capital: float
    risk_unacceptance_value: float
    trading_days_per_year: int


@dataclass(frozen=True)
class AgentConfig:
    """LLM agent connectivity parameters.

    Attributes
    ----------
    llm_name : str
        Model identifier used for market hypothesis analysis.
    agent_url : str
        Endpoint URL for the LLM agent provider or local inference engine.
    """

    llm_name: str
    agent_url: str


@dataclass(frozen=True)
class Settings:
    """Consolidated application settings loaded from TOML and environment variables.

    Attributes
    ----------
    portfolio : PortfolioConfig
        Portfolio and risk parameters.
    agent : AgentConfig
        LLM agent configuration.
    environment : str, default="development"
        Execution environment name.
    path : Path
        Default path to the TOML configuration file.
    gemini_api_key : str | None, optional
        API key for Google Gemini provider.
    finnhub_api_key : str | None, optional
        API key for Finnhub news provider.
    """

    portfolio: PortfolioConfig
    agent: AgentConfig

    environment: str = "development"
    path: Path = Path(__file__).resolve().parent / "general_settings.toml"

    gemini_api_key: str | None = None
    finnhub_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def trading_days_per_year(self) -> int:
        """Convenience property for portfolio trading days per year."""
        return self.portfolio.trading_days_per_year

    @property
    def risk_unacceptance_value(self) -> float:
        """Convenience property for portfolio risk unacceptance value."""
        return self.portfolio.risk_unacceptance_value

    @classmethod
    @lru_cache
    def from_toml(cls, path: Path | str | None = None) -> "Settings":
        """Load and instantiate Settings from a TOML configuration file.

        Parameters
        ----------
        path : Path | str | None, optional
            Path to the TOML configuration file. If None, defaults to `cls.path`.

        Returns
        -------
        Settings
            Loaded frozen Settings instance.
        """
        config_path = Path(path) if path is not None else cls.path
        with config_path.open("rb") as file:
            data = tomllib.load(file)

        portfolio_data = data["portfolio"]
        agent_data = data["agent"]

        return cls(
            portfolio=PortfolioConfig(
                initial_capital=float(portfolio_data["initial_capital"]),
                risk_unacceptance_value=float(
                    portfolio_data["risk_unacceptance_value"]
                ),
                trading_days_per_year=int(portfolio_data["trading_days_per_year"]),
            ),
            agent=AgentConfig(
                llm_name=agent_data.get("llm_name"),
                agent_url=agent_data.get("agent_url"),
            ),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
        )
