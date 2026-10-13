# 📈 algo_trade

<p align="center">
  <strong>Algorithmic Trading & Portfolio Optimization Engine</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Active%20Development-blueviolet?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/Quant-Finance-informational?style=for-the-badge" alt="Quant Finance"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome"></a>
</p>

---

An algorithmic trading, portfolio optimization, and quantitative backtesting engine implemented in Python. algo_trade combines Modern Portfolio Theory (MPT), econometric volatility forecasting, LLM-powered market sentiment analysis, and realistic market microstructure execution modeling.

---

## 🌟 Overview

algo_trade provides modular tools for quantitative asset allocation, risk modeling, and strategy backtesting. The platform evaluates portfolios under realistic execution conditions (turnover costs, drift, commissions) and incorporates AI-driven market views into quantitative allocation models like Black-Litterman.

---

## ⚡ Key Features

### 🎯 Portfolio Optimization & Modern Portfolio Theory
- 📊 **Mean-Variance Optimization**: Efficient Frontier generation via classical Markowitz formulation.
- 🎯 **Dual Objective Functions**:
  - **Maximum Sharpe Ratio**: Identifies the tangency portfolio against specified risk-free benchmarks.
  - **Maximum Sortino Ratio**: Optimizes for downside risk using lower partial moments.
- 📐 **Covariance Estimators**:
  - Sample Covariance Matrix (annualized).
  - Ledoit-Wolf Shrinkage covariance estimator for high-dimensional stability.
  - Univariate (E)GARCH conditional volatility projections.
- 🔮 **Expected Returns Modeling**:
  - Historical mean log returns.
  - CAPM implied equilibrium returns via market capitalization weights.
  - **Black-Litterman Model**: Blends market equilibrium priors with investor or AI-generated views and confidence intervals.
- ⚖️ **Turnover Regularization**: L1 penalties (fixed or structural cost coupling) to constrain transaction costs and portfolio churn.
- 🏦 **Macro Risk-Free Benchmarks**: Direct FRED rate fetching for 13-week T-Bills (`^IRX`), 10-year Treasury Notes (`^TNX`), and SOFR (`SR3=F`).

### 🤖 LLM Financial Views Agent
- 📰 **News Ingestion & Caching**: Historical company news retrieval via Finnhub with local DuckDB/Parquet caching (`HistoricalNewsProvider`).
- 🧠 **Multi-Provider LLM Support**: Interfaces with Google Gemini (via Google GenAI SDK) and local/remote OpenAI-compatible endpoints (Ollama, vLLM).
- 📋 **Structured Data Contracts**: Pydantic schemas (`AssetView`, `MarketViewsReport`) with confidence scoring, return bounds, and sentiment summaries.
- 📐 **Black-Litterman Translation**: Converts textual market views into pick ($P$), return ($Q$), and uncertainty ($\Omega$) matrices.

### 📉 Econometric & Volatility Modeling
- 🔬 **GARCH & EGARCH Models**: Univariate GARCH(1,1) and Exponential GARCH(1,1) with Student's $t$-distribution to capture return fat-tails and volatility clustering.
- 🔍 **Market Microstructure Analytics**:
  - Parkinson extreme-value rolling volatility estimator.
  - Corwin-Schultz high-low bid-ask spread estimators.
  - Overnight price gap distributions and intraday return analytics.

### 🔄 Out-of-Sample Backtesting Engine
- 🔁 **Walk-Forward Evaluation**: Rolling train-and-test windows via `BacktestEngine` with customizable lookback periods and rebalancing cycles.
- 🤖 **Agent-in-the-Loop Rebalancing**: Optional automated LLM sentiment analysis and view generation during backtest rebalance periods.
- 🛡️ **Execution Realism**: Simulates broker turnover commissions, transaction drag, and intra-period asset weight drift.
- 🏁 **Benchmark Comparison**: Automatic tracking against equal-weighted allocations and the S&P 500 (`^GSPC`).

### ⚙️ Configuration Management
- 🛠️ **Centralized Settings**: Application parameters managed via `src/config/general_settings.toml` and `.env` (portfolio capital, risk aversion, LLM model/endpoint, API keys).

### 📊 Performance & Risk Analytics
- 📈 **CAGR**: Compound Annual Growth Rate.
- 📉 **Max Drawdown (MDD)**: Peak-to-trough equity degradation.
- ⚖️ **Risk-Adjusted Ratios**: Sharpe Ratio, Sortino Ratio, and Calmar Ratio.
- 🌊 **Rolling Metrics**: Dynamic rolling Sharpe and Sortino ratio tracking over time.

---

## 🗂️ Project Structure

```text
algo_trade/
├── data/                    # 📥 Market data retrieval and preprocessing
│   ├── download_data.py     # Yahoo Finance historical ingestion
│   ├── news_provider.py     # Historical news fetching and DuckDB caching
│   └── processors.py        # Returns, spreads, volatility, and feature engineering
├── src/
│   ├── agents/              # 🤖 LLM market analyst agent and views translator
│   │   ├── agent_view.py    # FinancialViewsAgent (Gemini & OpenAI/Ollama)
│   │   ├── data_contract.py # Pydantic schemas for structured market views
│   │   ├── prompts.py       # System and analyst prompts
│   │   └── views_translator.py # Matrix mapping (P, Q, Omega) for Black-Litterman
│   ├── config/              # ⚙️ Configuration management & application settings
│   │   ├── settings.py      # Settings dataclasses and loader
│   │   └── general_settings.toml # General portfolio and agent parameters
│   ├── portfolio/           # 🧠 Optimization and econometric models
│   │   ├── markowitz.py     # Mean-variance optimization & efficient frontier
│   │   ├── black_litterman.py # Black-Litterman allocation model
│   │   └── garch.py         # ARCH / GARCH conditional volatility forecasting
│   └── backtest/            # ⚙️ Walk-forward backtesting and metrics
│       ├── engine.py        # BacktestEngine simulation & turnover execution
│       └── metrics.py       # Performance, drawdown, and risk ratios
├── notebooks/               # 📓 Research and exploratory workflows
│   ├── agents/              # LLM agent workflows and mocked tests
│   ├── market_analysis/     # Distributions, market memory, and volatility
│   ├── modern_portfolio_theory/ # Shrinkage & optimization experiments
│   └── models_backtesting/  # Strategy backtesting walkthroughs
├── pyproject.toml           # 📦 Packaging configuration
└── requirements.txt         # 📋 Project dependencies
```

---

## 🚀 Installation & Setup

### 📋 Prerequisites
- **Python 3.11+**
- **Git**

### 💻 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<username>/algo_trade.git
   cd algo_trade
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate on Windows:
   .venv\Scripts\activate

   # Activate on Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Optional: Install package in editable development mode:*
   ```bash
   pip install -e .
   ```

---

## 💡 Quickstart & Usage

### 1️⃣ Market Data Ingestion
```python
from datetime import datetime
from data.download_data import download_tickers_history

# Fetch historical daily data for a basket of assets
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

prices_df = download_tickers_history(
    start_date=start_date, end_date=end_date, tickers=tickers
)
```

### 2️⃣ Portfolio Optimization
```python
from src.portfolio import optimize_portfolio

# Compute the Efficient Frontier with Ledoit-Wolf shrinkage & L1 turnover penalty
frontier_df = optimize_portfolio(
    tickers_df=prices_df,
    rf_base="T_BILLS",
    cov_model="LEDOIT_WOLF",
    returns_model="HISTORICAL",
    opt_type="BACKTEST",
    l1_coeff=0.01,
)

print(frontier_df.head())
```

### 3️⃣ Strategy Backtesting
```python
from src.backtest import BacktestEngine

# Initialize backtesting engine (loads settings from TOML)
engine = BacktestEngine()

# Run out-of-sample walk-forward backtest
backtest_results = engine.perform_backtesting(
    tickers_df=prices_df,
    init_portfolio_value=10000.0,
    optimization_metric="GARCH_SHARPE",
    returns_model="HISTORICAL",
    lookback_window=504,       # ~2 years training window
    rebalancing_period=21,     # Rebalance monthly
    broker_commission=0.0005,  # 5 bps transaction fee
    l1_penalty_type="FIXED",
)

print(backtest_results[["portfolio_value", "equal_portfolio_value"]].tail())
```

### 4️⃣ LLM Financial Views Agent (Optional)
```python
from src.agents import FinancialViewsAgent

# Initialize agent (supports Google Gemini SDK or local Ollama/vLLM)
agent = FinancialViewsAgent(
    model="gemini-2.5-flash",
    api_key="YOUR_GEMINI_API_KEY",
    googleSdk=True,
)

# Generate structured views from market context
report = agent.analyze_market_and_generate_views(
    as_of_date="2024-01-01",
    asset_universe=["AAPL", "MSFT", "GOOGL", "AMZN"],
    news_payload=news_records,
)

print(report.market_sentiment_summary)
for view in report.views:
    print(f"{view.base_asset}: expected outperformance {view.expected_outperformance:.1%}, confidence {view.confidence}")
```

---

## 🔬 Research Notebooks

Interactive Jupyter notebooks are organized in the [`notebooks/`](notebooks/) directory:
- 🤖 **Agents**: LLM market news synthesis, structured view generation, and mock scenarios.
- 📊 **Market Analysis**: Statistical return distribution tests, fat-tail behavior, and volatility clustering.
- 📐 **Modern Portfolio Theory**: Empirical comparisons across Sample Covariance, Ledoit-Wolf Shrinkage, and Black-Litterman models.
- 🧪 **Models Backtesting**: Walk-forward backtesting demonstrations evaluating GARCH and EGARCH allocation strategies.

---

## 🗺️ Roadmap

- [x] 🤖 **LLM Agent Integration**: Incorporate Large Language Models (Gemini, Ollama, vLLM) to synthesize market sentiment into Black-Litterman investor views.
- [ ] 📉 **Non-Linear Slippage**: Dynamic impact cost modeling based on trading volume and order book depth.
- [ ] 🔌 **Live Execution Connector**: Interactive Brokers (IBKR) API integration for automated paper and live trade execution.

---

## ⚠️ Disclaimer

> [!WARNING]
> This software is intended strictly for research and educational purposes. It does not constitute financial, investment, or trading advice. Past performance simulated in backtests is no guarantee of future returns.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

