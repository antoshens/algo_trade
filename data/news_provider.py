import os
import typing
from datetime import timedelta

import duckdb
import finnhub
import pandas as pd
from dotenv import load_dotenv


class HistoricalNewsProvider:
    def __init__(
        self, api_key: str | None = None, cache_file: str = "./cache/news_cache.parquet"
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.cache_file = cache_file
        self.news_df = self._load_cache()

    def _load_cache(self) -> pd.DataFrame:
        if os.path.exists(self.cache_file):
            return duckdb.query(f"SELECT * FROM '{self.cache_file}'").df()

        return pd.DataFrame(columns=["date", "ticker", "title", "text"])

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        df = self.news_df  # noqa: F841
        duckdb.query(f"COPY df TO '{self.cache_file}' (FORMAT PARQUET)")

    def fetch_historical_news_fmp(
        self,
        tickers: typing.Sequence[str],
        start_date: str,
        end_date: str,
    ):
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY is not configured.")

        new_records = []
        for ticker in tickers:
            finnhub_client = finnhub.Client(api_key=self.api_key)

            try:
                response = finnhub_client.company_news(
                    ticker, _from=start_date, to=end_date
                )
                for item in response:
                    new_records.append(
                        {
                            "date": pd.to_datetime(item.get("datetime"), unit="s"),
                            "ticker": ticker,
                            "title": item.get("headline", ""),
                            "text": item.get("summary", ""),
                        }
                    )
            except Exception as e:
                print(f"[News Error] Could not fetch news for {ticker}: {e}")

        if new_records:
            df_new = pd.DataFrame(new_records)
            self.news_df = pd.concat([self.news_df, df_new], ignore_index=True)
            self.news_df.drop_duplicates(
                subset=["date", "ticker", "title"], inplace=True
            )
            self.news_df.sort_values(by="date", inplace=True)
            self._save_cache()
            print(f"Cache has been updated: saved {len(self.news_df)} news.")

    def get_news_for_rebalance(
        self,
        as_of_date: str | pd.Timestamp,
        tickers: typing.Sequence[str],
        lookback_days: int = 7,
        max_news_per_ticker: int = 3,
    ) -> dict[str, list[str]]:
        as_of = pd.to_datetime(as_of_date)
        start_dt = as_of - timedelta(days=lookback_days)

        if self.news_df.empty:
            return {ticker: [] for ticker in tickers}

        # Filter Point-in-Time
        mask = (
            (self.news_df["date"] >= start_dt)
            & (self.news_df["date"] < as_of)
            & (self.news_df["ticker"].isin(tickers))
        )
        sub_df = self.news_df[mask]

        payload = {}
        for ticker in tickers:
            ticker_news = sub_df[sub_df["ticker"] == ticker].tail(max_news_per_ticker)
            # Compacted strings "YYYY-MM-DD: Title - Short text"
            headlines = [
                f"{row['date'].strftime('%Y-%m-%d')}: {row['title']}"
                for _, row in ticker_news.iterrows()
            ]
            payload[ticker] = headlines

        return payload
