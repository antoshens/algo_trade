"""
Financial Views Agent Module.

Provides an LLM-powered financial analyst agent capable of parsing market news,
synthesizing market sentiment, and producing structured market views (via Pydantic contracts)
for assets in an investment universe, supporting both Google Gemini SDK and OpenAI-compatible endpoints.
"""

import json
from collections.abc import Sequence

import pandas as pd
from google import genai
from google.genai import types
from openai import OpenAI

from .data_contract import MarketViewsReport
from .prompts import (
    FINANCIAL_ANALYST_AGENT_SYSTEM_PROMPT,
    FINANCIAL_ANALYST_AGENT_USER_PROMPT,
)


class FinancialViewsAgent:
    """
    LLM-powered financial market analyst agent.

    Interfaces with either Google Gemini API (via Google GenAI SDK) or local/remote
    OpenAI-compatible LLM servers (such as vLLM or Ollama) to analyze news headlines
    and qualitative market context, outputting structured investment views conformant
    to `MarketViewsReport`.

    Attributes
    ----------
    client : genai.Client | OpenAI
        Underlying API client instance.
    model : str
        Name of the LLM model used for inference.
    temperature : float
        Sampling temperature used for generation.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,  # for vLLM/Ollama
        googleSdk: bool = False,
        temperature: float = 0.1,
    ):
        """
        Initialize the FinancialViewsAgent.

        Parameters
        ----------
        model : str
            The LLM model identifier (e.g., 'gemini-2.5-flash', 'llama3:8b').
        api_key : str | None, optional
            API key for the chosen provider. If using an
            OpenAI-compatible server (with `base_url`) and None is provided, defaults to 'ollama'.
        base_url : str | None, optional
            Base URL for OpenAI-compatible endpoints (e.g., 'http://localhost:11434/v1' for Ollama/vLLM).
            When None and api_key is provided, Google GenAI SDK is used.
        googleSdk : bool, optional
            Whether Google GenAI SDK is utilized. False by default.
        temperature : float, optional
            Sampling temperature for inference, by default 0.1.
        """
        if base_url is not None:  # vLLM/Ollama
            api_key = "ollama"
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif api_key is not None:  # not a local model
            if googleSdk:  # Google SDK
                self.client = genai.Client(api_key=api_key)
            else:  # OpenAI client
                self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(
                "At least one param ('api_key' or 'base_url) should be passed."
            )

        self.model = model
        self.temperature = temperature

    def analyze_market_and_generate_views(
        self,
        as_of_date: str,
        asset_universe: Sequence[str] | pd.Index,
        news_payload: dict[str, list[str]] | str,
    ) -> MarketViewsReport | None:
        """
        Analyze news headlines and context to generate structured investor views for the asset universe.

        Formats system and user prompts with the provided asset universe and news payload,
        calls the LLM using structured output schema enforcement (`MarketViewsReport`),
        and returns validated view objects. If an error occurs during inference or parsing,
        falls back to an empty report with a sentiment summary warning.

        Parameters
        ----------
        as_of_date : str
            Effective observation date of the analysis (e.g., '2026-09-30').
        asset_universe : Sequence[str] | pd.Index
            List or Index of asset tickers considered for view generation.
        news_payload : dict[str, list[str]] | str
            News articles, headlines, or market commentary provided as a dictionary or raw string.

        Returns
        -------
        MarketViewsReport | None
            Structured report containing market sentiment summary and individual asset views,
            or an empty fallback report if generation fails.
        """
        if isinstance(news_payload, dict):
            none_keys = [key for key, value in news_payload.items() if len(value) == 0]
            if len(none_keys) == len(news_payload):
                print(
                    "[Agent Warning] The 'news_payload' does not contain any news for any of the passed tickers."
                )
                return

        assets_list = list(asset_universe)
        formatted_sys_prompt = FINANCIAL_ANALYST_AGENT_SYSTEM_PROMPT.format(
            assets_list=", ".join(assets_list)
        )

        # Build input params
        if isinstance(news_payload, dict):
            context_str = json.dumps(news_payload, indent=2, ensure_ascii=False)
        else:
            context_str = str(news_payload)

        user_prompt = FINANCIAL_ANALYST_AGENT_USER_PROMPT.format(
            as_of_date=as_of_date,
            assets_list=", ".join(assets_list),
            context_str=context_str,
        )

        try:
            if isinstance(self.client, genai.Client):
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=formatted_sys_prompt,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=MarketViewsReport,
                    ),
                )

                response_text = response.text if response.text is not None else ""
                return MarketViewsReport.model_validate_json(response_text)
            else:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": formatted_sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=MarketViewsReport,
                    temperature=self.temperature,
                )

                report = response.choices[0].message.parsed
                if report is None:
                    print("[Agent Warning] No views generated.")
                else:
                    print(f"[Agent Success] Generated {len(report.views)} views.")

                return report

        except Exception as e:
            # Fallback in case of erros during parsing
            print(f"[Agent Warning] Failed to generate views as of {as_of_date}: {e}")
            return MarketViewsReport(
                as_of_date=as_of_date,
                market_sentiment_summary="[Agent fallback]: empty views generated due to an error.",
                views=[],
            )
