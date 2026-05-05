from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


DEFAULT_CACHE_ROOT = Path("data/agent_outputs/q4_2025_ai_tech")

Q4_2025_AI_TECH_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "AAPL", "company": "Apple", "earnings_date": "2026-01-29", "bucket": "mega_cap_platform"},
    {"ticker": "MSFT", "company": "Microsoft", "earnings_date": "2026-01-28", "bucket": "mega_cap_platform"},
    {"ticker": "GOOGL", "company": "Alphabet", "earnings_date": "2026-02-03", "bucket": "mega_cap_platform"},
    {"ticker": "AMZN", "company": "Amazon", "earnings_date": "2026-02-05", "bucket": "mega_cap_platform"},
    {"ticker": "META", "company": "Meta Platforms", "earnings_date": "2026-01-28", "bucket": "mega_cap_platform"},
    {"ticker": "TSLA", "company": "Tesla", "earnings_date": "2026-01-28", "bucket": "mega_cap_platform"},
    {"ticker": "NVDA", "company": "Nvidia", "earnings_date": "2026-02-25", "bucket": "mega_cap_platform"},
    {"ticker": "AMD", "company": "Advanced Micro Devices", "earnings_date": "2026-02-03", "bucket": "ai_semis"},
    {"ticker": "AVGO", "company": "Broadcom", "earnings_date": "2026-03-05", "bucket": "ai_semis"},
    {"ticker": "MU", "company": "Micron Technology", "earnings_date": "2025-12-17", "bucket": "ai_semis"},
    {"ticker": "QCOM", "company": "Qualcomm", "earnings_date": "2026-02-04", "bucket": "ai_semis"},
    {"ticker": "ARM", "company": "Arm Holdings", "earnings_date": "2026-02-04", "bucket": "ai_semis"},
    {"ticker": "TSM", "company": "Taiwan Semiconductor Manufacturing", "earnings_date": "2026-01-15", "bucket": "ai_semis"},
    {"ticker": "ASML", "company": "ASML Holding", "earnings_date": "2026-01-28", "bucket": "ai_semis"},
    {"ticker": "AMAT", "company": "Applied Materials", "earnings_date": "2026-02-12", "bucket": "ai_semis"},
    {"ticker": "LRCX", "company": "Lam Research", "earnings_date": "2026-01-28", "bucket": "ai_semis"},
    {"ticker": "KLAC", "company": "KLA", "earnings_date": "2026-01-29", "bucket": "ai_semis"},
    {"ticker": "ORCL", "company": "Oracle", "earnings_date": "2026-03-10", "bucket": "ai_infrastructure"},
    {"ticker": "DELL", "company": "Dell Technologies", "earnings_date": "2026-02-26", "bucket": "ai_infrastructure"},
    {"ticker": "SMCI", "company": "Super Micro Computer", "earnings_date": "2026-02-03", "bucket": "ai_infrastructure"},
    {"ticker": "ANET", "company": "Arista Networks", "earnings_date": "2026-02-17", "bucket": "ai_infrastructure"},
    {"ticker": "VRT", "company": "Vertiv", "earnings_date": "2026-02-18", "bucket": "ai_infrastructure"},
    {"ticker": "CRM", "company": "Salesforce", "earnings_date": "2026-02-25", "bucket": "software_cloud"},
    {"ticker": "ADBE", "company": "Adobe", "earnings_date": "2026-03-11", "bucket": "software_cloud"},
    {"ticker": "NOW", "company": "ServiceNow", "earnings_date": "2026-01-28", "bucket": "software_cloud"},
    {"ticker": "SNOW", "company": "Snowflake", "earnings_date": "2026-02-25", "bucket": "software_cloud"},
    {"ticker": "PLTR", "company": "Palantir", "earnings_date": "2026-02-03", "bucket": "software_cloud"},
    {"ticker": "DDOG", "company": "Datadog", "earnings_date": "2026-02-12", "bucket": "software_cloud"},
    {"ticker": "MDB", "company": "MongoDB", "earnings_date": "2026-03-04", "bucket": "software_cloud"},
    {"ticker": "NET", "company": "Cloudflare", "earnings_date": "2026-02-12", "bucket": "software_cloud"},
]

Q4_2025_CIO_ADVANTAGE_EXTENSION: list[dict[str, str]] = [
    {"ticker": "MRVL", "company": "Marvell Technology", "earnings_date": "2026-03-05", "bucket": "ai_semis"},
    {"ticker": "WDC", "company": "Western Digital", "earnings_date": "2026-01-29", "bucket": "ai_semis"},
    {"ticker": "STX", "company": "Seagate Technology", "earnings_date": "2026-01-21", "bucket": "ai_semis"},
    {"ticker": "COHR", "company": "Coherent", "earnings_date": "2026-02-05", "bucket": "ai_infrastructure"},
    {"ticker": "ALAB", "company": "Astera Labs", "earnings_date": "2026-02-10", "bucket": "ai_infrastructure"},
    {"ticker": "ETN", "company": "Eaton", "earnings_date": "2026-01-30", "bucket": "power_infrastructure"},
    {"ticker": "PWR", "company": "Quanta Services", "earnings_date": "2026-02-19", "bucket": "power_infrastructure"},
    {"ticker": "CEG", "company": "Constellation Energy", "earnings_date": "2026-02-19", "bucket": "power_infrastructure"},
    {"ticker": "NRG", "company": "NRG Energy", "earnings_date": "2026-02-27", "bucket": "power_infrastructure"},
    {"ticker": "GEV", "company": "GE Vernova", "earnings_date": "2026-01-22", "bucket": "power_infrastructure"},
    {"ticker": "EQIX", "company": "Equinix", "earnings_date": "2026-02-11", "bucket": "data_center_reit"},
    {"ticker": "DLR", "company": "Digital Realty", "earnings_date": "2026-02-12", "bucket": "data_center_reit"},
    {"ticker": "IRM", "company": "Iron Mountain", "earnings_date": "2026-02-20", "bucket": "data_center_reit"},
    {"ticker": "AMT", "company": "American Tower", "earnings_date": "2026-02-25", "bucket": "data_center_reit"},
    {"ticker": "CORZ", "company": "Core Scientific", "earnings_date": "2026-03-12", "bucket": "data_center_operator"},
    {"ticker": "IREN", "company": "IREN", "earnings_date": "2026-02-12", "bucket": "data_center_operator"},
    {"ticker": "CLS", "company": "Celestica", "earnings_date": "2026-01-27", "bucket": "ai_infrastructure"},
    {"ticker": "FLEX", "company": "Flex", "earnings_date": "2026-01-28", "bucket": "ai_infrastructure"},
    {"ticker": "TEAM", "company": "Atlassian", "earnings_date": "2026-01-29", "bucket": "software_cloud"},
    {"ticker": "ZS", "company": "Zscaler", "earnings_date": "2026-03-04", "bucket": "software_cloud"},
    {"ticker": "CRWD", "company": "CrowdStrike", "earnings_date": "2026-03-04", "bucket": "software_cloud"},
    {"ticker": "PANW", "company": "Palo Alto Networks", "earnings_date": "2026-02-12", "bucket": "software_cloud"},
    {"ticker": "OKTA", "company": "Okta", "earnings_date": "2026-03-03", "bucket": "software_cloud"},
    {"ticker": "APP", "company": "AppLovin", "earnings_date": "2026-02-12", "bucket": "software_cloud"},
    {"ticker": "SHOP", "company": "Shopify", "earnings_date": "2026-02-11", "bucket": "software_cloud"},
]


class VertexTextGenerator:
    """Tiny Vertex Gemini wrapper used by agents that expect OpenAI/LangChain-style clients."""

    def __init__(self, project: str, location: str = "us-central1", model: str = "gemini-2.5-flash"):
        self.project = project
        self.location = location
        self.model = model
        self.backend = None
        self.client = None
        self.vertex_model = None

        try:
            from google import genai  # type: ignore

            self.client = genai.Client(vertexai=True, project=project, location=location)
            self.backend = "google_genai"
        except Exception:
            import vertexai  # type: ignore
            from vertexai.generative_models import GenerativeModel  # type: ignore

            vertexai.init(project=project, location=location)
            self.vertex_model = GenerativeModel(model)
            self.backend = "vertexai"

    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, json_mode: bool = True) -> str:
        prompt = "".join(f"{message.get('role', 'user').upper()}: {message.get('content', '')}" for message in messages)
        if self.backend == "google_genai":
            from google.genai import types  # type: ignore

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=8192,
                response_mime_type="application/json" if json_mode else None,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text or "{}"

        from vertexai.generative_models import GenerationConfig  # type: ignore

        response = self.vertex_model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192,
                response_mime_type="application/json" if json_mode else None,
            ),
        )
        return response.text or "{}"


class VertexOpenAICompatibleClient:
    def __init__(self, generator: VertexTextGenerator):
        self.generator = generator
        self.chat = SimpleNamespace(completions=self)

    def create(self, model=None, temperature=1.0, messages=None, **kwargs):
        text = self.generator.generate(messages or [], temperature=temperature, json_mode=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class VertexLangChainCompatibleLLM:
    def __init__(self, generator: VertexTextGenerator, temperature: float = 1.0):
        self.generator = generator
        self.temperature = temperature

    def invoke(self, messages):
        text = self.generator.generate(messages or [], temperature=self.temperature, json_mode=True)
        return SimpleNamespace(content=text)


def q4_2025_ai_tech_earnings_df(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return the expanded Q4 2025 AI/Tech earnings universe."""
    df = pd.DataFrame(Q4_2025_AI_TECH_UNIVERSE)
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        df = df[df["ticker"].str.upper().isin(wanted)].copy()
    return df.reset_index(drop=True)


def q4_2025_cio_advantage_extension_df(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return extra Q4 2025 names where second-order CIO reasoning should matter more."""
    df = pd.DataFrame(Q4_2025_CIO_ADVANTAGE_EXTENSION)
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        df = df[df["ticker"].str.upper().isin(wanted)].copy()
    return df.reset_index(drop=True)


def q4_2025_combined_cio_advantage_df(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return the original AI/Tech universe plus the CIO-advantage extension."""
    df = pd.DataFrame([*Q4_2025_AI_TECH_UNIVERSE, *Q4_2025_CIO_ADVANTAGE_EXTENSION])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        df = df[df["ticker"].str.upper().isin(wanted)].copy()
    return df.reset_index(drop=True)


def ensure_cache_dirs(cache_root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    root = Path(cache_root)
    for folder in (
        "news_macro",
        "market_technical",
        "fundamental",
        "packets",
        "cio",
        "contexts",
        "errors",
        "tables",
    ):
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch(exist_ok=True)
    return root


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached_ticker_paths(ticker: str, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> dict[str, Path]:
    root = Path(cache_root)
    safe = ticker.upper()
    return {
        "news_macro": root / "news_macro" / f"{safe}.json",
        "market_technical": root / "market_technical" / f"{safe}.json",
        "fundamental": root / "fundamental" / f"{safe}.json",
        "packets": root / "packets" / f"{safe}.json",
        "cio": root / "cio" / f"{safe}.json",
        "context": root / "contexts" / f"{safe}.json",
        "errors": root / "errors" / f"{safe}.json",
    }


def has_cached_ticker(ticker: str, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> bool:
    paths = cached_ticker_paths(ticker, cache_root)
    return paths["packets"].exists() and paths["cio"].exists()


def load_cached_ticker(ticker: str, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> dict[str, Any]:
    paths = cached_ticker_paths(ticker, cache_root)
    payload: dict[str, Any] = {"ticker": ticker.upper()}
    for key, path in paths.items():
        payload[key] = load_json(path) if path.exists() else None
    return payload


def build_price_tables(
    earnings_df: pd.DataFrame | None = None,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    pre_trading_days: int = 7,
    post_trading_days: int = 7,
    long_post_trading_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and cache price/eval tables for the selected universe."""
    from openclam.agents.news_macro import news_macro_agent

    root = ensure_cache_dirs(cache_root)
    earnings_df = earnings_df.copy() if earnings_df is not None else q4_2025_ai_tech_earnings_df()
    summary_df, paths_df = news_macro_agent.build_earnings_price_eval(
        earnings_df=earnings_df[["ticker", "company", "earnings_date"]],
        pre_trading_days=pre_trading_days,
        post_trading_days=post_trading_days,
        long_post_trading_days=long_post_trading_days,
        benchmarks=("SPY", "QQQ"),
        price_anchor="event_close",
    )
    summary_df = summary_df.merge(
        earnings_df[["ticker", "bucket"]],
        on="ticker",
        how="left",
    )
    earnings_df.to_csv(root / "tables" / "universe.csv", index=False)
    summary_df.to_csv(root / "tables" / "price_summary.csv", index=False)
    paths_df.to_csv(root / "tables" / "price_paths.csv", index=False)
    return summary_df, paths_df


def _openai_temperature(model_name: str, preferred: float) -> float:
    return 1.0 if (model_name or "").lower().startswith("gpt-5") else preferred


def _vertex_error_hint(exc: Exception) -> str:
    text = repr(exc)
    if "DefaultCredentialsError" in text or "default credentials" in text:
        return "Vertex ADC is missing. Run `gcloud auth application-default login` locally."
    return text


def run_single_ticker_agents(
    ticker: str,
    company: str,
    earnings_date: str,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
    lookback_days: int = 14,
    max_news: int = 10,
    transcript_year: int = 2025,
    transcript_quarter: int = 4,
    llm_provider: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
    vertex_model: str = "gemini-2.5-flash",
    openai_model: str = "gpt-5-nano",
    news_model: str = "gpt-5.4-nano",
) -> dict[str, Any]:
    """Run all three agents for one ticker and save categorized outputs."""
    from openclam.agents.cio import cio_agent
    from openclam.agents.fundamental import fundamental_agent
    from openclam.agents.market_technical import market_technical_agent
    from openclam.agents.news_macro import news_macro_agent

    root = ensure_cache_dirs(cache_root)
    paths = cached_ticker_paths(ticker, root)
    if not force and has_cached_ticker(ticker, root):
        return load_cached_ticker(ticker, root)

    ticker = ticker.upper()
    vertex_project = vertex_project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    vertex_location = vertex_location or os.getenv("VERTEX_LOCATION", "us-central1")
    llm_provider = llm_provider or ("vertex" if vertex_project else "openai")
    agent_outputs: dict[str, Any] = {}
    agent_errors: dict[str, str] = {}
    vertex_generator = None
    if vertex_project:
        try:
            vertex_generator = VertexTextGenerator(vertex_project, vertex_location, vertex_model)
        except Exception as exc:
            agent_errors["vertex_generator"] = _vertex_error_hint(exc)

    try:
        context = news_macro_agent.collect_context(
            ticker=ticker,
            company=company,
            event_date=earnings_date,
            lookback_days=lookback_days,
            max_news=max_news,
            news_mode="event_window",
            news_sources=["finnhub", "newsapi", "yfinance"],
            use_sample_if_empty=False,
            news_end_offset_days=0,
        )
        news_report = news_macro_agent.generate_report(
            context,
            provider="auto",
            model=news_model,
            gemini_model=vertex_model,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )
        agent_outputs["news_macro"] = news_report
        save_json(paths["context"], context)
        save_json(paths["news_macro"], news_report)
    except Exception as exc:
        agent_errors["news_macro"] = repr(exc)

    try:
        if vertex_generator:
            market_llm = VertexLangChainCompatibleLLM(vertex_generator, temperature=1.0)
        else:
            market_llm = market_technical_agent.create_market_llm(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=openai_model,
                temperature=_openai_temperature(openai_model, 0.2),
            )
        market_query = f"{company} ({ticker}) technical analysis as of {earnings_date.replace('-', '/')}"
        market_report = market_technical_agent.run_market_analysis(market_query, llm=market_llm, ticker=ticker)
        agent_outputs["market_technical"] = market_report
        save_json(paths["market_technical"], market_report)
    except Exception as exc:
        agent_errors["market_technical"] = repr(exc)

    try:
        kwargs: dict[str, Any] = {
            "ticker": ticker,
            "transcript_year": transcript_year,
            "transcript_quarter": transcript_quarter,
            "require_transcript": False,
        }
        if vertex_generator:
            kwargs.update(
                {
                    "model_name": vertex_model,
                    "temperature": 1.0,
                    "client": VertexOpenAICompatibleClient(vertex_generator),
                }
            )
        else:
            kwargs.update(
                {
                    "model_name": openai_model,
                    "temperature": _openai_temperature(openai_model, 0.0),
                }
            )
        fundamental_report = fundamental_agent.run_fundamental_analysis(**kwargs)
        agent_outputs["fundamental"] = fundamental_report
        save_json(paths["fundamental"], fundamental_report)
    except Exception as exc:
        agent_errors["fundamental"] = repr(exc)

    packets = []
    if "news_macro" in agent_outputs:
        packets.append(news_macro_agent.to_cio_agent_input(agent_outputs["news_macro"]))
    if "market_technical" in agent_outputs:
        packets.append(cio_agent.to_cio_packet_from_market(agent_outputs["market_technical"]))
    if "fundamental" in agent_outputs:
        packets.append(cio_agent.to_cio_packet_from_fundamental(agent_outputs["fundamental"], company=company))

    save_json(paths["packets"], packets)
    save_json(paths["errors"], agent_errors)
    return {
        "ticker": ticker,
        "company": company,
        "earnings_date": earnings_date,
        "paths": {key: str(value) for key, value in paths.items()},
        "packets": packets,
        "errors": agent_errors,
    }


def run_q4_2025_universe_agents(
    earnings_df: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """Run/cache all three agent outputs for the selected Q4 universe."""
    df = earnings_df.copy() if earnings_df is not None else q4_2025_ai_tech_earnings_df(tickers)
    packets_by_ticker: dict[str, list[dict[str, Any]]] = {}
    errors_by_ticker: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        result = run_single_ticker_agents(
            ticker=row.ticker,
            company=row.company,
            earnings_date=row.earnings_date,
            cache_root=cache_root,
            force=force,
            **kwargs,
        )
        packets_by_ticker[row.ticker] = result.get("packets") or []
        errors_by_ticker[row.ticker] = result.get("errors") or {}
    root = ensure_cache_dirs(cache_root)
    save_json(root / "tables" / "packets_by_ticker.json", packets_by_ticker)
    save_json(root / "tables" / "agent_errors_by_ticker.json", errors_by_ticker)
    return packets_by_ticker, errors_by_ticker


def rerun_cached_market_technical_agent(
    earnings_df: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
    vertex_model: str = "gemini-2.5-flash",
    openai_model: str = "gpt-5-nano",
) -> dict[str, list[dict[str, Any]]]:
    """Refresh only the Market/Technical agent and update cached CIO packets."""
    from openclam.agents.cio import cio_agent
    from openclam.agents.market_technical import market_technical_agent

    root = ensure_cache_dirs(cache_root)
    df = earnings_df.copy() if earnings_df is not None else q4_2025_combined_cio_advantage_df(tickers)
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        df = df[df["ticker"].str.upper().isin(wanted)].copy()

    vertex_project = vertex_project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    vertex_generator = None
    if vertex_project:
        vertex_generator = VertexTextGenerator(vertex_project, vertex_location, vertex_model)

    if vertex_generator:
        market_llm = VertexLangChainCompatibleLLM(vertex_generator, temperature=1.0)
    else:
        market_llm = market_technical_agent.create_market_llm(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=openai_model,
            temperature=_openai_temperature(openai_model, 0.2),
        )

    packets_by_ticker = load_cached_packets_by_ticker(root)
    for row in df.itertuples(index=False):
        ticker = str(row.ticker).upper()
        company = str(row.company)
        earnings_date = str(row.earnings_date)
        paths = cached_ticker_paths(ticker, root)
        query = f"{company} ({ticker}) technical analysis as of {earnings_date}"
        market_report = market_technical_agent.run_market_analysis(query, llm=market_llm, ticker=ticker)
        save_json(paths["market_technical"], market_report)

        packets = packets_by_ticker.get(ticker)
        if not packets and paths["packets"].exists():
            packets = load_json(paths["packets"])
        packets = packets or []
        packets = [
            packet
            for packet in packets
            if "technical" not in str(packet.get("agent_name", "")).lower()
            and "market" not in str(packet.get("agent_name", "")).lower()
        ]
        packets.append(cio_agent.to_cio_packet_from_market(market_report))
        packets_by_ticker[ticker] = packets
        save_json(paths["packets"], packets)

    save_json(root / "tables" / "packets_by_ticker.json", packets_by_ticker)
    return packets_by_ticker


def run_cached_cio_eval(
    summary_df: pd.DataFrame,
    packets_by_ticker: dict[str, list[dict[str, Any]]],
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
    use_llm_debate: bool = True,
    use_llm_decision: bool = True,
    llm_provider: str = "auto",
    debate_model: str = "gemini-2.5-flash",
    decision_model: str = "gemini-2.5-flash",
    vertex_project: str | None = None,
    vertex_location: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Run CIO eval from cached packets and save final/debate outputs."""
    from openclam.agents.cio import cio_agent

    root = ensure_cache_dirs(cache_root)
    cio_eval, cio_results = cio_agent.run_cio_eval(
        summary_df,
        packets_by_ticker,
        long_post_trading_days=long_post_trading_days,
        neutral_band=neutral_band,
        use_llm_debate=use_llm_debate,
        use_llm_decision=use_llm_decision,
        llm_provider=llm_provider,
        debate_model=debate_model,
        decision_model=decision_model,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    for ticker, workflow in cio_results.items():
        save_json(root / "cio" / f"{ticker}.json", workflow)
    summary = cio_agent.evaluate_cio_decisions(cio_eval)
    bucket_summary = summarize_cio_eval_by_bucket(cio_eval)
    cio_eval.to_csv(root / "tables" / "cio_eval.csv", index=False)
    save_json(root / "tables" / "cio_eval_summary.json", summary)
    save_json(root / "tables" / "cio_eval_by_bucket.json", bucket_summary)
    return cio_eval, cio_results, {"overall": summary, "by_bucket": bucket_summary}


def summarize_cio_eval_by_bucket(cio_eval: pd.DataFrame) -> dict[str, Any]:
    """Summarize CIO accuracy and debate trigger rate by universe bucket."""
    if "bucket" not in cio_eval:
        return {}
    rows: dict[str, Any] = {}
    for bucket, group in cio_eval.groupby("bucket", dropna=False):
        def _acc(col: str) -> float | None:
            if col not in group:
                return None
            evaluable = group[group[col].notna()]
            if evaluable.empty:
                return None
            return float((evaluable[col] == True).mean())

        rows[str(bucket)] = {
            "cases": int(len(group)),
            "debate_trigger_rate": float((group.get("cio_debate_triggered") == True).mean())
            if "cio_debate_triggered" in group
            else None,
            "short_accuracy": _acc("cio_short_direction_match"),
            "long_accuracy": _acc("cio_long_direction_match"),
            "avg_confidence": float(group["cio_confidence"].dropna().mean())
            if "cio_confidence" in group and not group["cio_confidence"].dropna().empty
            else None,
        }
    return rows


def load_cached_packets_by_ticker(cache_root: str | Path = DEFAULT_CACHE_ROOT) -> dict[str, list[dict[str, Any]]]:
    path = Path(cache_root) / "tables" / "packets_by_ticker.json"
    if path.exists():
        return load_json(path)
    packets: dict[str, list[dict[str, Any]]] = {}
    packet_dir = Path(cache_root) / "packets"
    if not packet_dir.exists():
        return packets
    for file in packet_dir.glob("*.json"):
        packets[file.stem.upper()] = load_json(file)
    return packets


def normalize_eval_stance(value: Any) -> str:
    """Normalize stance labels for strategy and baseline evaluation."""
    text = str(value or "").strip().lower()
    if text in {"bull", "bullish", "long", "positive", "buy", "outperform", "up"}:
        return "Bullish"
    if text in {"bear", "bearish", "short", "negative", "sell", "underperform", "down"}:
        return "Bearish"
    return "Neutral"


def stance_to_direction_label(stance: Any) -> str | None:
    normalized = normalize_eval_stance(stance)
    if normalized == "Bullish":
        return "up"
    if normalized == "Bearish":
        return "down"
    return None


def direction_match_label(
    predicted_direction: str | None,
    realized_direction: Any,
    abnormal_return: Any,
    neutral_band: float = 0.02,
) -> bool | None:
    """Score one predicted direction against realized abnormal return versus QQQ."""
    if pd.isna(abnormal_return) or realized_direction is None:
        return None
    abnormal_value = float(abnormal_return)
    if predicted_direction is None:
        return abs(abnormal_value) <= neutral_band
    return predicted_direction == realized_direction


def direction_match_reason(
    predicted_direction: str | None,
    realized_direction: Any,
    abnormal_return: Any,
    neutral_band: float = 0.02,
) -> str:
    if pd.isna(abnormal_return) or realized_direction is None:
        return "missing realized abnormal return"
    abnormal_value = float(abnormal_return)
    if predicted_direction is None:
        if abs(abnormal_value) <= neutral_band:
            return "neutral matched: abnormal return stayed inside neutral band"
        return "neutral missed: abnormal return moved outside neutral band"
    return "matched" if predicted_direction == realized_direction else "missed"


def score_stance_columns(
    df: pd.DataFrame,
    short_stance_col: str,
    long_stance_col: str,
    prefix: str,
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
) -> pd.DataFrame:
    """Add predicted directions and match columns for any stance pair."""
    out = df.copy()
    short_pred_col = f"{prefix}_short_predicted_direction"
    long_pred_col = f"{prefix}_long_predicted_direction"
    short_match_col = f"{prefix}_short_direction_match"
    long_match_col = f"{prefix}_long_direction_match"
    short_reason_col = f"{prefix}_short_direction_match_reason"
    long_reason_col = f"{prefix}_long_direction_match_reason"

    long_abnormal_col = f"abnormal_{long_post_trading_days}d_vs_qqq"
    long_direction_col = f"realized_{long_post_trading_days}d_direction_vs_qqq"

    out[short_pred_col] = out[short_stance_col].apply(stance_to_direction_label)
    out[long_pred_col] = out[long_stance_col].apply(stance_to_direction_label)

    out[short_match_col] = out.apply(
        lambda row: direction_match_label(
            row[short_pred_col],
            row.get("realized_direction_vs_qqq"),
            row.get("abnormal_vs_qqq"),
            neutral_band=neutral_band,
        ),
        axis=1,
    )
    out[short_reason_col] = out.apply(
        lambda row: direction_match_reason(
            row[short_pred_col],
            row.get("realized_direction_vs_qqq"),
            row.get("abnormal_vs_qqq"),
            neutral_band=neutral_band,
        ),
        axis=1,
    )

    out[long_match_col] = out.apply(
        lambda row: direction_match_label(
            row[long_pred_col],
            row.get(long_direction_col),
            row.get(long_abnormal_col),
            neutral_band=neutral_band,
        ),
        axis=1,
    )
    out[long_reason_col] = out.apply(
        lambda row: direction_match_reason(
            row[long_pred_col],
            row.get(long_direction_col),
            row.get(long_abnormal_col),
            neutral_band=neutral_band,
        ),
        axis=1,
    )
    return out


def summarize_strategy_accuracy(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    """Summarize short/long accuracy for a named strategy prefix."""
    def _one(horizon: str) -> dict[str, Any]:
        column = f"{prefix}_{horizon}_direction_match"
        if column not in df:
            return {
                f"{horizon}_evaluable": 0,
                f"{horizon}_matched": 0,
                f"{horizon}_accuracy": None,
            }
        evaluable = df[df[column].notna()]
        matched = int((evaluable[column] == True).sum()) if not evaluable.empty else 0
        total = int(len(evaluable))
        return {
            f"{horizon}_evaluable": total,
            f"{horizon}_matched": matched,
            f"{horizon}_accuracy": matched / total if total else None,
        }

    return {"strategy": prefix, **_one("short"), **_one("long")}


def build_strategy_comparison_table(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    """Build one clean overall comparison table across CIO, agents, and baselines."""
    return pd.DataFrame([summarize_strategy_accuracy(df, prefix) for prefix in prefixes])


def build_strategy_bucket_comparison(
    df: pd.DataFrame,
    prefixes: list[str],
    bucket_col: str = "bucket",
) -> pd.DataFrame:
    """Build a bucket-level comparison table for many strategy prefixes."""
    if bucket_col not in df:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bucket, group in df.groupby(bucket_col, dropna=False):
        row: dict[str, Any] = {"bucket": bucket, "cases": int(len(group))}
        for prefix in prefixes:
            summary = summarize_strategy_accuracy(group, prefix)
            row[f"{prefix}_short_accuracy"] = summary["short_accuracy"]
            row[f"{prefix}_long_accuracy"] = summary["long_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def _momentum_stance(value: Any, deadband: float = 0.0) -> str:
    if pd.isna(value):
        return "Neutral"
    number = float(value)
    if number > deadband:
        return "Bullish"
    if number < -deadband:
        return "Bearish"
    return "Neutral"


def add_simple_baselines(
    cio_eval: pd.DataFrame,
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
    momentum_return_col: str = "post_1d_return",
    momentum_deadband: float = 0.0,
) -> pd.DataFrame:
    """Add always-bullish and earnings-day momentum baselines."""
    out = cio_eval.copy()
    out["always_bullish_short_stance"] = "Bullish"
    out["always_bullish_long_stance"] = "Bullish"
    out = score_stance_columns(
        out,
        short_stance_col="always_bullish_short_stance",
        long_stance_col="always_bullish_long_stance",
        prefix="always_bullish",
        long_post_trading_days=long_post_trading_days,
        neutral_band=neutral_band,
    )

    out["earnings_momentum_short_stance"] = out[momentum_return_col].apply(
        lambda value: _momentum_stance(value, deadband=momentum_deadband)
    )
    out["earnings_momentum_long_stance"] = out["earnings_momentum_short_stance"]
    out = score_stance_columns(
        out,
        short_stance_col="earnings_momentum_short_stance",
        long_stance_col="earnings_momentum_long_stance",
        prefix="earnings_momentum",
        long_post_trading_days=long_post_trading_days,
        neutral_band=neutral_band,
    )
    return out


def add_cio_event_momentum_overlay(
    cio_eval: pd.DataFrame,
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
    momentum_return_col: str = "post_1d_return",
    momentum_deadband: float = 0.0,
) -> pd.DataFrame:
    """Add a transparent post-earnings strategy: short-term momentum, long-term CIO thesis."""
    out = cio_eval.copy()
    if "earnings_momentum_short_stance" not in out:
        out["earnings_momentum_short_stance"] = out[momentum_return_col].apply(
            lambda value: _momentum_stance(value, deadband=momentum_deadband)
        )
    out["cio_event_momentum_short_stance"] = out["earnings_momentum_short_stance"]
    out["cio_event_momentum_long_stance"] = out["cio_long_term_stance"].apply(normalize_eval_stance)

    neutral_long = out["cio_event_momentum_long_stance"] == "Neutral"
    out.loc[neutral_long, "cio_event_momentum_long_stance"] = out.loc[
        neutral_long, "earnings_momentum_short_stance"
    ]
    return score_stance_columns(
        out,
        short_stance_col="cio_event_momentum_short_stance",
        long_stance_col="cio_event_momentum_long_stance",
        prefix="cio_event_momentum",
        long_post_trading_days=long_post_trading_days,
        neutral_band=neutral_band,
    )


def _packet_agent_name(packet: dict[str, Any]) -> str:
    text = str(packet.get("agent_name") or "").strip().lower().replace("&", "and")
    if "technical" in text or "market" in text:
        return "market_technical"
    if "fundamental" in text:
        return "fundamental"
    if "news" in text or "macro" in text:
        return "news_macro"
    return text.replace(" ", "_") or "unknown_agent"


def add_single_agent_baselines(
    cio_eval: pd.DataFrame,
    packets_by_ticker: dict[str, list[dict[str, Any]]],
    agent_names: list[str] | None = None,
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
) -> pd.DataFrame:
    """Score each individual agent as its own baseline against the same realized returns."""
    out = cio_eval.copy()
    agent_names = agent_names or ["news_macro", "market_technical", "fundamental"]

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for ticker, packets in packets_by_ticker.items():
        for packet in packets or []:
            if isinstance(packet, dict):
                lookup[(str(ticker).upper(), _packet_agent_name(packet))] = packet

    for agent_name in agent_names:
        short_col = f"{agent_name}_short_stance"
        long_col = f"{agent_name}_long_stance"
        confidence_col = f"{agent_name}_confidence"
        out[short_col] = out["ticker"].apply(
            lambda ticker: normalize_eval_stance(
                lookup.get((str(ticker).upper(), agent_name), {}).get("short_term_stance")
            )
        )
        out[long_col] = out["ticker"].apply(
            lambda ticker: normalize_eval_stance(
                lookup.get((str(ticker).upper(), agent_name), {}).get("long_term_stance")
                or lookup.get((str(ticker).upper(), agent_name), {}).get("stance")
            )
        )
        out[confidence_col] = out["ticker"].apply(
            lambda ticker: lookup.get((str(ticker).upper(), agent_name), {}).get("confidence")
        )
        out = score_stance_columns(
            out,
            short_stance_col=short_col,
            long_stance_col=long_col,
            prefix=agent_name,
            long_post_trading_days=long_post_trading_days,
            neutral_band=neutral_band,
        )
    return out


def add_deepresearch_baseline(
    cio_eval: pd.DataFrame,
    deepresearch_df: pd.DataFrame,
    ticker_col: str = "ticker",
    short_col: str = "short_term_stance",
    long_col: str = "long_term_stance",
    prefix: str = "deepresearch",
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
) -> pd.DataFrame:
    """Merge externally supplied DeepResearch labels and score them as a baseline."""
    labels = deepresearch_df.copy()
    labels["_ticker_upper"] = labels[ticker_col].astype(str).str.upper()
    label_lookup = labels.set_index("_ticker_upper")
    out = cio_eval.copy()

    out[f"{prefix}_short_stance"] = out["ticker"].apply(
        lambda ticker: normalize_eval_stance(
            label_lookup[short_col].get(str(ticker).upper()) if short_col in label_lookup else None
        )
    )
    out[f"{prefix}_long_stance"] = out["ticker"].apply(
        lambda ticker: normalize_eval_stance(
            label_lookup[long_col].get(str(ticker).upper()) if long_col in label_lookup else None
        )
    )
    return score_stance_columns(
        out,
        short_stance_col=f"{prefix}_short_stance",
        long_stance_col=f"{prefix}_long_stance",
        prefix=prefix,
        long_post_trading_days=long_post_trading_days,
        neutral_band=neutral_band,
    )


def write_deepresearch_label_template(
    cio_eval: pd.DataFrame,
    path: str | Path | None = None,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> Path:
    """Write a CSV template for manually entered DeepResearch stance labels."""
    output_path = Path(path) if path is not None else Path(cache_root) / "tables" / "deepresearch_labels.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["ticker", "company", "earnings_date"]
    available_columns = [column for column in columns if column in cio_eval]
    template = cio_eval[available_columns].copy()
    template["short_term_stance"] = ""
    template["long_term_stance"] = ""
    template["deepresearch_rationale"] = ""
    template.to_csv(output_path, index=False)
    return output_path
