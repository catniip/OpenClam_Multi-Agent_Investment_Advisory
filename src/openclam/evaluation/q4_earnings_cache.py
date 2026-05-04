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
