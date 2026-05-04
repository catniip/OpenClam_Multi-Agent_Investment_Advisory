from __future__ import annotations

import json
import os
import re
import statistics
import contextlib
import io
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from urllib.parse import quote_plus


@dataclass
class NewsItem:
    title: str
    source: str = "unknown"
    published_at: str | None = None
    url: str | None = None
    summary: str | None = None


@dataclass
class MacroSignal:
    name: str
    value: str
    interpretation: str
    source: str = "computed"


@dataclass
class NewsMacroContext:
    ticker: str
    company: str
    event_date: str | None
    lookback_days: int
    news: list[NewsItem] = field(default_factory=list)
    macro_signals: list[MacroSignal] = field(default_factory=list)
    data_notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return format_context(self)


@dataclass
class InvestmentOpportunity:
    ticker: str
    company_name: str
    position: str
    trigger_logic: str
    demand_chain_thesis: str
    impact_score: int
    impact_assessment: str


@dataclass
class MainstreamStockRead:
    ticker: str
    company_name: str
    relevance: str
    market_reaction: str
    preliminary_view: str


@dataclass
class NewsMacroReport:
    ticker: str | None
    company: str | None
    news_summary: str
    core_insight: str
    short_term_stance: str
    long_term_stance: str
    stance_rationale: str
    mainstream_stocks: list[MainstreamStockRead]
    opportunities: list[InvestmentOpportunity]
    risk_notes: list[str]
    citations: list[dict[str, str | None]]
    raw_context: dict[str, Any]


POSITIVE_TERMS = {
    "beat",
    "beats",
    "growth",
    "upgrade",
    "raises",
    "raised",
    "strong",
    "record",
    "profit",
    "profitable",
    "approval",
    "partnership",
    "expands",
    "launch",
    "surge",
}

NEGATIVE_TERMS = {
    "miss",
    "misses",
    "cut",
    "cuts",
    "weak",
    "lawsuit",
    "probe",
    "investigation",
    "downgrade",
    "decline",
    "falls",
    "layoff",
    "recall",
    "warning",
    "risk",
    "slowdown",
}


RELATED_TICKER_UNIVERSE = {
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "TSLA", "QCOM", "AVGO"],
    "MSFT": ["AMZN", "GOOGL", "META", "NVDA", "ORCL", "CRM", "AVGO"],
    "AMZN": ["MSFT", "GOOGL", "META", "WMT", "SHOP", "ORCL", "NVDA"],
    "GOOG": ["MSFT", "AMZN", "META", "NVDA", "AVGO", "ORCL", "AAPL"],
    "GOOGL": ["MSFT", "AMZN", "META", "NVDA", "AVGO", "ORCL", "AAPL"],
    "META": ["GOOGL", "SNAP", "PINS", "MSFT", "AMZN", "NVDA", "AVGO"],
    "NVDA": ["AMD", "AVGO", "TSM", "ASML", "MU", "MSFT", "AMZN", "GOOGL"],
    "TSLA": ["GM", "F", "RIVN", "LCID", "ALB", "PANW", "NVDA"],
}


RELATED_COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOG": "Alphabet",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "NVDA": "Nvidia",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "TSM": "TSMC",
    "ASML": "ASML",
    "MU": "Micron",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "WMT": "Walmart",
    "SHOP": "Shopify",
    "SNAP": "Snap",
    "PINS": "Pinterest",
    "TSLA": "Tesla",
    "GM": "General Motors",
    "F": "Ford",
    "RIVN": "Rivian",
    "LCID": "Lucid",
    "ALB": "Albemarle",
    "QCOM": "Qualcomm",
    "PANW": "Palo Alto Networks",
}


MAG7_Q4_2025_EARNINGS = {
    "TSLA": ("Tesla", "2026-01-28"),
    "META": ("Meta Platforms", "2026-01-28"),
    "AAPL": ("Apple", "2026-01-29"),
    "MSFT": ("Microsoft", "2026-01-28"),
    "GOOGL": ("Alphabet", "2026-02-03"),
    "AMZN": ("Amazon", "2026-02-05"),
    "NVDA": ("Nvidia", "2026-02-25"),
}


def fetch_yfinance_news(ticker: str, max_items: int = 10) -> tuple[list[NewsItem], list[str]]:
    """Fetch recent Yahoo Finance news through yfinance when it is installed."""
    notes: list[str] = []
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return [], ["yfinance is not installed; skipped Yahoo Finance news."]

    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as exc:
        return [], [f"Yahoo Finance news fetch failed: {exc}"]

    items: list[NewsItem] = []
    for raw in raw_items[:max_items]:
        content = raw.get("content", raw)
        published = content.get("pubDate") or content.get("displayTime")
        provider = content.get("provider") or {}
        canonical_url = content.get("canonicalUrl") or {}
        click_url = content.get("clickThroughUrl") or {}
        items.append(
            NewsItem(
                title=content.get("title") or raw.get("title") or "",
                source=provider.get("displayName") or raw.get("publisher") or "Yahoo Finance",
                published_at=published,
                url=canonical_url.get("url") or click_url.get("url") or raw.get("link"),
                summary=content.get("summary") or raw.get("summary"),
            )
        )
    return [item for item in items if item.title], notes


def fetch_newsapi_news(
    company: str,
    ticker: str | None = None,
    api_key: str | None = None,
    lookback_days: int = 14,
    max_items: int = 10,
    event_date: str | None = None,
    news_mode: str = "latest",
    related_tickers: list[str] | None = None,
    news_end_offset_days: int = 1,
) -> tuple[list[NewsItem], list[str]]:
    """Fetch news from NewsAPI if NEWSAPI_KEY is available."""
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    if not api_key:
        return [], ["NEWSAPI_KEY is not set; skipped NewsAPI."]

    try:
        import requests  # type: ignore
    except Exception:
        return [], ["requests is not installed; skipped NewsAPI."]

    if news_mode not in {"latest", "event_window"}:
        raise ValueError("news_mode must be one of: 'latest' or 'event_window'.")

    anchor_date = _parse_date(event_date) if news_mode == "event_window" else date.today()
    anchor_date = anchor_date or date.today()
    from_date = (anchor_date - timedelta(days=lookback_days)).isoformat()
    to_date = (anchor_date + timedelta(days=news_end_offset_days)).isoformat()
    queries = _build_newsapi_queries(company, ticker, related_tickers=related_tickers)
    articles: list[tuple[str, dict[str, Any]]] = []
    notes = [f"NewsAPI queried {len(queries)} finance-focused query path(s) from {from_date} through {to_date}."]
    for label, query in queries:
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": max_items,
        }
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                headers={"X-Api-Key": api_key},
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("articles", [])
            articles.extend((label, article) for article in batch)
            notes.append(f"NewsAPI {label} query returned {len(batch)} article(s).")
        except Exception as exc:
            notes.append(f"NewsAPI {label} query failed: {exc}")

    items = [
        NewsItem(
            title=article.get("title") or "",
            source=f"{(article.get('source') or {}).get('name') or 'NewsAPI'} / {label}",
            published_at=article.get("publishedAt"),
            url=article.get("url"),
            summary=article.get("description"),
        )
        for label, article in articles
        if article.get("title")
    ]
    filtered = [item for item in items if _is_finance_relevant_news(item, ticker or "", company)]
    if items and not filtered:
        notes.append("NewsAPI returned articles, but all were filtered as low-relevance/non-financial noise.")
    elif len(filtered) < len(items):
        notes.append(f"Filtered out {len(items) - len(filtered)} low-relevance NewsAPI article(s).")
    return filtered, notes


def fetch_finnhub_company_news(
    ticker: str,
    api_key: str | None = None,
    lookback_days: int = 14,
    max_items: int = 10,
    event_date: str | None = None,
    news_mode: str = "latest",
    news_end_offset_days: int = 1,
) -> tuple[list[NewsItem], list[str]]:
    """Fetch company news from Finnhub's ticker-based company-news endpoint."""
    api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return [], ["FINNHUB_API_KEY is not set; skipped Finnhub company news."]

    try:
        import requests  # type: ignore
    except Exception:
        return [], ["requests is not installed; skipped Finnhub company news."]

    if news_mode not in {"latest", "event_window"}:
        raise ValueError("news_mode must be one of: 'latest' or 'event_window'.")

    anchor_date = _parse_date(event_date) if news_mode == "event_window" else date.today()
    anchor_date = anchor_date or date.today()
    from_date = (anchor_date - timedelta(days=lookback_days)).isoformat()
    to_date = (anchor_date + timedelta(days=news_end_offset_days)).isoformat()
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        articles = response.json()
    except Exception as exc:
        return [], [f"Finnhub company-news fetch failed: {exc}"]

    if not isinstance(articles, list):
        return [], ["Finnhub company-news returned an unexpected response shape."]

    items = []
    for article in articles[: max_items * 3]:
        published_at = article.get("datetime")
        if isinstance(published_at, (int, float)):
            published_at = datetime.utcfromtimestamp(published_at).isoformat() + "Z"
        items.append(
            NewsItem(
                title=article.get("headline") or "",
                source=article.get("source") or "Finnhub",
                published_at=published_at,
                url=article.get("url"),
                summary=article.get("summary"),
            )
        )
    return [item for item in items if item.title], [
        f"Fetched Finnhub company news for {ticker.upper()} from {from_date} through {to_date}."
    ]


def fetch_macro_proxies(
    ticker: str,
    event_date: str | None = None,
    lookback_days: int = 30,
    news_mode: str = "latest",
) -> tuple[list[MacroSignal], list[str]]:
    """Use market proxies as a lightweight macro read when yfinance is available."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return [], ["yfinance is not installed; skipped macro proxy fetch."]

    symbols = {
        "SPY": "S&P 500 proxy",
        "^VIX": "Equity volatility proxy",
        "^TNX": "10-year Treasury yield proxy",
    }
    signals: list[MacroSignal] = []
    notes: list[str] = []
    anchor_date = _parse_date(event_date) if news_mode == "event_window" else None
    start_date = anchor_date - timedelta(days=lookback_days) if anchor_date else None
    end_date = anchor_date + timedelta(days=1) if anchor_date else None

    for symbol, label in symbols.items():
        try:
            if start_date and end_date:
                hist = yf.download(
                    symbol,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                window_label = f"{lookback_days}-day move into {anchor_date.isoformat()}"
            else:
                hist = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
                window_label = "1-month move"
            closes = _extract_close(hist)
            if len(closes) < 2:
                notes.append(f"Not enough data for {symbol}.")
                continue
            start = _series_scalar(closes, 0)
            end = _series_scalar(closes, -1)
            pct = (end / start - 1) * 100
            direction = "supportive" if (symbol == "SPY" and pct >= 0) or (symbol == "^VIX" and pct <= 0) else "headwind"
            if symbol == "^TNX":
                direction = "higher-rate headwind" if pct > 0 else "lower-rate support"
            signals.append(
                MacroSignal(
                    name=label,
                    value=f"{pct:.2f}% {window_label}",
                    interpretation=f"{direction} for risk appetite around {ticker}",
                    source=f"yfinance:{symbol}",
                )
            )
        except Exception as exc:
            notes.append(f"Macro proxy fetch failed for {symbol}: {exc}")

    return signals, notes


def sample_context(ticker: str, company: str, event_date: str | None = None) -> NewsMacroContext:
    """Offline demo data so the notebook always runs before API keys are configured."""
    return NewsMacroContext(
        ticker=ticker,
        company=company,
        event_date=event_date,
        lookback_days=14,
        news=[
            NewsItem(
                title=f"{company} shares move as investors parse earnings outlook and AI spending plans",
                source="Sample News",
                published_at=event_date,
                summary="Investors focused on guidance quality, margin impact, and whether demand trends justify higher spending.",
            ),
            NewsItem(
                title=f"Analysts debate whether {company}'s recent growth can offset macro uncertainty",
                source="Sample News",
                published_at=event_date,
                summary="Commentary is mixed, with bullish views on revenue durability and cautious views on valuation.",
            ),
            NewsItem(
                title="Broader market sentiment softens as Treasury yields rise",
                source="Sample Macro",
                published_at=event_date,
                summary="Higher rates can pressure equity multiples, especially for long-duration growth stocks.",
            ),
        ],
        macro_signals=[
            MacroSignal("S&P 500 proxy", "-1.20% 1-month move", "headwind for broad risk appetite", "sample"),
            MacroSignal("10-year Treasury yield proxy", "+4.80% 1-month move", "higher-rate headwind", "sample"),
            MacroSignal("Equity volatility proxy", "+7.40% 1-month move", "risk-off signal", "sample"),
        ],
        data_notes=["Using offline sample context. Install yfinance and/or set NEWSAPI_KEY for live data."],
    )


def collect_context(
    ticker: str,
    company: str,
    event_date: str | None = None,
    lookback_days: int = 14,
    max_news: int = 10,
    use_sample_if_empty: bool = True,
    news_mode: str = "latest",
    news_sources: list[str] | None = None,
    news_end_offset_days: int = 1,
) -> NewsMacroContext:
    if news_mode not in {"latest", "event_window"}:
        raise ValueError("news_mode must be one of: 'latest' or 'event_window'.")

    news_sources = news_sources or ["finnhub", "newsapi", "yfinance"]
    invalid_sources = sorted(set(news_sources) - {"finnhub", "newsapi", "yfinance"})
    if invalid_sources:
        raise ValueError(f"Unsupported news source(s): {invalid_sources}. Use any of: finnhub, newsapi, yfinance.")

    news: list[NewsItem] = []
    macro_signals: list[MacroSignal] = []
    notes: list[str] = []
    related_tickers = _related_tickers_for(ticker)
    if related_tickers:
        notes.append(
            "Related-company news enabled for industry-chain context: "
            + ", ".join(related_tickers)
            + "."
        )

    if news_mode == "latest":
        notes.append("Using latest-news mode; event_date is kept as metadata but not used to filter news.")
    else:
        notes.append(
            f"Using event-window news ending {news_end_offset_days:+d} calendar day(s) from event_date."
        )

    if "finnhub" in news_sources:
        finnhub_news, finnhub_notes = fetch_finnhub_company_news(
            ticker,
            lookback_days=lookback_days,
            max_items=max_news,
            event_date=event_date,
            news_mode=news_mode,
            news_end_offset_days=news_end_offset_days,
        )
        news.extend(finnhub_news)
        notes.extend(finnhub_notes)
        related_item_budget = max(2, max_news // 3)
        for related_ticker in related_tickers[:5]:
            related_news, related_notes = fetch_finnhub_company_news(
                related_ticker,
                lookback_days=lookback_days,
                max_items=related_item_budget,
                event_date=event_date,
                news_mode=news_mode,
                news_end_offset_days=news_end_offset_days,
            )
            for item in related_news:
                item.source = f"{item.source} / related:{related_ticker}"
                if item.summary:
                    item.summary = f"Related-company context for {ticker.upper()} via {related_ticker}: {item.summary}"
            news.extend(related_news)
            notes.extend(related_notes)

    if "newsapi" in news_sources:
        api_news, api_notes = fetch_newsapi_news(
            company,
            ticker=ticker,
            lookback_days=lookback_days,
            max_items=max_news,
            event_date=event_date,
            news_mode=news_mode,
            related_tickers=related_tickers,
            news_end_offset_days=news_end_offset_days,
        )
        news.extend(api_news)
        notes.extend(api_notes)

    if "yfinance" in news_sources:
        yf_news, yf_notes = fetch_yfinance_news(ticker, max_news * 3)
        filter_notes: list[str] = []
        if news_mode == "event_window":
            yf_news, filter_notes = _filter_news_by_event_window(
                yf_news,
                event_date,
                lookback_days,
                news_end_offset_days=news_end_offset_days,
            )
        else:
            yf_news = yf_news[:max_news]
        news.extend(yf_news)
        notes.extend(yf_notes + filter_notes)

    macro, macro_notes = fetch_macro_proxies(
        ticker,
        event_date=event_date,
        lookback_days=lookback_days,
        news_mode=news_mode,
    )

    news = _rank_news_by_relevance(
        _dedupe_news(news),
        ticker,
        company,
        related_tickers=related_tickers,
    )[:max_news]
    macro_signals.extend(macro)
    notes.extend(macro_notes)

    if use_sample_if_empty and not news and not macro_signals:
        sample = sample_context(ticker, company, event_date)
        return sample

    return NewsMacroContext(
        ticker=ticker.upper(),
        company=company,
        event_date=event_date,
        lookback_days=lookback_days,
        news=news,
        macro_signals=macro_signals,
        data_notes=notes,
    )


def generate_report(
    context: NewsMacroContext,
    model: str = "gpt-5.4-nano",
    gemini_model: str = "gemini-2.5-flash",
    provider: str = "auto",
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
) -> NewsMacroReport:
    """Generate a structured report with Vertex AI, OpenAI, Gemini API, or heuristic fallback."""
    provider = provider.lower()
    if provider not in {"auto", "openai", "gemini", "vertex"}:
        raise ValueError("provider must be one of: 'auto', 'openai', 'gemini', or 'vertex'.")

    vertex_project = vertex_project or os.getenv("VERTEX_PROJECT")
    vertex_location = vertex_location or os.getenv("VERTEX_LOCATION", "us-central1")

    if provider == "vertex" or (provider == "auto" and vertex_project):
        try:
            return _generate_vertex_report(
                context,
                model=gemini_model,
                project=vertex_project,
                location=vertex_location,
            )
        except Exception as exc:
            context.data_notes.append(f"Vertex AI report generation failed: {exc}")
            if provider == "vertex":
                context.data_notes.append("Provider was forced to Vertex AI; used heuristic fallback.")
                return _generate_heuristic_report(context)

    if provider in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
        try:
            return _generate_llm_report(context, model=model)
        except Exception as exc:
            context.data_notes.append(f"OpenAI report generation failed: {exc}")
            if provider == "openai":
                context.data_notes.append("Provider was forced to OpenAI; used heuristic fallback.")
                return _generate_heuristic_report(context)

    if provider in {"auto", "gemini"} and os.getenv("GEMINI_API_KEY"):
        try:
            return _generate_gemini_report(context, model=gemini_model)
        except Exception as exc:
            context.data_notes.append(f"Gemini report generation failed: {exc}")
            if provider == "gemini":
                context.data_notes.append("Provider was forced to Gemini; used heuristic fallback.")

    return _generate_heuristic_report(context)


def analyze_news_text(
    news_text: str,
    ticker: str | None = None,
    company: str | None = None,
    source: str = "user-provided news",
    event_date: str | None = None,
    model: str = "gpt-5.4-nano",
    gemini_model: str = "gemini-2.5-flash",
    provider: str = "auto",
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
) -> NewsMacroReport:
    """
    Analyze pasted news text directly, which is convenient for Colab demos.

    provider options:
    - "auto": use Vertex AI if VERTEX_PROJECT exists; otherwise OpenAI if OPENAI_API_KEY exists; otherwise Gemini if GEMINI_API_KEY exists; otherwise heuristic fallback.
    - "openai": use OpenAI only; if it fails, use heuristic fallback.
    - "gemini": use Gemini only; if it fails, use heuristic fallback.
    - "vertex": use Vertex AI Gemini through the authenticated Google Cloud account; if it fails, use heuristic fallback.
    """
    first_line = next((line.strip() for line in news_text.splitlines() if line.strip()), "")
    context = NewsMacroContext(
        ticker=(ticker or "N/A").upper(),
        company=company or "User-provided event",
        event_date=event_date,
        lookback_days=0,
        news=[
            NewsItem(
                title=first_line[:180] or "User-provided news event",
                source=source,
                published_at=event_date,
                summary=news_text.strip(),
            )
        ],
        macro_signals=[],
        data_notes=["Context was provided directly by the user, not fetched from a news API."],
    )
    return generate_report(
        context,
        model=model,
        gemini_model=gemini_model,
        provider=provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )


def _generate_llm_report(context: NewsMacroContext, model: str) -> NewsMacroReport:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    prompt = _build_event_driven_prompt(context)
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    payload = _extract_json(response.output_text)
    return _report_from_payload(context, payload)


def _generate_gemini_report(context: NewsMacroContext, model: str) -> NewsMacroReport:
    prompt = _build_event_driven_prompt(context)
    try:
        from google import genai  # type: ignore
    except ImportError:
        raise ImportError("Install google-genai to use provider='gemini': pip install google-genai")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None) or str(response)

    payload = _extract_json(text)
    return _report_from_payload(context, payload)


def _generate_vertex_report(
    context: NewsMacroContext,
    model: str,
    project: str | None,
    location: str,
) -> NewsMacroReport:
    if not project:
        raise ValueError("vertex_project is required when provider='vertex'.")

    from google import genai  # type: ignore

    client = genai.Client(vertexai=True, project=project, location=location)
    prompt = _build_event_driven_prompt(context)
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None) or str(response)
    payload = _extract_json(text)
    return _report_from_payload(context, payload)


def _build_event_driven_prompt(context: NewsMacroContext) -> str:
    schema_hint = {
        "news_summary": "one sentence summarizing the news",
        "core_insight": "one sentence identifying the supply-chain second-order blind spot mainstream investors may miss",
        "short_term_stance": "Bullish / Neutral / Bearish for the main ticker over the next 1-10 trading days",
        "long_term_stance": "Bullish / Neutral / Bearish for the main ticker over the next 1-3 quarters",
        "stance_rationale": "brief explanation connecting the news and macro context to the short-term and long-term stance",
        "mainstream_stocks": [
            {
                "ticker": "large-cap US stock ticker",
                "company_name": "company name",
                "relevance": "why mainstream capital would look at this stock first",
                "market_reaction": "likely first-order market reaction",
                "preliminary_view": "brief view as a liquidity/consensus anchor, not the final second-order idea",
            }
        ],
        "opportunities": [
            {
                "ticker": "NYSE/NASDAQ/US-listed ADR ticker",
                "company_name": "company name",
                "position": "Long or Short",
                "trigger_logic": "Competitor Displacement / Substitution Effect / Solution Provider / Pick-and-Shovel / Cost Squeeze & Victims",
                "demand_chain_thesis": "Use this causal structure: Because A happens, B becomes scarce / demand rises, which causes C company's orders, revenue, or margins to change.",
                "impact_score": "integer 1-10",
                "impact_assessment": "specific explanation of the expected earnings impact over the next 1-3 quarters",
            }
        ],
        "risk_notes": ["key assumptions, counter-evidence, or data points that need verification"],
    }
    return f"""
# Role
You are a Wall Street event-driven hedge fund manager with 20 years of experience.
You are highly rational and skilled at divergent thinking and second-order effects analysis.
Your edge is finding overlooked US stock market opportunities hidden behind ordinary-looking news.
You do not care about surface-level public sentiment. You care about capital flows, supply-chain causality, incentives, bottlenecks, and industry-chain knock-on effects.

# Core Task
Read the supplied news and macro context. The main ticker in the context is the primary object of analysis. Some news items may come from related companies in the same industry chain; use those related-company items as cross-checks, competitive signals, demand-chain evidence, or second-order clues, but do not let them replace the main-ticker analysis.
First analyze the main ticker directly, then identify the mainstream large-cap stocks that investors are most likely to watch immediately, and finally map the full industry chain to find second-order US equity opportunities.
Your analysis order must be:

Step 1: Main-ticker event read.
Summarize what the news window implies for the main ticker's revenue growth, margins, capex intensity, competitive position, valuation sensitivity, and investor expectations. Separate direct evidence about the main ticker from inference based on related-company news.

Step 2: Mainstream large-stock screen.
List 3-5 directly related, highly liquid, large-cap US-listed stocks or ADRs that mainstream capital would likely react to first. This step establishes consensus, liquidity anchors, and first-order market reaction. It is not the main recommendation section.

Step 3: Industry-chain second-order inference.
Starting from the main-ticker read and the mainstream reaction, expand into upstream suppliers, critical components, substitute products, competitors, solution providers, downstream margin victims, logistics, insurance, infrastructure, or other knock-on beneficiaries/victims. Output 3-5 overlooked second-order US-listed opportunities, including both long and short ideas when appropriate. Quantify the potential earnings impact.

Step 4: Main-ticker stance.
Give a short-term stance and a long-term stance for the main ticker in the supplied context.
- short_term_stance: expected abnormal direction versus QQQ over the next 1-10 trading days after the event/news window.
- long_term_stance: expected abnormal direction versus QQQ over the next 1-3 quarters.
Each stance must be exactly one of: Bullish, Neutral, Bearish.
Avoid Neutral unless the expected abnormal move is likely to stay within roughly +/-2%. If short-term valuation compression, capex concern, margin pressure, weak guidance, demand slowdown, or investor disappointment dominates, choose Bearish even if long-term fundamentals remain strong. If near-term earnings surprise, raised guidance, accelerating demand, margin expansion, or positive capital-flow surprise dominates, choose Bullish. Short-term stance should reflect expected post-event stock reaction versus QQQ, not the company's long-term business quality.

# Second-Order Framework
1. Competitor Displacement: If a leader is impaired by bankruptcy, scandal, capacity constraints, or regulation, who can take share?
2. Substitution Effect: If product A becomes unavailable, expensive, or risky, what product B will customers or enterprises adopt instead?
3. Solution Provider: After the event, who provides cleanup, protection, remediation, compliance, rebuilding, or tools?
4. Pick-and-Shovel: If a trend accelerates, who sells the infrastructure, components, data, equipment, or critical inputs behind it?
5. Cost Squeeze & Victims: Does the event raise input costs, break supply chains, reduce availability, or pressure margins for downstream companies?

# Constraints
- All output must be in English.
- Recommend only US-listed companies or ADRs on NYSE/NASDAQ where possible.
- The main ticker is the center of the report. Related-company news is supporting evidence, not the headline conclusion.
- In the news_summary and core_insight, explicitly mention whether the strongest evidence is direct main-ticker news or related-company read-through.
- The mainstream screen may include the news protagonist and obvious mega-cap stocks.
- The second-order opportunity section should not simply repeat the news protagonist unless it is a short idea.
- Each second-order idea must explicitly identify where the company sits in the industry chain.
- Provide exact company names and tickers.
- Give an impact score from 1-10 based on likely earnings impact over the next 1-3 quarters if the event continues to develop.
- Provide short_term_stance and long_term_stance for the main ticker, not just for second-order ideas.
- Do not invent facts beyond the supplied context. If the thesis relies on inference, state that assumption in risk_notes.
- This is research support, not personalized investment advice.

Return strict JSON only. Do not return markdown. JSON shape:
{json.dumps(schema_hint, indent=2)}

Context:
{json.dumps(asdict(context), indent=2)}
"""


def _report_from_payload(context: NewsMacroContext, payload: dict[str, Any]) -> NewsMacroReport:
    citations = [{"title": item.title, "source": item.source, "url": item.url} for item in context.news]
    return NewsMacroReport(
        ticker=context.ticker if context.ticker != "N/A" else None,
        company=context.company,
        news_summary=str(payload["news_summary"]),
        core_insight=str(payload["core_insight"]),
        short_term_stance=_normalize_stance(payload.get("short_term_stance")),
        long_term_stance=_normalize_stance(payload.get("long_term_stance")),
        stance_rationale=str(payload.get("stance_rationale", "No stance rationale provided.")),
        mainstream_stocks=[
            MainstreamStockRead(
                ticker=str(item["ticker"]).upper(),
                company_name=str(item["company_name"]),
                relevance=str(item["relevance"]),
                market_reaction=str(item["market_reaction"]),
                preliminary_view=str(item["preliminary_view"]),
            )
            for item in payload.get("mainstream_stocks", [])[:5]
        ],
        opportunities=[
            InvestmentOpportunity(
                ticker=str(item["ticker"]).upper(),
                company_name=str(item["company_name"]),
                position=str(item["position"]),
                trigger_logic=str(item["trigger_logic"]),
                demand_chain_thesis=str(item["demand_chain_thesis"]),
                impact_score=int(item["impact_score"]),
                impact_assessment=str(item["impact_assessment"]),
            )
            for item in payload["opportunities"][:5]
        ],
        risk_notes=list(payload.get("risk_notes", [])),
        citations=citations,
        raw_context=asdict(context),
    )


def _generate_heuristic_report(context: NewsMacroContext) -> NewsMacroReport:
    texts = [f"{item.title}. {item.summary or ''}" for item in context.news]
    score = _sentiment_score(texts)
    joined_text = " ".join(texts).lower()
    has_ai = any(term in joined_text for term in ["ai", "artificial intelligence", "semiconductor", "data center", "cloud"])
    has_rates = any(term in joined_text for term in ["yield", "rate", "treasury", "inflation"])

    if has_ai:
        mainstream_stocks = [
            MainstreamStockRead(
                "MSFT",
                "Microsoft Corporation",
                "Cloud and AI capital spending are likely the first anchor mainstream capital will watch.",
                "If the news reinforces the AI investment cycle, investors may first reprice mega-cap cloud growth and margin pressure.",
                "Use it as a first-order capital-flow anchor, not the only second-order opportunity.",
            ),
            MainstreamStockRead(
                "AMZN",
                "Amazon.com, Inc.",
                "AWS is a core large-cap proxy for cloud and AI infrastructure demand.",
                "The market will watch whether AI demand converts into AWS growth and returns on capital spending.",
                "Use AWS demand as the starting point, then expand into the infrastructure supply chain.",
            ),
            MainstreamStockRead(
                "GOOGL",
                "Alphabet Inc.",
                "Google Cloud and internal AI chips make Alphabet a mainstream AI trade anchor.",
                "The market may price both AI growth and free-cash-flow pressure from capital spending.",
                "Useful for judging whether capital prefers platform stocks or supply-chain beneficiaries.",
            ),
        ]
        opportunities = [
            InvestmentOpportunity(
                "NVDA",
                "NVIDIA Corporation",
                "Long",
                "Pick-and-Shovel",
                "Because AI and data-center spending keeps rising, demand for accelerated computing chips and software stacks increases, which supports NVIDIA GPU and platform orders.",
                8,
                "If the event reinforces AI capex expectations, data-center revenue and margins could benefit over the next 1-3 quarters.",
            ),
            InvestmentOpportunity(
                "AVGO",
                "Broadcom Inc.",
                "Long",
                "Pick-and-Shovel",
                "Because hyperscalers expand AI clusters, demand for high-speed networking chips and custom ASICs increases, which benefits Broadcom semiconductor orders.",
                7,
                "A higher mix of AI networking and custom silicon could improve revenue quality and margins.",
            ),
            InvestmentOpportunity(
                "DELL",
                "Dell Technologies Inc.",
                "Long",
                "Solution Provider",
                "Because enterprises need to deploy AI into servers and private-cloud environments, demand for AI servers and infrastructure rises, which can lift Dell infrastructure orders.",
                6,
                "The earnings impact depends on whether AI server gross margins can offset hardware competition.",
            ),
        ]
    else:
        mainstream_stocks = [
            MainstreamStockRead(
                context.ticker if context.ticker != "N/A" else "SPY",
                context.company if context.company != "User-provided event" else "SPDR S&P 500 ETF Trust",
                "This is the most direct event-linked ticker or a broad market beta proxy.",
                "The market may first express first-order risk appetite through the index, sector leader, or news protagonist.",
                "Use it as a capital-flow anchor before searching for second-order industry-chain winners or victims.",
            )
        ]
        opportunities = [
            InvestmentOpportunity(
                "SPY",
                "SPDR S&P 500 ETF Trust",
                "Short/Hedge",
                "Cost Squeeze & Victims",
                "Because macro uncertainty rises, discount-rate pressure and risk-asset outflows can increase, which makes a broad-market ETF a portfolio-level hedge candidate.",
                5,
                "This is a broad hedge rather than a single-company earnings thesis, useful when detailed industry-chain data is unavailable.",
            ),
            InvestmentOpportunity(
                "VIXY",
                "ProShares VIX Short-Term Futures ETF",
                "Long/Hedge",
                "Solution Provider",
                "Because the event may amplify market volatility, demand for short-term volatility protection can rise, which can support VIX futures-linked products.",
                4,
                "This vehicle is heavily affected by futures roll costs and should be treated only as an event-window observation tool.",
            ),
            InvestmentOpportunity(
                "XLF",
                "Financial Select Sector SPDR Fund",
                "Short/Hedge",
                "Cost Squeeze & Victims",
                "Because rate or credit-risk shifts can compress financial-sector risk appetite, investors may reduce financial beta, which can pressure financial ETFs.",
                4,
                "The earnings impact is indirect and requires confirmation through net interest margins, credit costs, or capital-markets revenue.",
            ),
        ]

    return NewsMacroReport(
        ticker=context.ticker if context.ticker != "N/A" else None,
        company=context.company,
        news_summary=(
            f"News related to {context.company} suggests the market is reassessing demand, costs, and risk appetite."
        ),
        core_insight=(
            f"The heuristic sentiment score is {score:.2f}; start with mainstream large-stock capital-flow anchors, then search the industry chain for second-order beneficiaries, margin victims, and infrastructure suppliers."
        ),
        short_term_stance=_heuristic_stance(score, has_ai, has_rates, horizon="short"),
        long_term_stance=_heuristic_stance(score, has_ai, has_rates, horizon="long"),
        stance_rationale=(
            "Heuristic fallback stance based on simple news sentiment keywords, AI/cloud context, and rate-related macro terms. "
            "Use model-backed output for real evaluation."
        ),
        mainstream_stocks=mainstream_stocks,
        opportunities=opportunities,
        risk_notes=[
            "This is the heuristic fallback used when no OpenAI or Gemini API call succeeds; ideas are for notebook prototyping only.",
            "For higher-quality second-order reasoning, set OPENAI_API_KEY or GEMINI_API_KEY and provide the full news text.",
            *context.data_notes,
        ],
        citations=[{"title": item.title, "source": item.source, "url": item.url} for item in context.news],
        raw_context=asdict(context),
    )


def display_report(report: NewsMacroReport) -> None:
    print(format_report_markdown(report))


def display_context(context: NewsMacroContext) -> None:
    print(format_context(context))


def format_context(context: NewsMacroContext) -> str:
    lines = [
        f"NewsMacroContext: {context.company} ({context.ticker})",
        f"Event date: {context.event_date or 'latest'} | Lookback days: {context.lookback_days}",
        "",
        "News items, ranked by relevance:",
    ]
    if context.news:
        for idx, item in enumerate(context.news, start=1):
            lines.extend(
                [
                    f"{idx}. {item.title}",
                    f"   Source: {item.source} | Published: {item.published_at or 'unknown'}",
                    f"   Summary: {_clean_oneline(item.summary) or 'No summary available.'}",
                ]
            )
            if item.url:
                lines.append(f"   URL: {item.url}")
    else:
        lines.append("No news items found.")

    lines.extend(["", "Macro signals:"])
    if context.macro_signals:
        for signal in context.macro_signals:
            lines.append(f"- {signal.name}: {signal.value}; {signal.interpretation} ({signal.source})")
    else:
        lines.append("No macro signals found.")

    if context.data_notes:
        lines.extend(["", "Data notes:"])
        for note in context.data_notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


def format_report_markdown(report: NewsMacroReport) -> str:
    lines = [
        "**📰 News Summary & Core Insight**:",
        f"{report.news_summary} {report.core_insight}",
        "",
        "**🧭 Main-Ticker Stance**:",
        f"- Short-term stance: {report.short_term_stance}",
        f"- Long-term stance: {report.long_term_stance}",
        f"- Rationale: {report.stance_rationale}",
        "",
        "**🏦 Mainstream Large-Stock Screen**:",
        "",
    ]
    for idx, item in enumerate(report.mainstream_stocks, start=1):
        lines.extend(
            [
                f"{idx}. [{item.ticker}] {item.company_name}",
                f"- Relevance: {item.relevance}",
                f"- First-order market reaction: {item.market_reaction}",
                f"- Preliminary view: {item.preliminary_view}",
                "",
            ]
        )
    lines.extend(
        [
        "**📈 Second-Order Investment Opportunity Analysis**:",
        "",
        ]
    )
    for idx, item in enumerate(report.opportunities, start=1):
        lines.extend(
            [
                f"{idx}. [{item.ticker}] {item.company_name} ({item.position})",
                f"- Trigger logic: {item.trigger_logic}",
                f"- Demand-chain thesis: {item.demand_chain_thesis}",
                f"- Earnings impact assessment: [Relevance & earnings impact score: {item.impact_score}/10] - {item.impact_assessment}",
                "",
            ]
        )
    if report.risk_notes:
        lines.append("**⚠️ Key Assumptions & Risks**:")
        for note in report.risk_notes:
            lines.append(f"- {note}")
    return "\n".join(lines).strip()


def save_report(report: NewsMacroReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=2, ensure_ascii=False)


def mag7_q4_2025_earnings_df():
    """Hard-coded Q4 2025 earnings window anchors for the Mag Seven case study."""
    pd = _import_pandas()
    return pd.DataFrame(
        [
            {"ticker": ticker, "company": company, "earnings_date": earnings_date}
            for ticker, (company, earnings_date) in MAG7_Q4_2025_EARNINGS.items()
        ]
    )


def build_earnings_price_eval(
    earnings_df=None,
    pre_trading_days: int = 7,
    post_trading_days: int = 7,
    long_post_trading_days: int = 30,
    benchmarks: tuple[str, ...] = ("SPY", "QQQ"),
):
    """Build event-window price paths and return metrics around earnings dates."""
    pd = _import_pandas()
    np = _import_numpy()
    yf = _import_yfinance()
    earnings_df = earnings_df.copy() if earnings_df is not None else mag7_q4_2025_earnings_df()
    long_return_col = f"post_{long_post_trading_days}d_return"
    long_direction_col = f"realized_{long_post_trading_days}d_direction_vs_qqq"

    price_paths = []
    summary_rows = []
    for row in earnings_df.itertuples(index=False):
        event_date = pd.Timestamp(row.earnings_date).normalize()
        ticker_close, ticker_download_error = _download_close_series(
            yf,
            row.ticker,
            event_date,
            pre_trading_days,
            long_post_trading_days,
        )
        ticker_path, ticker_error = _event_window_from_close(
            ticker_close,
            event_date,
            pre_trading_days,
            long_post_trading_days,
        )
        if ticker_error:
            summary_rows.append(
                {
                    "ticker": row.ticker,
                    "company": row.company,
                    "earnings_date": row.earnings_date,
                    "error": ticker_download_error or ticker_error,
                }
            )
            continue

        ticker_path["ticker"] = row.ticker
        ticker_path["company"] = row.company
        ticker_path["symbol_type"] = "stock"
        price_paths.append(ticker_path)

        metrics = _compute_event_returns(ticker_path, pre_trading_days, post_trading_days, np, long_post_trading_days)
        summary = {
            "ticker": row.ticker,
            "company": row.company,
            "earnings_date": row.earnings_date,
            **metrics,
        }

        for benchmark in benchmarks:
            benchmark_close, benchmark_download_error = _download_close_series(
                yf,
                benchmark,
                event_date,
                pre_trading_days,
                long_post_trading_days,
            )
            benchmark_path, benchmark_error = _event_window_from_close(
                benchmark_close,
                event_date,
                pre_trading_days,
                long_post_trading_days,
            )
            if benchmark_error:
                summary[f"{benchmark.lower()}_post_7d_return"] = np.nan
                summary[f"abnormal_vs_{benchmark.lower()}"] = np.nan
                summary[f"{benchmark.lower()}_post_{long_post_trading_days}d_return"] = np.nan
                summary[f"abnormal_{long_post_trading_days}d_vs_{benchmark.lower()}"] = np.nan
                summary[f"{benchmark.lower()}_error"] = benchmark_download_error or benchmark_error
                continue
            benchmark_metrics = _compute_event_returns(
                benchmark_path,
                pre_trading_days,
                post_trading_days,
                np,
                long_post_trading_days,
            )
            stock_short_return = metrics.get("post_7d_return", np.nan)
            benchmark_short_return = benchmark_metrics.get("post_7d_return", np.nan)
            stock_long_return = metrics.get(long_return_col, np.nan)
            benchmark_long_return = benchmark_metrics.get(long_return_col, np.nan)
            summary[f"{benchmark.lower()}_post_7d_return"] = benchmark_short_return
            summary[f"abnormal_vs_{benchmark.lower()}"] = stock_short_return - benchmark_short_return
            summary[f"{benchmark.lower()}_post_{long_post_trading_days}d_return"] = benchmark_long_return
            summary[f"abnormal_{long_post_trading_days}d_vs_{benchmark.lower()}"] = (
                stock_long_return - benchmark_long_return
            )

        abnormal = summary.get("abnormal_vs_qqq", np.nan)
        summary["realized_direction_vs_qqq"] = None if pd.isna(abnormal) else ("up" if abnormal > 0 else "down")
        long_abnormal = summary.get(f"abnormal_{long_post_trading_days}d_vs_qqq", np.nan)
        summary[long_direction_col] = (
            None if pd.isna(long_abnormal) else ("up" if long_abnormal > 0 else "down")
        )
        summary["long_horizon_trading_days"] = long_post_trading_days
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    paths_df = pd.concat(price_paths, ignore_index=True) if price_paths else pd.DataFrame()
    return summary_df, paths_df


def run_agent_event_window_eval(
    summary_df,
    lookback_days: int = 14,
    max_news: int = 10,
    news_sources: list[str] | None = None,
    provider: str = "auto",
    model: str = "gpt-5.4-nano",
    gemini_model: str = "gemini-2.5-flash",
    neutral_band: float = 0.02,
    news_end_offset_days: int = -1,
    long_post_trading_days: int | None = None,
    quiet: bool = True,
):
    """Run the News & Macro Agent against no-leakage event windows and score both horizons."""
    pd = _import_pandas()
    news_sources = news_sources or ["finnhub", "newsapi", "yfinance"]
    if long_post_trading_days is None:
        if "long_horizon_trading_days" in summary_df.columns and not summary_df["long_horizon_trading_days"].dropna().empty:
            long_post_trading_days = int(summary_df["long_horizon_trading_days"].dropna().iloc[0])
        else:
            long_post_trading_days = 30
    long_return_col = f"post_{long_post_trading_days}d_return"
    long_abnormal_col = f"abnormal_{long_post_trading_days}d_vs_qqq"
    long_direction_col = f"realized_{long_post_trading_days}d_direction_vs_qqq"
    required = [
        "ticker",
        "company",
        "earnings_date",
        "post_7d_return",
        "abnormal_vs_qqq",
        "realized_direction_vs_qqq",
        long_return_col,
        long_abnormal_col,
        long_direction_col,
    ]
    agent_eval = summary_df[required].copy()

    agent_eval["news_context_ready"] = agent_eval["earnings_date"].notna()
    agent_eval["report_ready"] = False
    agent_eval["agent_short_term_stance"] = None
    agent_eval["agent_long_term_stance"] = None
    agent_eval["agent_stance"] = None
    agent_eval["stance_rationale"] = None
    agent_eval["short_direction_match"] = None
    agent_eval["short_direction_match_reason"] = None
    agent_eval["long_direction_match"] = None
    agent_eval["long_direction_match_reason"] = None
    agent_eval["direction_match"] = None
    agent_eval["direction_match_reason"] = None
    agent_reports: dict[str, NewsMacroReport] = {}

    for idx, row in agent_eval.iterrows():
        if not row["news_context_ready"]:
            agent_eval.loc[idx, "short_direction_match_reason"] = "missing earnings date"
            agent_eval.loc[idx, "long_direction_match_reason"] = "missing earnings date"
            agent_eval.loc[idx, "direction_match_reason"] = "missing earnings date"
            continue

        if quiet:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", UserWarning)
                event_context = collect_context(
                    ticker=row["ticker"],
                    company=row["company"],
                    event_date=row["earnings_date"],
                    news_mode="event_window",
                    lookback_days=lookback_days,
                    max_news=max_news,
                    news_sources=news_sources,
                    use_sample_if_empty=False,
                    news_end_offset_days=news_end_offset_days,
                )
                event_report = generate_report(
                    event_context,
                    provider=provider,
                    model=model,
                    gemini_model=gemini_model,
                )
        else:
            event_context = collect_context(
                ticker=row["ticker"],
                company=row["company"],
                event_date=row["earnings_date"],
                news_mode="event_window",
                lookback_days=lookback_days,
                max_news=max_news,
                news_sources=news_sources,
                use_sample_if_empty=False,
                news_end_offset_days=news_end_offset_days,
            )
            event_report = generate_report(
                event_context,
                provider=provider,
                model=model,
                gemini_model=gemini_model,
            )
        agent_reports[row["ticker"]] = event_report

        short_term_stance = getattr(event_report, "short_term_stance", "Neutral")
        long_term_stance = getattr(event_report, "long_term_stance", "Neutral")
        stance_rationale = getattr(event_report, "stance_rationale", "No stance rationale available.")
        short_predicted = _stance_to_direction(short_term_stance)
        long_predicted = _stance_to_direction(long_term_stance)
        short_realized = row["realized_direction_vs_qqq"]
        long_realized = row[long_direction_col]
        short_abnormal = row["abnormal_vs_qqq"]
        long_abnormal = row[long_abnormal_col]

        agent_eval.loc[idx, "report_ready"] = True
        agent_eval.loc[idx, "agent_short_term_stance"] = short_term_stance
        agent_eval.loc[idx, "agent_long_term_stance"] = long_term_stance
        agent_eval.loc[idx, "agent_stance"] = f"ST: {short_term_stance}; LT: {long_term_stance}"
        agent_eval.loc[idx, "stance_rationale"] = stance_rationale
        short_match = _direction_match(short_predicted, short_realized, short_abnormal, pd, neutral_band)
        long_match = _direction_match(long_predicted, long_realized, long_abnormal, pd, neutral_band)
        short_reason = _direction_match_reason(
            short_predicted,
            short_realized,
            short_abnormal,
            pd,
            neutral_band,
        )
        long_reason = _direction_match_reason(
            long_predicted,
            long_realized,
            long_abnormal,
            pd,
            neutral_band,
        )

        agent_eval.loc[idx, "short_direction_match"] = short_match
        agent_eval.loc[idx, "short_direction_match_reason"] = short_reason
        agent_eval.loc[idx, "long_direction_match"] = long_match
        agent_eval.loc[idx, "long_direction_match_reason"] = long_reason
        agent_eval.loc[idx, "direction_match"] = short_match
        agent_eval.loc[idx, "direction_match_reason"] = short_reason

    return agent_eval, agent_reports


def format_return_columns(df):
    """Return a display copy with return columns formatted as percentages."""
    pd = _import_pandas()
    display_df = df.copy()
    return_cols = [col for col in display_df.columns if "return" in col or "abnormal" in col]
    for col in return_cols:
        display_df[col] = display_df[col].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    return display_df


def plot_earnings_eval(summary_df, paths_df) -> None:
    """Plot normalized event-window paths and 7-day abnormal returns vs QQQ."""
    if paths_df is None or paths_df.empty:
        print("No price paths available. Check yfinance access or event dates.")
        return

    plt = _import_matplotlib_pyplot()
    plt.figure(figsize=(11, 6))
    for ticker, group in paths_df.groupby("ticker"):
        plt.plot(group["relative_trading_day"], group["normalized_price"], marker="o", linewidth=1.8, label=ticker)
    plt.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.title("Magnificent Seven Normalized Price Path Around Q4 2025 Earnings")
    plt.xlabel("Trading days relative to pre-earnings anchor day")
    plt.ylabel("Normalized close price, anchor = 1.0")
    plt.legend(ncol=4)
    plt.grid(alpha=0.25)
    plt.show()

    plot_df = summary_df.dropna(subset=["abnormal_vs_qqq"]).sort_values("abnormal_vs_qqq")
    plt.figure(figsize=(10, 5))
    colors = ["#c44e52" if value < 0 else "#4c72b0" for value in plot_df["abnormal_vs_qqq"]]
    plt.bar(plot_df["ticker"], plot_df["abnormal_vs_qqq"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Post-Earnings 7-Trading-Day Abnormal Return vs QQQ")
    plt.ylabel("Abnormal return")
    plt.gca().yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    plt.grid(axis="y", alpha=0.25)
    plt.show()


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise ImportError("pandas is required for evaluation helpers.") from exc
    return pd


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise ImportError("numpy is required for evaluation helpers.") from exc
    return np


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        raise ImportError("yfinance is required for evaluation helpers.") from exc
    return yf


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise ImportError("matplotlib is required for plotting evaluation results.") from exc
    return plt


def _extract_close(downloaded):
    pd = _import_pandas()
    if downloaded is None or downloaded.empty:
        return pd.Series(dtype=float)
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            close = downloaded["Close"]
        else:
            close = downloaded.xs("Close", axis=1, level=-1, drop_level=False)
        if hasattr(close, "iloc") and not isinstance(close, pd.Series):
            close = close.iloc[:, 0]
    else:
        close = downloaded.get("Close", pd.Series(dtype=float))
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def _series_scalar(series, position: int) -> float:
    value = series.iloc[position]
    if hasattr(value, "iloc"):
        value = value.iloc[0]
    return float(value)


def _download_close_series(yf, symbol: str, event_date, pre_days: int, post_days: int):
    start = (event_date - _import_pandas().Timedelta(days=max(30, pre_days * 4))).date().isoformat()
    end = (event_date + _import_pandas().Timedelta(days=max(30, post_days * 2))).date().isoformat()
    attempts: list[str] = []

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        close = _extract_close(data)
        if not close.empty:
            return close, None
        attempts.append("yf.download returned no rows")
    except Exception as exc:
        attempts.append(f"yf.download failed: {exc}")

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data = yf.Ticker(symbol).history(
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
            )
        close = _extract_close(data)
        if not close.empty:
            return close, None
        attempts.append("Ticker.history returned no rows")
    except Exception as exc:
        attempts.append(f"Ticker.history failed: {exc}")

    return _import_pandas().Series(dtype=float), (
        f"No close prices for {symbol} from {start} through {end}. "
        "This usually means Yahoo/yfinance failed or rate-limited the notebook, not that the ticker is delisted. "
        + "; ".join(attempts)
    )


def _event_window_from_close(close, event_date, pre_days: int, post_days: int):
    pd = _import_pandas()
    if close.empty:
        return pd.DataFrame(), "No close prices available."

    dates = pd.Series(close.index, index=range(len(close)))
    before = dates[dates < event_date]
    if before.empty:
        return pd.DataFrame(), "No trading day before earnings date."

    anchor_pos = before.index.max()
    start_pos = max(0, anchor_pos - pre_days)
    end_pos = min(len(close) - 1, anchor_pos + post_days)
    window = close.iloc[start_pos : end_pos + 1].copy()
    rel_days = list(range(start_pos - anchor_pos, end_pos - anchor_pos + 1))
    anchor_price = float(close.iloc[anchor_pos])
    result = pd.DataFrame(
        {
            "date": window.index,
            "relative_trading_day": rel_days,
            "close": window.values,
            "normalized_price": window.values / anchor_price,
        }
    )
    return result, None


def _value_at_or_after(path, relative_day: int, np):
    subset = path[path["relative_trading_day"] >= relative_day]
    if subset.empty:
        return np.nan
    return float(subset.iloc[0]["normalized_price"])


def _value_at_or_before(path, relative_day: int, np):
    subset = path[path["relative_trading_day"] <= relative_day]
    if subset.empty:
        return np.nan
    return float(subset.iloc[-1]["normalized_price"])


def _compute_event_returns(path, pre_days: int, post_days: int, np, long_post_days: int = 30) -> dict[str, float]:
    if path.empty:
        return {}
    pre_start = _value_at_or_after(path, -pre_days, np)
    anchor = _value_at_or_before(path, 0, np)
    d1 = _value_at_or_after(path, 1, np)
    d3 = _value_at_or_after(path, 3, np)
    d7 = _value_at_or_after(path, post_days, np)
    d_long = _value_at_or_after(path, long_post_days, np)
    return {
        "pre_7d_return": anchor / pre_start - 1 if pre_start and anchor else np.nan,
        "post_1d_return": d1 / anchor - 1 if d1 and anchor else np.nan,
        "post_3d_return": d3 / anchor - 1 if d3 and anchor else np.nan,
        "post_7d_return": d7 / anchor - 1 if d7 and anchor else np.nan,
        f"post_{long_post_days}d_return": d_long / anchor - 1 if d_long and anchor else np.nan,
        "full_window_return": d7 / pre_start - 1 if d7 and pre_start else np.nan,
        f"full_{long_post_days}d_window_return": d_long / pre_start - 1 if d_long and pre_start else np.nan,
    }


def _stance_to_direction(stance: str | None) -> str:
    if stance == "Bullish":
        return "up"
    if stance == "Bearish":
        return "down"
    return "neutral"


def _direction_match(predicted: str, realized: str | None, abnormal, pd, neutral_band: float):
    if realized is None or pd.isna(abnormal):
        return None
    if predicted == "neutral":
        return abs(float(abnormal)) <= neutral_band
    return predicted == realized


def _direction_match_reason(predicted: str, realized: str | None, abnormal, pd, neutral_band: float) -> str:
    if pd.isna(abnormal):
        return "missing post-earnings 7-trading-day abnormal return"
    if realized is None:
        return "missing realized direction"
    if predicted == "neutral":
        if abs(float(abnormal)) <= neutral_band:
            return f"neutral matched: abnormal return stayed within +/-{neutral_band:.0%}"
        return f"neutral missed: abnormal return moved outside +/-{neutral_band:.0%}"
    return "matched" if predicted == realized else "missed"


def _dedupe_news(items: Iterable[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for item in items:
        key = re.sub(r"\W+", " ", item.title.lower()).strip()
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _rank_news_by_relevance(
    items: list[NewsItem],
    ticker: str,
    company: str,
    related_tickers: list[str] | None = None,
) -> list[NewsItem]:
    return sorted(
        items,
        key=lambda item: _news_relevance_score(item, ticker, company, related_tickers=related_tickers),
        reverse=True,
    )


def _build_newsapi_queries(
    company: str,
    ticker: str | None,
    related_tickers: list[str] | None = None,
) -> list[tuple[str, str]]:
    company_part = f'"{company}"'
    if ticker:
        company_part = f'("{company}" OR {ticker.upper()})'
    finance_terms = "(stock OR shares OR earnings OR revenue OR guidance OR analyst OR price target OR margin OR investors)"
    industry_terms = (
        "(AI chip OR semiconductor OR data center OR cloud infrastructure OR supply chain OR networking "
        "OR power OR competitor OR inference OR GPU)"
    )
    exclusions = "NOT driver NOT download NOT pypi NOT linux NOT gaming NOT crypto"
    queries = [
        ("direct-company", f"{company_part} AND {finance_terms} {exclusions}"),
        ("industry-chain", f"{company_part} AND {industry_terms} {exclusions}"),
    ]
    related_terms = _related_newsapi_terms(related_tickers or [])
    if related_terms:
        queries.append(
            (
                "related-company",
                f"({related_terms}) AND ({finance_terms} OR {industry_terms}) {exclusions}",
            )
        )
    return queries


def _is_finance_relevant_news(item: NewsItem, ticker: str, company: str) -> bool:
    text = f"{item.title} {item.summary or ''}".lower()
    source = item.source.lower()

    noisy_sources = {
        "softpedia.com",
        "pypi.org",
        "gizmodo.com",
        "slashdot.org",
    }
    if source in noisy_sources:
        return False

    noisy_terms = [
        "driver",
        "download",
        "windows 10",
        "windows 11",
        "pypi",
        "linux distribution",
        "crypto investor",
    ]
    if any(term in text for term in noisy_terms):
        return False

    relevant_terms = [
        "stock",
        "shares",
        "earnings",
        "revenue",
        "guidance",
        "analyst",
        "price target",
        "margin",
        "investors",
        "semiconductor",
        "ai chip",
        "data center",
        "gpu",
        "supply chain",
        "cloud",
        "inference",
        "competitor",
    ]
    if any(term in text for term in relevant_terms):
        return True
    if ticker and ticker.lower() in text:
        return True
    return any(term in text for term in _company_terms(company))


def _news_relevance_score(
    item: NewsItem,
    ticker: str,
    company: str,
    related_tickers: list[str] | None = None,
) -> tuple[float, str]:
    text = f"{item.title} {item.summary or ''}".lower()
    company_terms = _company_terms(company)
    score = 0.0

    if ticker.lower() in text:
        score += 8.0
    for term in company_terms:
        if term and term in text:
            score += 4.0

    related_tickers = related_tickers or []
    for related_ticker in related_tickers:
        if related_ticker.lower() in text or f"related:{related_ticker.lower()}" in item.source.lower():
            score += 2.5
        related_company = RELATED_COMPANY_NAMES.get(related_ticker.upper())
        if related_company and any(term in text for term in _company_terms(related_company)):
            score += 1.5

    if any(term in text for term in ["earnings", "revenue", "guidance", "margin", "forecast", "outlook"]):
        score += 2.0
    if any(term in text for term in ["analyst", "upgrade", "downgrade", "price target", "rating"]):
        score += 1.5
    if any(term in text for term in ["ai", "semiconductor", "chip", "data center", "supply chain", "regulation"]):
        score += 1.0

    source = item.source.lower()
    if "finnhub" in source:
        score += 1.0
    elif "yahoo" in source:
        score += 0.5

    published_date = _parse_date(item.published_at)
    date_key = published_date.isoformat() if published_date else ""
    return score, date_key


def _related_tickers_for(ticker: str) -> list[str]:
    ticker = ticker.upper()
    return [related for related in RELATED_TICKER_UNIVERSE.get(ticker, []) if related != ticker]


def _related_newsapi_terms(related_tickers: list[str]) -> str:
    terms: list[str] = []
    for related_ticker in related_tickers[:6]:
        terms.append(related_ticker.upper())
        company = RELATED_COMPANY_NAMES.get(related_ticker.upper())
        if company:
            terms.append(f'"{company}"')
    return " OR ".join(dict.fromkeys(terms))


def _company_terms(company: str) -> list[str]:
    base = re.sub(r"\b(inc|incorporated|corp|corporation|ltd|limited|plc|company|co)\b\.?", "", company.lower())
    terms = [base.strip()]
    terms.extend(part for part in re.split(r"\W+", base) if len(part) >= 4)
    return list(dict.fromkeys(term for term in terms if term))


def _clean_oneline(value: str | None, max_chars: int = 280) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _normalize_stance(value: Any) -> str:
    stance = str(value or "Neutral").strip().lower()
    if "bull" in stance or "positive" in stance or "看好" in stance:
        return "Bullish"
    if "bear" in stance or "negative" in stance or "不看好" in stance:
        return "Bearish"
    return "Neutral"


def _heuristic_stance(score: float, has_ai: bool, has_rates: bool, horizon: str) -> str:
    if score > 0.15 and (has_ai or horizon == "short"):
        return "Bullish"
    if score < -0.15 or (has_rates and not has_ai):
        return "Bearish"
    return "Neutral"


def _filter_news_by_event_window(
    items: list[NewsItem],
    event_date: str | None,
    lookback_days: int,
    news_end_offset_days: int = 1,
) -> tuple[list[NewsItem], list[str]]:
    anchor_date = _parse_date(event_date)
    if not anchor_date:
        return items, []

    start = anchor_date - timedelta(days=lookback_days)
    end = anchor_date + timedelta(days=news_end_offset_days)
    filtered = []
    unknown_date_items = []
    for item in items:
        published_date = _parse_date(item.published_at)
        if not published_date:
            unknown_date_items.append(item)
        elif start <= published_date <= end:
            filtered.append(item)

    notes = [
        f"Filtered Yahoo Finance news to event window {start.isoformat()} through {end.isoformat()}."
    ]
    if items and not filtered:
        notes.append(
            "Yahoo Finance usually returns current news only; no Yahoo items matched the requested event window."
        )
    return filtered + unknown_date_items, notes


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _sentiment_score(texts: list[str]) -> float:
    scores: list[float] = []
    for text in texts:
        tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        pos = len(tokens & POSITIVE_TERMS)
        neg = len(tokens & NEGATIVE_TERMS)
        if pos or neg:
            scores.append((pos - neg) / max(pos + neg, 1))
    if not scores:
        return 0.0
    return statistics.mean(scores)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON.")
    return json.loads(match.group(0))
