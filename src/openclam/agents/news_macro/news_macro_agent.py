from __future__ import annotations

import json
import os
import re
import statistics
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
    to_date = (anchor_date + timedelta(days=1)).isoformat()
    queries = _build_newsapi_queries(company, ticker)
    articles = []
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
            articles.extend(batch)
            notes.append(f"NewsAPI {label} query returned {len(batch)} article(s).")
        except Exception as exc:
            notes.append(f"NewsAPI {label} query failed: {exc}")

    items = [
        NewsItem(
            title=article.get("title") or "",
            source=(article.get("source") or {}).get("name") or "NewsAPI",
            published_at=article.get("publishedAt"),
            url=article.get("url"),
            summary=article.get("description"),
        )
        for article in articles
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
    to_date = (anchor_date + timedelta(days=1)).isoformat()
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
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                notes.append(f"Not enough data for {symbol}.")
                continue
            start = float(closes.iloc[0])
            end = float(closes.iloc[-1])
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

    if news_mode == "latest":
        notes.append("Using latest-news mode; event_date is kept as metadata but not used to filter news.")

    if "finnhub" in news_sources:
        finnhub_news, finnhub_notes = fetch_finnhub_company_news(
            ticker,
            lookback_days=lookback_days,
            max_items=max_news,
            event_date=event_date,
            news_mode=news_mode,
        )
        news.extend(finnhub_news)
        notes.extend(finnhub_notes)

    if "newsapi" in news_sources:
        api_news, api_notes = fetch_newsapi_news(
            company,
            ticker=ticker,
            lookback_days=lookback_days,
            max_items=max_news,
            event_date=event_date,
            news_mode=news_mode,
        )
        news.extend(api_news)
        notes.extend(api_notes)

    if "yfinance" in news_sources:
        yf_news, yf_notes = fetch_yfinance_news(ticker, max_news * 3)
        filter_notes: list[str] = []
        if news_mode == "event_window":
            yf_news, filter_notes = _filter_news_by_event_window(yf_news, event_date, lookback_days)
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

    news = _rank_news_by_relevance(_dedupe_news(news), ticker, company)[:max_news]
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
    model: str = "gpt-4.1-mini",
    gemini_model: str = "gemini-2.5-flash",
    provider: str = "auto",
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
) -> NewsMacroReport:
    """Generate a structured report with OpenAI, Gemini API, Vertex AI, or heuristic fallback."""
    provider = provider.lower()
    if provider not in {"auto", "openai", "gemini", "vertex"}:
        raise ValueError("provider must be one of: 'auto', 'openai', 'gemini', or 'vertex'.")

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


def analyze_news_text(
    news_text: str,
    ticker: str | None = None,
    company: str | None = None,
    source: str = "user-provided news",
    event_date: str | None = None,
    model: str = "gpt-4.1-mini",
    gemini_model: str = "gemini-2.5-flash",
    provider: str = "auto",
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
) -> NewsMacroReport:
    """
    Analyze pasted news text directly, which is convenient for Colab demos.

    provider options:
    - "auto": use OpenAI if OPENAI_API_KEY exists; otherwise Gemini if GEMINI_API_KEY exists; otherwise heuristic fallback.
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
Read the supplied news and macro context. First identify the mainstream large-cap stocks that investors are most likely to watch immediately. Then map the full industry chain to find second-order US equity opportunities.
Your analysis order must be:

Step 1: Mainstream large-stock screen.
List 3-5 directly related, highly liquid, large-cap US-listed stocks or ADRs that mainstream capital would likely react to first. This step establishes consensus, liquidity anchors, and first-order market reaction. It is not the main recommendation section.

Step 2: Industry-chain second-order inference.
Starting from the Step 1 mainstream reaction, expand into upstream suppliers, critical components, substitute products, competitors, solution providers, downstream margin victims, logistics, insurance, infrastructure, or other knock-on beneficiaries/victims. Output 3-5 overlooked second-order US-listed opportunities, including both long and short ideas when appropriate. Quantify the potential earnings impact.

# Second-Order Framework
1. Competitor Displacement: If a leader is impaired by bankruptcy, scandal, capacity constraints, or regulation, who can take share?
2. Substitution Effect: If product A becomes unavailable, expensive, or risky, what product B will customers or enterprises adopt instead?
3. Solution Provider: After the event, who provides cleanup, protection, remediation, compliance, rebuilding, or tools?
4. Pick-and-Shovel: If a trend accelerates, who sells the infrastructure, components, data, equipment, or critical inputs behind it?
5. Cost Squeeze & Victims: Does the event raise input costs, break supply chains, reduce availability, or pressure margins for downstream companies?

# Constraints
- All output must be in English.
- Recommend only US-listed companies or ADRs on NYSE/NASDAQ where possible.
- Step 1 may include the news protagonist and obvious mega-cap stocks.
- Step 2 should not simply repeat the news protagonist unless it is a short idea.
- Step 2 must explicitly identify where the company sits in the industry chain.
- Provide exact company names and tickers.
- Give an impact score from 1-10 based on likely earnings impact over the next 1-3 quarters if the event continues to develop.
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


def _dedupe_news(items: Iterable[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for item in items:
        key = re.sub(r"\W+", " ", item.title.lower()).strip()
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _rank_news_by_relevance(items: list[NewsItem], ticker: str, company: str) -> list[NewsItem]:
    return sorted(items, key=lambda item: _news_relevance_score(item, ticker, company), reverse=True)


def _build_newsapi_queries(company: str, ticker: str | None) -> list[tuple[str, str]]:
    company_part = f'"{company}"'
    if ticker:
        company_part = f'("{company}" OR {ticker.upper()})'
    finance_terms = "(stock OR shares OR earnings OR revenue OR guidance OR analyst OR price target OR margin OR investors)"
    industry_terms = (
        "(AI chip OR semiconductor OR data center OR cloud infrastructure OR supply chain OR networking "
        "OR power OR competitor OR inference OR GPU)"
    )
    exclusions = "NOT driver NOT download NOT pypi NOT linux NOT gaming NOT crypto"
    return [
        ("direct-company", f"{company_part} AND {finance_terms} {exclusions}"),
        ("industry-chain", f"{company_part} AND {industry_terms} {exclusions}"),
    ]


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


def _news_relevance_score(item: NewsItem, ticker: str, company: str) -> tuple[float, str]:
    text = f"{item.title} {item.summary or ''}".lower()
    company_terms = _company_terms(company)
    score = 0.0

    if ticker.lower() in text:
        score += 6.0
    for term in company_terms:
        if term and term in text:
            score += 3.0

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


def _filter_news_by_event_window(
    items: list[NewsItem],
    event_date: str | None,
    lookback_days: int,
) -> tuple[list[NewsItem], list[str]]:
    anchor_date = _parse_date(event_date)
    if not anchor_date:
        return items, []

    start = anchor_date - timedelta(days=lookback_days)
    end = anchor_date + timedelta(days=1)
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
