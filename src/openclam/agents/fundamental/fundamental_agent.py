from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from textwrap import dedent
from typing import Any, Sequence

import pandas as pd
import requests


FUNDAMENTAL_SYSTEM_PROMPT = """
You are a senior equity research Fundamental Analyst Agent.

Use only the provided evidence. Do not invent facts.

Important rules:
- yfinance may not include analyst consensus EPS/revenue estimates.
- If consensus estimates are missing, beat_or_miss MUST be "Unclear".
- If earnings call transcript snippets are provided, you MUST infer management_tone.
- Only use management_tone = "Unclear" if transcript is missing or irrelevant.
- If guidance text contains forward-looking expectations, classify guidance_change as Raise, Cut, Maintain, or Unclear.
- Do NOT call QoQ changes YoY. Only use YoY if the input explicitly says YoY.
- Do NOT say undervalued or overvalued unless peer comparison or valuation benchmark is provided.
- If valuation data is available but no peer comparison exists, describe it as "premium valuation" or "valuation requires peer comparison."
- Every positive or negative signal must be supported by evidence from the input.
- Use guidance_change = "Raise" only if the transcript explicitly says guidance was raised or increased versus prior guidance.
- If management gives positive forward outlook but no prior guidance comparison is provided, use guidance_change = "Maintain" or "Unclear".
- negative_signals must not be empty. If fundamentals are strong, still include valuation, tariff, margin, demand, or missing-data risks when supported by input.

Analyze:
1. Revenue, profitability, cash flow, leverage, growth, and valuation.
2. Whether fundamentals are improving or weakening.
3. Management tone from transcript.
4. Guidance direction if available.
5. Medium-term thesis impact.
6. Key positive and negative signals.
7. Missing information and uncertainty.

Return JSON only.
"""

_GUIDANCE_KEYWORDS = (
    "we expect",
    "we anticipate",
    "outlook",
    "guidance",
    "next quarter",
    "december quarter",
    "next year",
    "gross margin",
    "revenue to grow",
    "operating expenses",
    "headwind",
    "tailwind",
)

_STATEMENT_ROW_ALIASES: dict[str, tuple[str, ...]] = {
    "Total Revenue": ("Revenue",),
    "Gross Profit": (),
    "Operating Income": ("Operating Income Loss",),
    "Net Income": ("Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"),
    "Operating Cash Flow": (
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Net Cash Provided By Operating Activities",
    ),
    "Free Cash Flow": ("Free Cash Flow",),
    "Total Debt": (
        "Total Debt",
        "Total Debt And Capital Lease Obligation",
        "Long Term Debt And Capital Lease Obligation",
    ),
    "Cash And Cash Equivalents": (
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
    ),
    "Capital Expenditure": (
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
    ),
}


class JsonModelMixin:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), indent=indent)


@dataclass
class FinancialMetric(JsonModelMixin):
    name: str
    current_value: float | None = None
    prior_value: float | None = None
    consensus_estimate: float | None = None
    unit: str | None = None


@dataclass
class FundamentalInput(JsonModelMixin):
    ticker: str
    company_name: str
    quarter: str
    earnings_summary: str
    financial_metrics: list[FinancialMetric]
    earnings_call_snippets: list[str] = field(default_factory=list)
    guidance_text: str | None = None


@dataclass
class FundamentalOutput(JsonModelMixin):
    ticker: str
    stance: str
    core_judgment: str
    positive_signals: list[str]
    negative_signals: list[str]
    key_evidence: list[str]
    beat_or_miss: str
    guidance_change: str
    management_tone: str
    thesis_impact: str
    thesis_impact_reasoning: str
    confidence: float
    confidence_reasoning: str
    missing_information: list[str]

    def __post_init__(self) -> None:
        self.confidence = _coerce_confidence(self.confidence)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FundamentalOutput":
        return cls(
            ticker=str(payload.get("ticker", "")),
            stance=str(payload.get("stance", "Watch")),
            core_judgment=str(payload.get("core_judgment", "")),
            positive_signals=_coerce_str_list(payload.get("positive_signals")),
            negative_signals=_coerce_str_list(payload.get("negative_signals")),
            key_evidence=_coerce_str_list(payload.get("key_evidence")),
            beat_or_miss=str(payload.get("beat_or_miss", "Unclear")),
            guidance_change=str(payload.get("guidance_change", "Unclear")),
            management_tone=str(payload.get("management_tone", "Unclear")),
            thesis_impact=str(payload.get("thesis_impact", "Unclear")),
            thesis_impact_reasoning=str(payload.get("thesis_impact_reasoning", "")),
            confidence=_coerce_confidence(payload.get("confidence", 0.0)),
            confidence_reasoning=str(payload.get("confidence_reasoning", "")),
            missing_information=_coerce_str_list(payload.get("missing_information")),
        )

    @classmethod
    def model_validate_json(cls, raw_text: str) -> "FundamentalOutput":
        payload = _extract_json_payload(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("FundamentalOutput expects a JSON object")
        return cls.from_dict(payload)


@dataclass
class FundamentalWorkflowResult(JsonModelMixin):
    fundamental_input: FundamentalInput
    fundamental_output: FundamentalOutput


def _candidate_row_names(row_name: str | Sequence[str]) -> list[str]:
    if isinstance(row_name, str):
        candidates = [row_name]
    else:
        candidates = list(row_name)

    expanded: list[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        expanded.extend(_STATEMENT_ROW_ALIASES.get(candidate, ()))

    deduped: list[str] = []
    for candidate in expanded:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    """Parse model confidence robustly and clamp it to [0, 1]."""
    if value is None:
        confidence = default
    elif isinstance(value, str):
        stripped = value.strip()
        cleaned = stripped.replace("%", "")
        try:
            confidence = float(cleaned)
            if "%" in stripped or confidence >= 10:
                confidence = confidence / 100
        except ValueError:
            confidence = default
    else:
        try:
            confidence = float(value)
            if confidence >= 10:
                confidence = confidence / 100
        except (TypeError, ValueError):
            confidence = default
    return max(0.0, min(1.0, confidence))


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item is not None]
        except Exception:
            pass
        return [stripped]
    return [str(value)]


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract a JSON object from raw model output, including fenced/verbose text."""
    text = (raw_text or "").strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Fundamental Agent output.")

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Greedy regex can capture too much if text contains multiple objects.
        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(text[start:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
        raise


def _format_statement_period(statement: pd.DataFrame | None) -> str:
    if statement is None or statement.empty:
        return "Unknown"

    try:
        return str(pd.Timestamp(statement.columns[0]).date())
    except Exception:
        return str(statement.columns[0])


def safe_get_statement_value(
    statement: pd.DataFrame | None,
    row_name: str | Sequence[str],
    col_idx: int = 0,
) -> float | None:
    """Safely get a numeric value from a yfinance financial statement."""
    try:
        if statement is None or statement.empty:
            return None
        if col_idx >= len(statement.columns):
            return None

        for candidate in _candidate_row_names(row_name):
            if candidate not in statement.index:
                continue
            value = _safe_float(statement.loc[candidate].iloc[col_idx])
            if value is not None:
                return value
        return None
    except Exception:
        return None


def get_clean_statement(
    statement: pd.DataFrame | None,
    required_rows: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    if statement is None or statement.empty:
        return None

    cleaned = statement.copy()
    try:
        cleaned = cleaned.loc[:, sorted(cleaned.columns, key=lambda col: pd.Timestamp(col), reverse=True)]
    except Exception:
        pass

    if required_rows is None:
        return cleaned.dropna(axis=1, how="all")

    valid_cols: list[Any] = []
    for col in cleaned.columns:
        has_required_data = True
        for row in required_rows:
            matching_rows = [candidate for candidate in _candidate_row_names(row) if candidate in cleaned.index]
            if not matching_rows:
                has_required_data = False
                break

            if all(pd.isna(cleaned.loc[candidate, col]) for candidate in matching_rows):
                has_required_data = False
                break

        if has_required_data:
            valid_cols.append(col)

    if not valid_cols:
        return cleaned.dropna(axis=1, how="all")

    return cleaned[valid_cols]


def get_transcript_from_fmp(
    ticker: str,
    year: int,
    quarter: int,
    max_chunks: int = 20,
) -> list[str]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []

    candidate_periods = [
        (year, quarter),
        (year, max(1, quarter - 1)),
        (year, min(4, quarter + 1)),
        (year - 1, quarter),
        (year + 1, quarter),
    ]

    seen_periods: list[tuple[int, int]] = []
    for candidate_year, candidate_quarter in candidate_periods:
        period = (candidate_year, candidate_quarter)
        if period in seen_periods:
            continue
        seen_periods.append(period)

        url = (
            "https://financialmodelingprep.com/stable/earning-call-transcript"
            f"?symbol={ticker}&year={candidate_year}&quarter={candidate_quarter}&apikey={api_key}"
        )

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue

        if isinstance(data, list):
            text = data[0].get("content", "") if data else ""
        elif isinstance(data, dict):
            text = data.get("content", "")
        else:
            text = ""

        if not text:
            continue

        chunks = [chunk.strip() for chunk in text.split("\n") if len(chunk.strip()) > 50]
        if chunks:
            return chunks[:max_chunks]

    return []


def extract_guidance_text(transcript_snippets: Sequence[str] | None) -> str:
    if not transcript_snippets:
        return "No guidance information available."

    selected = [
        snippet
        for snippet in transcript_snippets
        if any(keyword in snippet.lower() for keyword in _GUIDANCE_KEYWORDS)
    ]

    if not selected:
        return "Transcript available, but no explicit guidance statement was extracted."

    return "\n".join(selected[:6])


def get_yfinance_fundamental_input(
    ticker: str,
    transcript_year: int | None = None,
    transcript_quarter: int | None = None,
    require_transcript: bool = False,
) -> FundamentalInput:
    """
    Build a FundamentalInput object automatically from yfinance data.

    Provide transcript_year and transcript_quarter if you also want FMP
    earnings-call transcript snippets and extracted guidance text.
    """

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to build FundamentalInput. Install it with `pip install yfinance`."
        ) from exc

    normalized_ticker = ticker.strip().upper()
    stock = yf.Ticker(normalized_ticker)

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    company_name = info.get("longName") or info.get("shortName") or normalized_ticker

    income_stmt = get_clean_statement(
        stock.quarterly_income_stmt,
        required_rows=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )
    cashflow = get_clean_statement(
        stock.quarterly_cashflow,
        required_rows=["Operating Cash Flow", "Free Cash Flow"],
    )
    balance_sheet = get_clean_statement(stock.quarterly_balance_sheet)

    revenue = safe_get_statement_value(income_stmt, "Total Revenue", 0)
    revenue_prior = safe_get_statement_value(income_stmt, "Total Revenue", 1)
    gross_profit = safe_get_statement_value(income_stmt, "Gross Profit", 0)
    gross_profit_prior = safe_get_statement_value(income_stmt, "Gross Profit", 1)
    operating_income = safe_get_statement_value(income_stmt, "Operating Income", 0)
    operating_income_prior = safe_get_statement_value(income_stmt, "Operating Income", 1)
    net_income = safe_get_statement_value(income_stmt, "Net Income", 0)
    net_income_prior = safe_get_statement_value(income_stmt, "Net Income", 1)
    operating_cash_flow = safe_get_statement_value(cashflow, "Operating Cash Flow", 0)
    operating_cash_flow_prior = safe_get_statement_value(cashflow, "Operating Cash Flow", 1)
    free_cash_flow = safe_get_statement_value(cashflow, "Free Cash Flow", 0)
    free_cash_flow_prior = safe_get_statement_value(cashflow, "Free Cash Flow", 1)
    total_debt = safe_get_statement_value(balance_sheet, "Total Debt", 0)
    cash = safe_get_statement_value(balance_sheet, "Cash And Cash Equivalents", 0)

    market_cap = _safe_float(info.get("marketCap"))
    trailing_pe = _safe_float(info.get("trailingPE"))
    forward_pe = _safe_float(info.get("forwardPE"))
    profit_margins = _safe_float(info.get("profitMargins"))
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))

    latest_quarter = _format_statement_period(income_stmt)

    metrics = [
        FinancialMetric("Revenue", revenue, revenue_prior, None, "USD"),
        FinancialMetric("Gross Profit", gross_profit, gross_profit_prior, None, "USD"),
        FinancialMetric("Operating Income", operating_income, operating_income_prior, None, "USD"),
        FinancialMetric("Net Income", net_income, net_income_prior, None, "USD"),
        FinancialMetric("Operating Cash Flow", operating_cash_flow, operating_cash_flow_prior, None, "USD"),
        FinancialMetric("Free Cash Flow", free_cash_flow, free_cash_flow_prior, None, "USD"),
        FinancialMetric("Total Debt", total_debt, None, None, "USD"),
        FinancialMetric("Cash And Cash Equivalents", cash, None, None, "USD"),
        FinancialMetric("Market Cap", market_cap, None, None, "USD"),
        FinancialMetric("Trailing P/E", trailing_pe, None, None, "ratio"),
        FinancialMetric("Forward P/E", forward_pe, None, None, "ratio"),
        FinancialMetric("Profit Margin", profit_margins, None, None, "ratio"),
        FinancialMetric("Revenue Growth", revenue_growth, None, None, "ratio"),
        FinancialMetric("Earnings Growth", earnings_growth, None, None, "ratio"),
    ]

    if free_cash_flow is None:
        capex = safe_get_statement_value(cashflow, "Capital Expenditure", 0)
        if operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow + capex if capex < 0 else operating_cash_flow - capex
            metrics[5].current_value = free_cash_flow

    if free_cash_flow_prior is None:
        capex_prior = safe_get_statement_value(cashflow, "Capital Expenditure", 1)
        if operating_cash_flow_prior is not None and capex_prior is not None:
            free_cash_flow_prior = (
                operating_cash_flow_prior + capex_prior if capex_prior < 0 else operating_cash_flow_prior - capex_prior
            )
            metrics[5].prior_value = free_cash_flow_prior

    earnings_summary = f"""
    Company: {company_name}
    Ticker: {normalized_ticker}
    Latest valid financial quarter: {latest_quarter}

    Important: current vs prior values below are quarter-over-quarter comparisons, NOT year-over-year.

    Revenue QoQ: current={revenue}, previous_quarter={revenue_prior}
    Gross Profit QoQ: current={gross_profit}, previous_quarter={gross_profit_prior}
    Operating Income QoQ: current={operating_income}, previous_quarter={operating_income_prior}
    Net Income QoQ: current={net_income}, previous_quarter={net_income_prior}
    Operating Cash Flow QoQ: current={operating_cash_flow}, previous_quarter={operating_cash_flow_prior}
    Free Cash Flow QoQ: current={free_cash_flow}, previous_quarter={free_cash_flow_prior}

    Market Cap: {market_cap}
    Trailing P/E: {trailing_pe}
    Forward P/E: {forward_pe}
    Profit Margin: {profit_margins}
    Revenue Growth from yfinance: {revenue_growth}
    Earnings Growth from yfinance: {earnings_growth}

    Do not describe QoQ changes as YoY unless the field explicitly says YoY.
    """

    earnings_call_snippets: list[str]
    guidance_text: str
    if transcript_year is not None and transcript_quarter is not None:
        if os.getenv("FMP_API_KEY"):
            transcript_snippets = get_transcript_from_fmp(
                ticker=normalized_ticker,
                year=transcript_year,
                quarter=transcript_quarter,
            )
        else:
            transcript_snippets = []

        if transcript_snippets:
            earnings_call_snippets = transcript_snippets
            guidance_text = extract_guidance_text(transcript_snippets)
        elif not os.getenv("FMP_API_KEY"):
            if require_transcript:
                raise ValueError(
                    "FMP_API_KEY is not set, so the earnings call transcript could not be retrieved."
                )
            earnings_call_snippets = [
                "FMP_API_KEY is not set, so earnings call transcript retrieval was skipped.",
            ]
            guidance_text = "FMP_API_KEY is not set, so guidance information could not be retrieved."
        else:
            if require_transcript:
                raise ValueError(
                    f"No earnings call transcript was returned for {normalized_ticker} "
                    f"(year={transcript_year}, quarter={transcript_quarter})."
                )
            earnings_call_snippets = [
                "No earnings call transcript available from Financial Modeling Prep.",
            ]
            guidance_text = "No guidance information available."
    else:
        earnings_call_snippets = [
            "Transcript fetch skipped. Provide transcript_year and transcript_quarter to load FMP transcript data.",
        ]
        guidance_text = "No guidance information available."

    return FundamentalInput(
        ticker=normalized_ticker,
        company_name=company_name,
        quarter=latest_quarter,
        earnings_summary=earnings_summary,
        financial_metrics=metrics,
        earnings_call_snippets=earnings_call_snippets,
        guidance_text=guidance_text,
    )


def build_fundamental_prompt(data: FundamentalInput) -> str:
    metrics_text = "\n".join(
        [
            f"- {metric.name}: current={metric.current_value}, previous_quarter={metric.prior_value}, "
            f"consensus_estimate={metric.consensus_estimate}, unit={metric.unit}"
            for metric in data.financial_metrics
        ]
    )
    call_text = "\n".join(data.earnings_call_snippets or ["No earnings call snippets provided."])

    return dedent(
        f"""
        Analyze the following company fundamental data.

        Ticker: {data.ticker}
        Company: {data.company_name}
        Quarter: {data.quarter}

        Earnings / Fundamental Summary:
        {data.earnings_summary}

        Financial Metrics:
        {metrics_text}

        Guidance:
        {data.guidance_text or "No guidance information provided."}

        Earnings Call Snippets:
        {call_text}

        Return a JSON object with exactly these fields:

        {{
          "ticker": "...",
          "stance": "Bullish/Bearish/Neutral/Watch",
          "core_judgment": "...",
          "positive_signals": ["..."],
          "negative_signals": ["..."],
          "key_evidence": ["..."],
          "beat_or_miss": "Beat/Miss/Mixed/Unclear",
          "guidance_change": "Raise/Cut/Maintain/Unclear",
          "management_tone": "...",
          "thesis_impact": "Strengthened/Unchanged/Weakened/Unclear",
          "thesis_impact_reasoning": "...",
          "confidence": 0.0,
          "confidence_reasoning": "...",
          "missing_information": ["..."]
        }}
        """
    ).strip()


class FundamentalAgent:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required to use FundamentalAgent. Install it with `pip install openai`."
                ) from exc

            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self.model_name = model_name
        self.temperature = temperature
        self.client = client

    def run(self, data: FundamentalInput) -> FundamentalOutput:
        messages = [
            {"role": "system", "content": FUNDAMENTAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_fundamental_prompt(data)},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    *messages,
                    {
                        "role": "user",
                        "content": "Return only one valid JSON object. Do not use markdown fences or explanatory text.",
                    },
                ],
            )

        raw_text = response.choices[0].message.content or "{}"

        try:
            return FundamentalOutput.model_validate_json(raw_text)
        except Exception as exc:
            raise ValueError(f"Unable to parse Fundamental Agent output: {raw_text}") from exc


def create_fundamental_agent(
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    client: Any | None = None,
) -> FundamentalAgent:
    """Create a configured FundamentalAgent instance."""
    return FundamentalAgent(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        client=client,
    )


def analyze_fundamental_input(
    data: FundamentalInput,
    agent: FundamentalAgent | None = None,
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    client: Any | None = None,
) -> FundamentalOutput:
    """
    Run the fundamental agent on a pre-built FundamentalInput payload.

    This mirrors the thin-notebook pattern used elsewhere in the repo:
    fetch/build structured input first, then pass it to a top-level module
    function that handles the model call.
    """
    active_agent = agent or create_fundamental_agent(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        client=client,
    )
    return active_agent.run(data)


def run_fundamental_analysis(
    ticker: str,
    transcript_year: int | None = None,
    transcript_quarter: int | None = None,
    require_transcript: bool = False,
    agent: FundamentalAgent | None = None,
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    client: Any | None = None,
) -> FundamentalOutput:
    """
    Convenience wrapper that builds the FundamentalInput and returns the
    FundamentalOutput in one call.
    """
    data = get_yfinance_fundamental_input(
        ticker=ticker,
        transcript_year=transcript_year,
        transcript_quarter=transcript_quarter,
        require_transcript=require_transcript,
    )
    return analyze_fundamental_input(
        data,
        agent=agent,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        client=client,
    )


def run_fundamental_workflow(
    ticker: str,
    transcript_year: int | None = None,
    transcript_quarter: int | None = None,
    require_transcript: bool = False,
    agent: FundamentalAgent | None = None,
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    client: Any | None = None,
) -> FundamentalWorkflowResult:
    """
    End-to-end helper that returns both the generated input payload and the
    model's final structured output.
    """
    data = get_yfinance_fundamental_input(
        ticker=ticker,
        transcript_year=transcript_year,
        transcript_quarter=transcript_quarter,
        require_transcript=require_transcript,
    )
    result = analyze_fundamental_input(
        data,
        agent=agent,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        client=client,
    )
    return FundamentalWorkflowResult(
        fundamental_input=data,
        fundamental_output=result,
    )


__all__ = [
    "analyze_fundamental_input",
    "create_fundamental_agent",
    "FUNDAMENTAL_SYSTEM_PROMPT",
    "FinancialMetric",
    "FundamentalAgent",
    "FundamentalInput",
    "FundamentalOutput",
    "FundamentalWorkflowResult",
    "build_fundamental_prompt",
    "extract_guidance_text",
    "get_clean_statement",
    "get_transcript_from_fmp",
    "get_yfinance_fundamental_input",
    "run_fundamental_analysis",
    "run_fundamental_workflow",
    "safe_get_statement_value",
]
