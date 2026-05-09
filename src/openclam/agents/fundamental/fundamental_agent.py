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
    transcript_status: dict[str, Any] = field(default_factory=dict)


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
    transcript_status: dict[str, Any] = field(default_factory=dict)

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
            transcript_status=payload.get("transcript_status") if isinstance(payload.get("transcript_status"), dict) else {},
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
    as_of_date: Any | None = None,
) -> pd.DataFrame | None:
    if statement is None or statement.empty:
        return None

    cleaned = statement.copy()
    try:
        cleaned = cleaned.loc[:, sorted(cleaned.columns, key=lambda col: pd.Timestamp(col), reverse=True)]
    except Exception:
        pass

    if as_of_date is not None:
        try:
            cutoff = pd.Timestamp(as_of_date).normalize()
            eligible_cols = []
            for col in cleaned.columns:
                try:
                    col_ts = pd.Timestamp(col).normalize()
                except Exception:
                    continue
                if col_ts <= cutoff:
                    eligible_cols.append(col)
            cleaned = cleaned.loc[:, eligible_cols]
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


def _has_real_transcript(snippets: Sequence[str] | None) -> bool:
    if not snippets:
        return False
    placeholder_prefixes = (
        "fmp_api_key is not set",
        "no earnings call transcript available",
        "transcript fetch skipped",
    )
    for snippet in snippets:
        text = str(snippet or "").strip().lower()
        if text and not text.startswith(placeholder_prefixes):
            return True
    return False


def _has_guidance_text(guidance_text: str | None) -> bool:
    text = str(guidance_text or "").strip().lower()
    if not text:
        return False
    unavailable_markers = (
        "no guidance information available",
        "fmp_api_key is not set",
        "transcript available, but no explicit guidance",
    )
    return not text.startswith(unavailable_markers)


def _calibrate_fundamental_confidence(output: FundamentalOutput, data: FundamentalInput) -> FundamentalOutput:
    """Lightly calibrate model confidence to the actual evidence packet quality."""
    available_core_metrics = sum(
        1 for metric in data.financial_metrics[:8] if metric.current_value is not None
    )
    has_transcript = _has_real_transcript(data.earnings_call_snippets)
    has_guidance = _has_guidance_text(data.guidance_text)
    has_consensus = any(metric.consensus_estimate is not None for metric in data.financial_metrics)

    evidence_floor = 0.0
    evidence_notes: list[str] = []
    if available_core_metrics >= 5:
        evidence_floor = max(evidence_floor, 0.4)
        evidence_notes.append("core financial statement metrics are available")
    if has_transcript:
        evidence_floor = max(evidence_floor, 0.55)
        evidence_notes.append("FMP earnings-call transcript snippets are available")
    if has_guidance:
        evidence_floor = max(evidence_floor, 0.6)
        evidence_notes.append("forward-looking guidance text was extracted")
    if has_consensus:
        evidence_floor = max(evidence_floor, 0.7)
        evidence_notes.append("consensus estimates are available")

    if evidence_floor > output.confidence:
        output.confidence = round(evidence_floor, 2)
        calibration_note = "Confidence calibrated upward because " + ", ".join(evidence_notes) + "."
        output.confidence_reasoning = (
            f"{output.confidence_reasoning} {calibration_note}".strip()
            if output.confidence_reasoning
            else calibration_note
        )
    return output


def get_yfinance_fundamental_input(
    ticker: str,
    transcript_year: int | None = None,
    transcript_quarter: int | None = None,
    require_transcript: bool = False,
    as_of_date: str | None = None,
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
        as_of_date=as_of_date,
    )
    cashflow = get_clean_statement(
        stock.quarterly_cashflow,
        required_rows=["Operating Cash Flow", "Free Cash Flow"],
        as_of_date=as_of_date,
    )
    balance_sheet = get_clean_statement(stock.quarterly_balance_sheet, as_of_date=as_of_date)

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

    historical_backtest = as_of_date is not None
    if historical_backtest:
        market_cap = None
        trailing_pe = None
        forward_pe = None
        profit_margins = None
        revenue_growth = None
        earnings_growth = None
        info_snapshot_note = (
            "Point-in-time valuation and growth snapshot fields from yfinance info were omitted "
            "to avoid forward-looking leakage in historical evaluation."
        )
    else:
        market_cap = _safe_float(info.get("marketCap"))
        trailing_pe = _safe_float(info.get("trailingPE"))
        forward_pe = _safe_float(info.get("forwardPE"))
        profit_margins = _safe_float(info.get("profitMargins"))
        revenue_growth = _safe_float(info.get("revenueGrowth"))
        earnings_growth = _safe_float(info.get("earningsGrowth"))
        info_snapshot_note = "Using current yfinance info snapshot fields."

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
    Info Snapshot Note: {info_snapshot_note}

    Do not describe QoQ changes as YoY unless the field explicitly says YoY.
    """

    earnings_call_snippets: list[str]
    guidance_text: str
    transcript_status: dict[str, Any] = {
        "requested": transcript_year is not None and transcript_quarter is not None,
        "requested_year": transcript_year,
        "requested_quarter": transcript_quarter,
        "fmp_key_loaded": bool(os.getenv("FMP_API_KEY")),
        "loaded": False,
        "chunks": 0,
        "guidance_extracted": False,
        "source": "Financial Modeling Prep",
        "message": "",
    }
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
            transcript_status.update(
                {
                    "loaded": True,
                    "chunks": len(transcript_snippets),
                    "guidance_extracted": _has_guidance_text(guidance_text),
                    "message": "FMP transcript snippets loaded.",
                }
            )
        elif not os.getenv("FMP_API_KEY"):
            if require_transcript:
                raise ValueError(
                    "FMP_API_KEY is not set, so the earnings call transcript could not be retrieved."
                )
            earnings_call_snippets = [
                "FMP_API_KEY is not set, so earnings call transcript retrieval was skipped.",
            ]
            guidance_text = "FMP_API_KEY is not set, so guidance information could not be retrieved."
            transcript_status["message"] = "FMP_API_KEY is not set."
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
            transcript_status["message"] = (
                f"FMP returned no transcript for {normalized_ticker} "
                f"year={transcript_year}, quarter={transcript_quarter}."
            )
    else:
        earnings_call_snippets = [
            "Transcript fetch skipped. Provide transcript_year and transcript_quarter to load FMP transcript data.",
        ]
        guidance_text = "No guidance information available."
        transcript_status["message"] = "Transcript fetch skipped because no transcript period was provided."

    return FundamentalInput(
        ticker=normalized_ticker,
        company_name=company_name,
        quarter=latest_quarter,
        earnings_summary=earnings_summary,
        financial_metrics=metrics,
        earnings_call_snippets=earnings_call_snippets,
        guidance_text=guidance_text,
        transcript_status=transcript_status,
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
    output = _calibrate_fundamental_confidence(active_agent.run(data), data)
    output.transcript_status = data.transcript_status
    return output


def run_fundamental_analysis(
    ticker: str,
    transcript_year: int | None = None,
    transcript_quarter: int | None = None,
    require_transcript: bool = False,
    as_of_date: str | None = None,
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
        as_of_date=as_of_date,
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
    as_of_date: str | None = None,
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
        as_of_date=as_of_date,
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


def mag7_q4_2025_earnings_df():
    """Return the shared Mag 7 Q4 2025 earnings case study used in other notebooks."""
    from openclam.agents.news_macro.news_macro_agent import mag7_q4_2025_earnings_df as _source

    return _source()


def build_earnings_price_eval(
    earnings_df=None,
    pre_trading_days: int = 7,
    post_trading_days: int = 7,
    long_post_trading_days: int = 30,
    benchmarks: tuple[str, ...] = ("SPY", "QQQ"),
    price_anchor: str = "event_close",
):
    """Reuse the shared event-window pricing helper for the fundamental notebook."""
    from openclam.agents.news_macro.news_macro_agent import build_earnings_price_eval as _source

    return _source(
        earnings_df=earnings_df,
        pre_trading_days=pre_trading_days,
        post_trading_days=post_trading_days,
        long_post_trading_days=long_post_trading_days,
        benchmarks=benchmarks,
        price_anchor=price_anchor,
    )


def _normalize_eval_stance(stance: str | None) -> str:
    stance_text = str(stance or "").strip().lower()
    if stance_text == "bullish":
        return "Bullish"
    if stance_text == "bearish":
        return "Bearish"
    if stance_text in {"neutral", "watch"}:
        return "Neutral"
    return "Neutral"


def _infer_short_term_stance(report: FundamentalOutput) -> str:
    """Infer an event-driven stance from guidance, tone, and surprise fields."""
    score = 0

    guidance = str(report.guidance_change or "").strip().lower()
    if guidance == "raise":
        score += 2
    elif guidance == "cut":
        score -= 2

    beat = str(report.beat_or_miss or "").strip().lower()
    if beat == "beat":
        score += 1
    elif beat == "miss":
        score -= 1

    tone = str(report.management_tone or "").strip().lower()
    if any(token in tone for token in ("optimistic", "positive", "constructive", "confident", "bullish")):
        score += 1
    elif any(token in tone for token in ("cautious", "negative", "weak", "concern", "bearish")):
        score -= 1

    thesis_impact = str(report.thesis_impact or "").strip().lower()
    if thesis_impact == "strengthened":
        score += 1
    elif thesis_impact == "weakened":
        score -= 1

    long_stance = _normalize_eval_stance(report.stance)
    if long_stance == "Bullish":
        score += 1
    elif long_stance == "Bearish":
        score -= 1

    signal_balance = len(report.positive_signals or []) - len(report.negative_signals or [])
    if signal_balance >= 2:
        score += 1
    elif signal_balance <= -2:
        score -= 1

    if score >= 2:
        return "Bullish"
    if score <= -2:
        return "Bearish"
    return "Neutral"


def _stance_to_direction(stance: str | None) -> str | None:
    normalized = _normalize_eval_stance(stance)
    if normalized == "Bullish":
        return "up"
    if normalized == "Bearish":
        return "down"
    return None


def _direction_match(
    predicted_direction: str | None,
    realized_direction: str | None,
    abnormal_return: Any,
    pd_module,
    neutral_band: float = 0.02,
):
    if predicted_direction is None or realized_direction is None:
        return None
    if pd_module.isna(abnormal_return):
        return None
    if abs(abnormal_return) < neutral_band:
        return None
    actual_direction = "up" if abnormal_return > 0 else "down"
    return predicted_direction == actual_direction


def _direction_match_reason(
    predicted_direction: str | None,
    realized_direction: str | None,
    abnormal_return: Any,
    pd_module,
    neutral_band: float = 0.02,
) -> str:
    if predicted_direction is None or realized_direction is None:
        return "prediction or realization missing"
    if pd_module.isna(abnormal_return):
        return "abnormal return not available"
    if abs(abnormal_return) < neutral_band:
        return f"moved within neutral band ({neutral_band * 100:.1f}%)"
    actual_direction = "up" if abnormal_return > 0 else "down"
    if predicted_direction == actual_direction:
        return f"correctly predicted {actual_direction} move ({abnormal_return * 100:.2f}%)"
    return f"predicted {predicted_direction} but realized {actual_direction} ({abnormal_return * 100:.2f}%)"


def run_agent_event_window_eval(
    summary_df,
    transcript_year: int | None = 2025,
    transcript_quarter: int | None = 4,
    require_transcript: bool = False,
    agent: FundamentalAgent | None = None,
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    client: Any | None = None,
    neutral_band: float = 0.02,
    long_post_trading_days: int | None = None,
):
    """Run the fundamental agent on an earnings event set and score both horizons."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), {}

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
    agent_eval = summary_df[[column for column in required if column in summary_df.columns]].copy()

    agent_eval["news_context_ready"] = agent_eval["earnings_date"].notna()
    agent_eval["report_ready"] = False
    agent_eval["agent_short_term_stance"] = None
    agent_eval["agent_long_term_stance"] = None
    agent_eval["agent_stance"] = None
    agent_eval["agent_confidence"] = None
    agent_eval["agent_guidance_change"] = None
    agent_eval["agent_management_tone"] = None
    agent_eval["agent_thesis_impact"] = None
    agent_eval["agent_beat_or_miss"] = None
    agent_eval["confidence_rationale"] = None
    agent_eval["stance_rationale"] = None
    agent_eval["short_direction_match"] = None
    agent_eval["short_direction_match_reason"] = None
    agent_eval["long_direction_match"] = None
    agent_eval["long_direction_match_reason"] = None
    agent_eval["direction_match"] = None
    agent_eval["direction_match_reason"] = None
    agent_reports: dict[str, FundamentalOutput] = {}

    active_agent = agent or create_fundamental_agent(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        client=client,
    )

    for idx, row in agent_eval.iterrows():
        if not row["news_context_ready"]:
            agent_eval.loc[idx, "short_direction_match_reason"] = "missing earnings date"
            agent_eval.loc[idx, "long_direction_match_reason"] = "missing earnings date"
            agent_eval.loc[idx, "direction_match_reason"] = "missing earnings date"
            continue

        ticker = str(row["ticker"]).upper()
        try:
            report = run_fundamental_analysis(
                ticker=ticker,
                transcript_year=transcript_year,
                transcript_quarter=transcript_quarter,
                require_transcript=require_transcript,
                as_of_date=str(row["earnings_date"]),
                agent=active_agent,
            )
            agent_reports[ticker] = report

            short_term_stance = _infer_short_term_stance(report)
            long_term_stance = _normalize_eval_stance(report.stance)
            short_predicted = _stance_to_direction(short_term_stance)
            long_predicted = _stance_to_direction(long_term_stance)

            short_realized = row.get("realized_direction_vs_qqq")
            long_realized = row.get(long_direction_col)
            short_abnormal = row.get("abnormal_vs_qqq")
            long_abnormal = row.get(long_abnormal_col)

            agent_eval.loc[idx, "report_ready"] = True
            agent_eval.loc[idx, "agent_short_term_stance"] = short_term_stance
            agent_eval.loc[idx, "agent_long_term_stance"] = long_term_stance
            agent_eval.loc[idx, "agent_stance"] = f"ST: {short_term_stance}; LT: {long_term_stance}"
            agent_eval.loc[idx, "agent_confidence"] = report.confidence
            agent_eval.loc[idx, "agent_guidance_change"] = report.guidance_change
            agent_eval.loc[idx, "agent_management_tone"] = report.management_tone
            agent_eval.loc[idx, "agent_thesis_impact"] = report.thesis_impact
            agent_eval.loc[idx, "agent_beat_or_miss"] = report.beat_or_miss
            agent_eval.loc[idx, "confidence_rationale"] = report.confidence_reasoning
            agent_eval.loc[idx, "stance_rationale"] = report.core_judgment or report.thesis_impact_reasoning

            short_match = _direction_match(short_predicted, short_realized, short_abnormal, pd, neutral_band)
            long_match = _direction_match(long_predicted, long_realized, long_abnormal, pd, neutral_band)
            short_reason = _direction_match_reason(short_predicted, short_realized, short_abnormal, pd, neutral_band)
            long_reason = _direction_match_reason(long_predicted, long_realized, long_abnormal, pd, neutral_band)

            agent_eval.loc[idx, "short_direction_match"] = short_match
            agent_eval.loc[idx, "short_direction_match_reason"] = short_reason
            agent_eval.loc[idx, "long_direction_match"] = long_match
            agent_eval.loc[idx, "long_direction_match_reason"] = long_reason
            agent_eval.loc[idx, "direction_match"] = short_match
            agent_eval.loc[idx, "direction_match_reason"] = short_reason
        except Exception as exc:
            error_text = f"agent error: {exc}"
            agent_eval.loc[idx, "confidence_rationale"] = error_text
            agent_eval.loc[idx, "short_direction_match_reason"] = error_text
            agent_eval.loc[idx, "long_direction_match_reason"] = error_text
            agent_eval.loc[idx, "direction_match_reason"] = error_text

    return agent_eval, agent_reports


def summarize_eval_results(agent_eval):
    """Summarize short/long stance evaluation into presentation-friendly metrics."""

    def _metric(prefix: str) -> dict[str, Any]:
        match_col = f"{prefix}_direction_match"
        reason_col = f"{prefix}_direction_match_reason"
        stance_col = f"agent_{prefix}_term_stance"
        evaluable = agent_eval[agent_eval[match_col].notna()] if match_col in agent_eval else pd.DataFrame()
        matched = int((evaluable[match_col] == True).sum()) if not evaluable.empty else 0
        total = int(len(evaluable))
        neutral_rate = None
        if stance_col in agent_eval:
            neutral_rate = float((agent_eval[stance_col] == "Neutral").mean())
        reason_counts = agent_eval[reason_col].value_counts(dropna=False).to_dict() if reason_col in agent_eval else {}
        return {
            f"{prefix}_evaluable_cases": total,
            f"{prefix}_matched_cases": matched,
            f"{prefix}_missed_cases": total - matched,
            f"{prefix}_accuracy": matched / total if total else None,
            f"{prefix}_neutral_rate": neutral_rate,
            f"{prefix}_not_evaluable_cases": int(agent_eval[match_col].isna().sum()) if match_col in agent_eval else None,
            f"{prefix}_reason_counts": reason_counts,
        }

    return {
        "cases": int(len(agent_eval)),
        **_metric("short"),
        **_metric("long"),
    }


__all__ = [
    "analyze_fundamental_input",
    "build_earnings_price_eval",
    "create_fundamental_agent",
    "FUNDAMENTAL_SYSTEM_PROMPT",
    "FinancialMetric",
    "FundamentalAgent",
    "FundamentalInput",
    "FundamentalOutput",
    "FundamentalWorkflowResult",
    "mag7_q4_2025_earnings_df",
    "run_agent_event_window_eval",
    "build_fundamental_prompt",
    "extract_guidance_text",
    "get_clean_statement",
    "get_transcript_from_fmp",
    "get_yfinance_fundamental_input",
    "run_fundamental_analysis",
    "run_fundamental_workflow",
    "safe_get_statement_value",
    "summarize_eval_results",
]
