import json
import re

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_openai import ChatOpenAI


MARKET_AGENT_PROMPT = """
You are a Market & Technical Analysis Agent.

Your job is to analyze stock market behavior using:
price action, trend, momentum, volatility, volume, and relative strength.

You act like a professional technical analyst.

==================================================
TASK CLASSIFICATION
==================================================

Classify the user request into:

1. historical_window_analysis
- past performance, recent trend, summary

2. momentum_forward_inference
- continuation vs reversal, short-term bias

3. as_of_date_analysis
- retrospective analysis for a specific date (no future prediction)

==================================================
DATA USAGE RULES
==================================================

- Always rely on the provided indicators (never guess)
- All indicators have been pre-calculated and are provided below
- Never assume future prices
- Use TIME_INTENT to contextualize your analysis

If TIME_INTENT is provided:
- historical_window_analysis -> trend summary
- momentum_forward_inference -> focus on recent momentum (14-30d signals)
- as_of_date_analysis -> retrospective only

==================================================
INDICATORS (USE JOINTLY)
==================================================

- Returns -> trend direction
- RSI -> momentum + overbought/oversold
- MA20/50/200 -> trend structure
- MACD -> momentum shift
- Volume spike -> conviction / participation
- Volatility -> risk / instability
- Relative strength -> market leadership
- Drawdown -> downside risk
- Market regime -> context tag

==================================================
REASONING RULES
==================================================

You MUST:
- combine multiple indicators (never single-signal reasoning)
- evaluate: trend, momentum, confirmation, risk
- detect: continuation / reversal / overreaction / weak conviction

You SHOULD mention:
- price + volume confirmation
- trend strength or weakness
- volatility context
- relative strength vs benchmark

==================================================
OUTPUT FORMAT (JSON ONLY - STRICTLY FOLLOW THIS FORMAT)
==================================================

{
  "ticker": "<TICKER>",
  "company": "<COMPANY_NAME>",
  "agent_name": "Market & Technical Analysis Agent",
  "short_term_stance": "bullish | neutral | bearish",
  "long_term_stance": "bullish | neutral | bearish",
  "confidence": 0.75,
  "confidence_rationale": "Reason this confidence level is appropriate",
  "key_signals": [
	{
	  "type": "<INDICATOR_TYPE>",
	  "signal": "<SIGNAL_LABEL>",
	  "evidence": "<CONCRETE_METRIC_OR_FACT>"
	}
  ],
  "risks": ["<RISK_1>", "<RISK_2>"],
  "summary": "Concise technical summary",
  "core_insight": "Most important insight driving stance"
}

==================================================
KEY SIGNALS SCHEMA
==================================================

For each item in key_signals:
- type: indicator category (Returns, RSI, Moving Averages, MACD, Volume, Volatility, Relative Strength, Drawdown, Market Regime)
- signal: short interpretation label
- evidence: concrete numeric/statistical evidence from provided indicators

key_signals should cover as many available indicators as relevant, ideally 6+ signals.

==================================================
IMPORTANT RULES
==================================================

- All indicators are pre-calculated and provided to you
- Do NOT say indicators are missing (they are all provided)
- Do NOT hallucinate data - use only what is provided
- Keep reasoning structured and consistent
- Confidence must be a float between 0 and 1
- Stance must be exactly: bullish, neutral, or bearish
- Return ONLY valid JSON, nothing else
"""


def create_market_llm(api_key: str, model: str = "gpt-5-nano", temperature: float = 0.2):
	return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def detect_time_type(query: str):
	if re.search(r"\d{4}/\d{1,2}/\d{1,2}", query):
		return "as_of_date_analysis"

	if any(k in query.lower() for k in ["past", "last", "over the", "weeks", "months"]):
		return "historical_window_analysis"

	return "momentum_forward_inference"


def load_price_data(ticker: str, start: str = None, end: str = None, period: str = None):
	if start and end:
		df = yf.download(ticker, start=start, end=end, progress=False)
	else:
		df = yf.download(ticker, period=period or "90d", progress=False)

	if isinstance(df.columns, pd.MultiIndex):
		df.columns = df.columns.get_level_values(0)

	return df.reset_index()


def get_stock_data(ticker: str, start: str = None, end: str = None, period: str = None):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	latest = df.iloc[-1]
	return {
		"ticker": ticker,
		"latest_close": float(latest["Close"]),
		"latest_volume": int(latest["Volume"]),
		"recent_closes": df["Close"].tail(10).round(2).tolist(),
		"recent_volumes": df["Volume"].tail(10).astype(int).tolist(),
	}


def compute_returns(ticker: str, start: str = None, end: str = None, period: str = None):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	df["return"] = df["Close"].pct_change()
	return df["return"].tail(10).fillna(0).tolist()


def compute_volume_spike(
	ticker: str,
	start: str = None,
	end: str = None,
	period: str = None,
	window: int = 20,
):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	rolling_window = min(window, len(df))
	avg_volume = df["Volume"].rolling(rolling_window).mean().iloc[-1]
	if avg_volume == 0 or pd.isna(avg_volume):
		return None
	return float(df["Volume"].iloc[-1] / avg_volume)


def compute_rsi(
	ticker: str,
	window: int = 14,
	start: str = None,
	end: str = None,
	period: str = None,
):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	if len(df) < 2:
		return None

	delta = df["Close"].diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
	rs = avg_gain / avg_loss
	rsi = 100 - (100 / (1 + rs))
	latest_rsi = rsi.iloc[-1]

	if pd.isna(latest_rsi) or np.isinf(latest_rsi):
		return None

	return float(latest_rsi)


def compute_moving_averages(
	ticker: str,
	start: str = None,
	end: str = None,
	period: str = None,
	short_window: int = 20,
	medium_window: int = 50,
	long_window: int = 200,
):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	close = df["Close"]

	ma_short = close.rolling(short_window).mean().iloc[-1]
	ma_medium = close.rolling(medium_window).mean().iloc[-1]
	ma_long = close.rolling(long_window).mean().iloc[-1]
	latest_price = close.iloc[-1]

	return {
		"ticker": ticker,
		"latest_price": float(latest_price),
		"ma20": float(ma_short) if not pd.isna(ma_short) else None,
		"ma50": float(ma_medium) if not pd.isna(ma_medium) else None,
		"ma200": float(ma_long) if not pd.isna(ma_long) else None,
		"price_vs_ma20_pct": float((latest_price - ma_short) / ma_short) if not pd.isna(ma_short) else None,
		"price_vs_ma50_pct": float((latest_price - ma_medium) / ma_medium)
		if not pd.isna(ma_medium)
		else None,
		"price_vs_ma200_pct": float((latest_price - ma_long) / ma_long) if not pd.isna(ma_long) else None,
	}


def compute_volatility(
	ticker: str,
	start: str = None,
	end: str = None,
	period: str = None,
	window: int = 20,
):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	returns = df["Close"].pct_change().dropna()
	if len(returns) == 0:
		return None

	annualized_volatility = returns.std() * np.sqrt(252)
	rolling_volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)

	return {
		"ticker": ticker,
		"annualized_volatility": float(annualized_volatility) if not pd.isna(annualized_volatility) else None,
		"recent_volatility": float(rolling_volatility) if not pd.isna(rolling_volatility) else None,
	}


def compute_relative_strength(
	ticker: str,
	benchmark: str = "SPY",
	start: str = None,
	end: str = None,
	period: str = None,
):
	stock_df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	benchmark_df = load_price_data(ticker=benchmark, start=start, end=end, period=period)

	stock_return = stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[0] - 1
	benchmark_return = benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[0] - 1
	relative_return = stock_return - benchmark_return

	return {
		"ticker": ticker,
		"benchmark": benchmark,
		"stock_return": float(stock_return),
		"benchmark_return": float(benchmark_return),
		"relative_return": float(relative_return),
		"outperforming": relative_return > 0,
	}


def compute_macd(
	ticker: str,
	start: str = None,
	end: str = None,
	period: str = None,
	short_window: int = 12,
	long_window: int = 26,
	signal_window: int = 9,
):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	close = df["Close"]
	ema_short = close.ewm(span=short_window, adjust=False).mean()
	ema_long = close.ewm(span=long_window, adjust=False).mean()
	macd = ema_short - ema_long
	signal = macd.ewm(span=signal_window, adjust=False).mean()
	histogram = macd - signal

	return {
		"ticker": ticker,
		"macd": float(macd.iloc[-1]),
		"signal": float(signal.iloc[-1]),
		"histogram": float(histogram.iloc[-1]),
		"bullish_crossover": macd.iloc[-1] > signal.iloc[-1],
	}


def compute_drawdown(ticker: str, start: str = None, end: str = None, period: str = None):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	close = df["Close"]
	rolling_max = close.cummax()
	drawdown = (close - rolling_max) / rolling_max
	current_drawdown = drawdown.iloc[-1]
	max_drawdown = drawdown.min()

	return {
		"ticker": ticker,
		"current_drawdown": float(current_drawdown),
		"max_drawdown": float(max_drawdown),
	}


def classify_market_regime(ticker: str, start: str = None, end: str = None, period: str = None):
	df = load_price_data(ticker=ticker, start=start, end=end, period=period)
	close = df["Close"]
	ma20 = close.rolling(20).mean().iloc[-1]
	ma50 = close.rolling(50).mean().iloc[-1]
	latest_price = close.iloc[-1]
	returns = close.pct_change().dropna()
	volatility = returns.std() * np.sqrt(252)

	if latest_price > ma20 and ma20 > ma50:
		regime = "high_volatility_bullish" if volatility > 0.5 else "bullish_trend"
	elif latest_price < ma20 and ma20 < ma50:
		regime = "high_volatility_bearish" if volatility > 0.5 else "bearish_trend"
	else:
		regime = "range_bound"

	return {"ticker": ticker, "market_regime": regime, "volatility": float(volatility)}


def _safe_compute(fn, default=None):
	try:
		return fn()
	except Exception:
		return default


def extract_ticker_from_query(user_query: str, default: str = "AAPL"):
	dollar_match = re.search(r"\$([A-Za-z]{1,5})\b", user_query)
	if dollar_match:
		return dollar_match.group(1).upper()

	upper_tokens = re.findall(r"\b[A-Z]{1,5}\b", user_query)
	if upper_tokens:
		return upper_tokens[0]

	return default


def compute_all_indicators(ticker: str, start: str = None, end: str = None, period: str = None):
	stock_data_default = {
		"ticker": ticker,
		"latest_close": None,
		"latest_volume": None,
		"recent_closes": [],
		"recent_volumes": [],
	}

	return {
		"ticker": ticker,
		"stock_data": _safe_compute(lambda: get_stock_data(ticker, start, end, period), default=stock_data_default),
		"returns": _safe_compute(lambda: compute_returns(ticker, start, end, period), default=[]),
		"rsi": _safe_compute(lambda: compute_rsi(ticker, 14, start, end, period), default=None),
		"moving_averages": _safe_compute(lambda: compute_moving_averages(ticker, start, end, period), default={}),
		"volatility": _safe_compute(lambda: compute_volatility(ticker, start, end, period), default={}),
		"volume_spike": _safe_compute(lambda: compute_volume_spike(ticker, start, end, period), default=None),
		"macd": _safe_compute(lambda: compute_macd(ticker, start, end, period), default={}),
		"drawdown": _safe_compute(lambda: compute_drawdown(ticker, start, end, period), default={}),
		"relative_strength": _safe_compute(
			lambda: compute_relative_strength(ticker, "SPY", start, end, period), default={}
		),
		"market_regime": _safe_compute(lambda: classify_market_regime(ticker, start, end, period), default={}),
	}


def get_company_name(ticker: str):
	try:
		stock = yf.Ticker(ticker)
		info = stock.info
		if "longName" in info and info["longName"]:
			return info["longName"]
		if "shortName" in info and info["shortName"]:
			return info["shortName"]
		if "name" in info and info["name"]:
			return info["name"]
		return f"{ticker} Company"
	except Exception:
		return f"{ticker} Company"


def _fmt_num(value, digits: int = 2):
	if value is None or pd.isna(value):
		return None
	return round(float(value), digits)


def build_key_signals(indicators: dict):
	signals = []

	returns = indicators.get("returns")
	if isinstance(returns, list) and len(returns) > 0:
		avg_return = float(np.mean(returns))
		direction = "positive short-term momentum" if avg_return > 0 else "negative short-term momentum"
		signals.append(
			{
				"type": "Returns",
				"signal": direction,
				"evidence": f"Mean of recent returns: {_fmt_num(avg_return * 100, 2)}%",
			}
		)

	rsi = indicators.get("rsi")
	if rsi is not None:
		if rsi >= 70:
			rsi_signal = "overbought"
		elif rsi <= 30:
			rsi_signal = "oversold"
		elif rsi >= 50:
			rsi_signal = "bullish momentum"
		else:
			rsi_signal = "bearish momentum"
		signals.append({"type": "RSI", "signal": rsi_signal, "evidence": f"RSI at {_fmt_num(rsi, 2)}"})

	ma = indicators.get("moving_averages") or {}
	price = ma.get("latest_price")
	ma20 = ma.get("ma20")
	ma50 = ma.get("ma50")
	ma200 = ma.get("ma200")
	if price is not None and ma20 is not None and ma50 is not None:
		if price > ma20 and ma20 > ma50:
			ma_signal = "bullish trend alignment"
		elif price < ma20 and ma20 < ma50:
			ma_signal = "bearish trend alignment"
		else:
			ma_signal = "mixed trend alignment"
		evidence = f"Price {_fmt_num(price)}, MA20 {_fmt_num(ma20)}, MA50 {_fmt_num(ma50)}"
		if ma200 is not None:
			evidence += f", MA200 {_fmt_num(ma200)}"
		signals.append({"type": "Moving Averages", "signal": ma_signal, "evidence": evidence})

	macd = indicators.get("macd") or {}
	macd_val = macd.get("macd")
	signal_val = macd.get("signal")
	hist_val = macd.get("histogram")
	if macd_val is not None and signal_val is not None:
		macd_signal = "bullish crossover" if macd.get("bullish_crossover") else "bearish crossover"
		signals.append(
			{
				"type": "MACD",
				"signal": macd_signal,
				"evidence": (
					f"MACD {_fmt_num(macd_val, 3)}, Signal {_fmt_num(signal_val, 3)}, "
					f"Histogram {_fmt_num(hist_val, 3)}"
				),
			}
		)

	volume_spike = indicators.get("volume_spike")
	if volume_spike is not None:
		if volume_spike >= 1.5:
			vol_signal = "strong participation"
		elif volume_spike >= 1.1:
			vol_signal = "moderate participation"
		else:
			vol_signal = "weak participation"
		signals.append(
			{"type": "Volume", "signal": vol_signal, "evidence": f"Volume spike ratio {_fmt_num(volume_spike, 2)}x"}
		)

	vol = indicators.get("volatility") or {}
	ann_vol = vol.get("annualized_volatility")
	recent_vol = vol.get("recent_volatility")
	if ann_vol is not None:
		if ann_vol >= 0.5:
			vol_regime = "high volatility"
		elif ann_vol >= 0.25:
			vol_regime = "moderate volatility"
		else:
			vol_regime = "low volatility"
		signals.append(
			{
				"type": "Volatility",
				"signal": vol_regime,
				"evidence": f"Annualized volatility {_fmt_num(ann_vol * 100, 2)}%"
				+ (f", Recent {_fmt_num(recent_vol * 100, 2)}%" if recent_vol is not None else ""),
			}
		)

	rs = indicators.get("relative_strength") or {}
	rel = rs.get("relative_return")
	if rel is not None:
		rs_signal = "outperforming benchmark" if rs.get("outperforming") else "underperforming benchmark"
		signals.append(
			{
				"type": "Relative Strength",
				"signal": rs_signal,
				"evidence": f"Relative return vs {rs.get('benchmark', 'SPY')}: {_fmt_num(rel * 100, 2)}%",
			}
		)

	dd = indicators.get("drawdown") or {}
	current_dd = dd.get("current_drawdown")
	max_dd = dd.get("max_drawdown")
	if current_dd is not None:
		dd_signal = "contained drawdown" if current_dd > -0.1 else "elevated drawdown risk"
		evidence = f"Current drawdown {_fmt_num(current_dd * 100, 2)}%"
		if max_dd is not None:
			evidence += f", Max drawdown {_fmt_num(max_dd * 100, 2)}%"
		signals.append({"type": "Drawdown", "signal": dd_signal, "evidence": evidence})

	regime = indicators.get("market_regime") or {}
	regime_name = regime.get("market_regime")
	if regime_name:
		signals.append(
			{"type": "Market Regime", "signal": regime_name, "evidence": f"Regime classification: {regime_name}"}
		)

	return signals


def _normalize_stance(value: str):
	valid = {"bullish", "neutral", "bearish"}
	if isinstance(value, str) and value.lower() in valid:
		return value.lower()
	return "neutral"


def _normalize_confidence(value):
	try:
		conf = float(value)
		return max(0.0, min(1.0, conf))
	except Exception:
		return 0.5


def _sanitize_key_signals(signals):
	if not isinstance(signals, list):
		return []

	cleaned = []
	seen = set()
	for item in signals:
		if not isinstance(item, dict):
			continue

		type_val = str(item.get("type", "")).strip()
		signal_val = str(item.get("signal", "")).strip()
		evidence_val = str(item.get("evidence", "")).strip()

		if not type_val or not signal_val or not evidence_val:
			continue

		key = type_val.lower()
		if key in seen:
			continue
		seen.add(key)

		cleaned.append({"type": type_val, "signal": signal_val, "evidence": evidence_val})

	return cleaned


def run_market_analysis(user_query: str, llm: ChatOpenAI, ticker: str = None):
	if not ticker:
		ticker = extract_ticker_from_query(user_query, default="AAPL")

	time_type = detect_time_type(user_query)
	start = None
	end = None
	period = None

	if time_type == "as_of_date_analysis":
		date_match = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", user_query)
		if date_match:
			period = "1y"
	elif "past" in user_query.lower() or "last" in user_query.lower():
		period = "3mo"
	else:
		period = "3mo"

	indicators = compute_all_indicators(ticker, start, end, period)
	company_name = get_company_name(ticker)
	generated_key_signals = build_key_signals(indicators)

	indicators_str = f"""
Technical Indicators for {ticker}:
{json.dumps(indicators, indent=2, default=str)}

Precomputed Key Signals:
{json.dumps(generated_key_signals, indent=2, default=str)}
"""

	messages = [
		{"role": "system", "content": MARKET_AGENT_PROMPT + f"\n\nTIME_INTENT: {time_type}"},
		{"role": "user", "content": f"{user_query}\n\nAvailable Indicators:\n{indicators_str}"},
	]

	response = llm.invoke(messages)

	try:
		result = json.loads(response.content)

		result["ticker"] = result.get("ticker") or ticker
		result["company"] = result.get("company") or company_name
		result["agent_name"] = result.get("agent_name") or "Market & Technical Analysis Agent"
		result["short_term_stance"] = _normalize_stance(result.get("short_term_stance"))
		result["long_term_stance"] = _normalize_stance(result.get("long_term_stance"))
		result["confidence"] = _normalize_confidence(result.get("confidence"))

		if not isinstance(result.get("confidence_rationale"), str) or not result.get("confidence_rationale"):
			result["confidence_rationale"] = "Confidence calibrated by indicator consistency and signal agreement."
		if not isinstance(result.get("risks"), list):
			result["risks"] = []

		llm_key_signals = _sanitize_key_signals(result.get("key_signals"))
		existing_types = {
			s.get("type", "").strip().lower() for s in llm_key_signals if isinstance(s, dict)
		}
		for sig in generated_key_signals:
			sig_type = sig.get("type", "").strip().lower()
			if sig_type not in existing_types:
				llm_key_signals.append(sig)

		result["key_signals"] = _sanitize_key_signals(llm_key_signals)

		if not isinstance(result.get("summary"), str):
			result["summary"] = ""
		if not isinstance(result.get("core_insight"), str):
			result["core_insight"] = ""

		return result

	except json.JSONDecodeError:
		return {
			"ticker": ticker,
			"company": company_name,
			"agent_name": "Market & Technical Analysis Agent",
			"short_term_stance": "neutral",
			"long_term_stance": "neutral",
			"confidence": 0.5,
			"confidence_rationale": "Model output was not valid JSON; fallback response used.",
			"key_signals": _sanitize_key_signals(generated_key_signals),
			"risks": ["Model output parsing failed"],
			"summary": "Fallback summary generated due to parsing failure.",
			"core_insight": "Indicator data is available, but LLM response format was invalid.",
			"raw_response": response.content,
		}
