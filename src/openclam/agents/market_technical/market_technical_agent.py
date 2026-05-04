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


# ==================================================
# EVALUATION FRAMEWORK FOR MARKET TECHNICAL AGENT
# ==================================================


def _import_pandas():
	return pd


def _import_numpy():
	return np


def _import_yfinance():
	return yf


def _download_close_series(yf_module, ticker: str, event_date, pre_trading_days: int, lookback_days: int):
	"""Download close prices around event date."""
	try:
		start_date = (pd.Timestamp(event_date) - pd.Timedelta(days=pre_trading_days + 30)).strftime("%Y-%m-%d")
		end_date = (pd.Timestamp(event_date) + pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
		
		df = yf_module.download(ticker, start=start_date, end=end_date, progress=False)
		if isinstance(df.columns, pd.MultiIndex):
			df.columns = df.columns.get_level_values(0)
		
		df = df.reset_index()
		if df.empty:
			return None, f"No data returned for {ticker}"
		
		df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
		df = df.sort_values("Date").reset_index(drop=True)
		
		return df, None
	except Exception as e:
		return None, str(e)


def _event_window_from_close(close_df, event_date, pre_trading_days: int, lookback_days: int):
	"""Extract event window from close series."""
	try:
		if close_df is None or close_df.empty:
			return None, "No data available"
		
		event_ts = pd.Timestamp(event_date).normalize()
		
		# Find trading days around event
		trading_dates = close_df["Date"].values
		event_idx = None
		
		# Find closest date to event
		for i, date in enumerate(trading_dates):
			if pd.Timestamp(date).normalize() >= event_ts:
				event_idx = i
				break
		
		if event_idx is None:
			event_idx = len(trading_dates) - 1
		
		pre_idx = max(0, event_idx - pre_trading_days)
		post_idx = min(len(trading_dates), event_idx + lookback_days + 1)
		
		window = close_df.iloc[pre_idx:post_idx].copy()
		if window.empty:
			return None, "Event window is empty"
		
		baseline_price = window.iloc[0]["Close"]
		window["price_return"] = (window["Close"] - baseline_price) / baseline_price * 100
		window["days_from_event"] = (window["Date"] - event_ts).dt.days
		
		return window, None
	except Exception as e:
		return None, str(e)


def _compute_event_returns(price_window, pre_trading_days: int, post_trading_days: int, np_module, long_lookback_days: int):
	"""Compute returns at different horizons from price window."""
	metrics = {}
	
	if price_window is None or price_window.empty:
		return metrics
	
	baseline_price = price_window.iloc[0]["Close"]
	
	# Short-term (7 days post event)
	short_window = price_window[price_window["days_from_event"] <= post_trading_days]
	if not short_window.empty:
		short_price = short_window.iloc[-1]["Close"]
		metrics["post_7d_return"] = float((short_price - baseline_price) / baseline_price)
	else:
		metrics["post_7d_return"] = np_module.nan
	
	# Long-term (user-defined, default 30 days)
	long_window = price_window[price_window["days_from_event"] <= long_lookback_days]
	if not long_window.empty:
		long_price = long_window.iloc[-1]["Close"]
		long_col = f"post_{long_lookback_days}d_return"
		metrics[long_col] = float((long_price - baseline_price) / baseline_price)
	else:
		long_col = f"post_{long_lookback_days}d_return"
		metrics[long_col] = np_module.nan
	
	return metrics


def _stance_to_direction(stance: str):
	"""Convert stance to direction: bullish->up, bearish->down, neutral->None."""
	stance_lower = str(stance).lower().strip()
	if stance_lower == "bullish":
		return "up"
	elif stance_lower == "bearish":
		return "down"
	else:
		return None


def _direction_match(predicted_direction, realized_direction, abnormal_return, pd_module, neutral_band: float = 0.02):
	"""Check if predicted direction matches realized direction."""
	if predicted_direction is None or realized_direction is None:
		return None
	
	if pd_module.isna(abnormal_return):
		return None
	
	# Neutral band: treat small moves as ambiguous
	if abs(abnormal_return) < neutral_band:
		return None
	
	actual_dir = "up" if abnormal_return > 0 else "down"
	return predicted_direction == actual_dir


def _direction_match_reason(predicted_direction, realized_direction, abnormal_return, pd_module, neutral_band: float = 0.02):
	"""Generate reason for direction match."""
	if predicted_direction is None or realized_direction is None:
		return "prediction or realization missing"
	
	if pd_module.isna(abnormal_return):
		return "abnormal return not available"
	
	if abs(abnormal_return) < neutral_band:
		return f"moved within neutral band ({neutral_band*100:.1f}%)"
	
	actual_dir = "up" if abnormal_return > 0 else "down"
	if predicted_direction == actual_dir:
		return f"correctly predicted {actual_dir} move ({abnormal_return*100:.2f}%)"
	else:
		return f"predicted {predicted_direction} but realized {actual_dir} ({abnormal_return*100:.2f}%)"


def build_earnings_price_eval(
	earnings_df=None,
	pre_trading_days: int = 7,
	post_trading_days: int = 7,
	long_post_trading_days: int = 30,
	benchmarks: tuple = ("SPY", "QQQ"),
):
	"""Build event-window price paths and return metrics around earnings dates."""
	earnings_df = earnings_df.copy() if earnings_df is not None else pd.DataFrame()
	
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
	llm: ChatOpenAI,
	lookback_days: int = 14,
	neutral_band: float = 0.02,
	long_post_trading_days: int = None,
	quiet: bool = True,
):
	"""Run market technical agent against event windows and score both horizons."""
	if summary_df.empty:
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
	
	# Filter to only rows with required columns
	available_cols = [col for col in required if col in summary_df.columns]
	agent_eval = summary_df[available_cols].copy()

	agent_eval["news_context_ready"] = agent_eval["earnings_date"].notna()
	agent_eval["report_ready"] = False
	agent_eval["agent_short_term_stance"] = None
	agent_eval["agent_long_term_stance"] = None
	agent_eval["agent_stance"] = None
	agent_eval["agent_confidence"] = None
	agent_eval["confidence_rationale"] = None
	agent_eval["stance_rationale"] = None
	agent_eval["short_direction_match"] = None
	agent_eval["short_direction_match_reason"] = None
	agent_eval["long_direction_match"] = None
	agent_eval["long_direction_match_reason"] = None
	agent_eval["direction_match"] = None
	agent_eval["direction_match_reason"] = None
	agent_reports = {}

	for idx, row in agent_eval.iterrows():
		if not row["news_context_ready"]:
			agent_eval.loc[idx, "report_ready"] = False
			agent_eval.loc[idx, "agent_short_term_stance"] = None
			agent_eval.loc[idx, "agent_long_term_stance"] = None
			agent_eval.loc[idx, "agent_stance"] = None
			agent_eval.loc[idx, "agent_confidence"] = None
			agent_eval.loc[idx, "confidence_rationale"] = None
			agent_eval.loc[idx, "stance_rationale"] = None
			agent_eval.loc[idx, "short_direction_match"] = None
			agent_eval.loc[idx, "short_direction_match_reason"] = "missing earnings date"
			agent_eval.loc[idx, "long_direction_match"] = None
			agent_eval.loc[idx, "long_direction_match_reason"] = "missing earnings date"
			agent_eval.loc[idx, "direction_match"] = None
			agent_eval.loc[idx, "direction_match_reason"] = "missing earnings date"
			continue

		ticker = row["ticker"]
		company = row["company"]
		earnings_date = row["earnings_date"]

		# Build query for agent analyzing as-of the earnings date
		query = f"{company} ({ticker}) technical analysis as of {earnings_date}"
		
		try:
			agent_report = run_market_analysis(query, llm=llm, ticker=ticker)
			agent_reports[ticker] = agent_report

			short_term_stance = agent_report.get("short_term_stance", "neutral")
			long_term_stance = agent_report.get("long_term_stance", "neutral")
			confidence_score = agent_report.get("confidence", 0.5)
			
			short_predicted = _stance_to_direction(short_term_stance)
			long_predicted = _stance_to_direction(long_term_stance)
			
			short_realized = row.get("realized_direction_vs_qqq")
			long_realized = row.get(long_direction_col)
			
			short_abnormal = row.get("abnormal_vs_qqq", np.nan)
			long_abnormal = row.get(long_abnormal_col, np.nan)

			agent_eval.loc[idx, "report_ready"] = True
			agent_eval.loc[idx, "agent_short_term_stance"] = short_term_stance
			agent_eval.loc[idx, "agent_long_term_stance"] = long_term_stance
			agent_eval.loc[idx, "agent_stance"] = f"ST: {short_term_stance}; LT: {long_term_stance}"
			agent_eval.loc[idx, "agent_confidence"] = confidence_score
			agent_eval.loc[idx, "confidence_rationale"] = agent_report.get("confidence_rationale", "")
			agent_eval.loc[idx, "stance_rationale"] = agent_report.get("summary", "")
			
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
		except Exception as e:
			agent_eval.loc[idx, "report_ready"] = False
			agent_eval.loc[idx, "agent_short_term_stance"] = None
			agent_eval.loc[idx, "agent_long_term_stance"] = None
			agent_eval.loc[idx, "agent_stance"] = None
			agent_eval.loc[idx, "agent_confidence"] = None
			agent_eval.loc[idx, "confidence_rationale"] = f"agent error: {str(e)}"
			agent_eval.loc[idx, "stance_rationale"] = None
			agent_eval.loc[idx, "short_direction_match"] = None
			agent_eval.loc[idx, "short_direction_match_reason"] = f"agent error: {str(e)}"
			agent_eval.loc[idx, "long_direction_match"] = None
			agent_eval.loc[idx, "long_direction_match_reason"] = f"agent error: {str(e)}"
			agent_eval.loc[idx, "direction_match"] = None
			agent_eval.loc[idx, "direction_match_reason"] = f"agent error: {str(e)}"

	return agent_eval, agent_reports


def summarize_eval_results(agent_eval):
	"""Summarize short/long stance evaluation into presentation-friendly metrics."""
	
	def _metric(prefix: str):
		match_col = f"{prefix}_direction_match"
		reason_col = f"{prefix}_direction_match_reason"
		stance_col = f"agent_{prefix}_term_stance"
		
		evaluable = agent_eval[agent_eval[match_col].notna()] if match_col in agent_eval else pd.DataFrame()
		matched = int((evaluable[match_col] == True).sum()) if not evaluable.empty else 0
		total = int(len(evaluable))
		
		neutral_rate = None
		if stance_col and stance_col in agent_eval:
			neutral_rate = float((agent_eval[stance_col] == "neutral").mean())
		
		reason_counts = {}
		if reason_col in agent_eval:
			reason_counts = agent_eval[reason_col].value_counts(dropna=False).to_dict()
		
		return {
			f"{prefix}_evaluable_cases": total,
			f"{prefix}_matched_cases": matched,
			f"{prefix}_missed_cases": total - matched,
			f"{prefix}_accuracy": matched / total if total > 0 else None,
			f"{prefix}_neutral_rate": neutral_rate,
			f"{prefix}_not_evaluable_cases": int(agent_eval[match_col].isna().sum()) if match_col in agent_eval else 0,
			f"{prefix}_reason_counts": reason_counts,
		}

	summary = {
		"cases": int(len(agent_eval)),
		**_metric("short"),
		**_metric("long"),
	}
	
	return summary
