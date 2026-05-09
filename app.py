from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(REPO_ROOT / ".env")
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "agent_outputs" / "q4_2025_ai_tech"
PRIMARY_BLUE = "#0b73d9"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def inject_ui_styles() -> None:
    st.markdown(
        f"""
        <style>
        div.stButton > button:first-child {{
            background-color: {PRIMARY_BLUE};
            border-color: {PRIMARY_BLUE};
            color: white;
            font-weight: 700;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #075cad;
            border-color: #075cad;
            color: white;
        }}
        div.stButton > button:first-child:focus {{
            box-shadow: 0 0 0 0.15rem rgba(11, 115, 217, 0.25);
            border-color: {PRIMARY_BLUE};
            color: white;
        }}
        div[data-testid="stProgress"] > div > div > div > div {{
            background-color: {PRIMARY_BLUE};
        }}
        div[data-testid="stProgress"] > div > div {{
            background-color: #e8eef6;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_tables(cache_root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(cache_root)
    price_path = root / "tables" / "price_summary.csv"
    cio_path = root / "tables" / "cio_eval_full_55.csv"
    strategy_path = root / "tables" / "agent_strategy_summary_with_external_baselines.csv"

    price_df = pd.read_csv(price_path) if price_path.exists() else pd.DataFrame()
    cio_df = pd.read_csv(cio_path) if cio_path.exists() else price_df.copy()
    strategy_df = pd.read_csv(strategy_path) if strategy_path.exists() else pd.DataFrame()
    return price_df, cio_df, strategy_df


def format_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1%}"
    except Exception:
        return "N/A"


def stance_badge(label: Any) -> str:
    text = str(label or "Neutral").strip()
    normalized = text.title()
    color = {
        "Bullish": "#0b8f4d",
        "Bearish": "#b42318",
        "Neutral": "#596579",
    }.get(normalized, "#596579")
    return (
        f"<span style='background:{color}; color:white; padding:0.20rem 0.55rem; "
        f"border-radius:999px; font-size:0.84rem; font-weight:700'>{normalized}</span>"
    )


def render_metric_row(row: pd.Series) -> None:
    cols = st.columns(4)
    cols[0].metric("7D return", format_pct(row.get("post_7d_return")))
    cols[1].metric("7D abnormal vs QQQ", format_pct(row.get("abnormal_vs_qqq")))
    cols[2].metric("30D return", format_pct(row.get("post_30d_return")))
    cols[3].metric("30D abnormal vs QQQ", format_pct(row.get("abnormal_30d_vs_qqq")))


def render_signal_list(title: str, items: Any, limit: int = 8) -> None:
    if not items:
        return
    st.write(f"**{title}:**")
    for item in list(items)[:limit]:
        if isinstance(item, dict):
            signal_type = item.get("type")
            signal = item.get("signal") or item.get("title") or item.get("name")
            evidence = item.get("evidence") or item.get("summary") or item.get("interpretation")
            impact = item.get("impact_score")
            heading = " | ".join(str(part) for part in [signal_type, signal] if part)
            if heading:
                st.markdown(f"**{heading}**")
            if evidence:
                st.write(evidence)
            if impact is not None:
                st.caption(f"Impact score: {impact}")
        else:
            st.write(f"- {item}")


def render_readable_block(title: str, value: Any) -> None:
    if not value:
        return
    st.write(f"**{title}:**")
    if isinstance(value, dict):
        for key, text in value.items():
            label = str(key).replace("_", " ").title()
            st.markdown(f"**{label}**")
            st.write(text)
    elif isinstance(value, list):
        for item in value:
            st.write(f"- {item}")
    else:
        st.write(value)


def render_citations(payload: dict[str, Any]) -> None:
    citations = payload.get("citations") or []
    if not citations:
        return
    st.write("**Referenced news sources:**")
    for item in citations:
        title = str(item.get("title") or "Untitled source")
        source = str(item.get("source") or "Unknown source")
        url = str(item.get("url") or "").strip()
        linked_title = f"[{title}]({url})" if url else title
        st.markdown(f"- {linked_title}  \n  <span style='color:#596579'>{source}</span>", unsafe_allow_html=True)


def render_debate(cio_payload: dict[str, Any]) -> None:
    debate = cio_payload.get("debate") or {}
    trigger = debate.get("trigger") or {}
    st.write(f"**Triggered:** {debate.get('triggered', False)}")
    st.write(f"**Conflict level:** {trigger.get('conflict_level', 'N/A')}")
    reasons = trigger.get("debate_reason") or []
    if reasons:
        st.write("**Debate reason:**")
        for reason in reasons:
            st.write(f"- {reason}")

    responses = debate.get("debate_responses") or []
    if responses:
        st.write("**Debate responses:**")
        for index, item in enumerate(responses, start=1):
            st.markdown(f"#### Response {index}: {item.get('response_type', 'revision')}")
            cols = st.columns(3)
            cols[0].markdown(f"Short<br>{stance_badge(item.get('revised_short_term_stance'))}", unsafe_allow_html=True)
            cols[1].markdown(f"Long<br>{stance_badge(item.get('revised_long_term_stance'))}", unsafe_allow_html=True)
            cols[2].metric("Confidence", f"{float(item.get('revised_confidence') or 0):.2f}")
            render_readable_block("Agreement", item.get("agreement"))
            render_readable_block("Disagreement", item.get("disagreement"))
            render_readable_block("Revision rationale", item.get("stance_revision_rationale"))
            evidence_needed = item.get("evidence_needed")
            if evidence_needed:
                render_readable_block("Evidence needed", evidence_needed)


def infer_ticker_company_date(query: str, cio_df: pd.DataFrame) -> tuple[str, str, date]:
    text = query.strip()
    if not text:
        return "", "", date.today()
    upper = text.upper()
    if "ticker" in cio_df:
        ticker_matches = cio_df[cio_df["ticker"].astype(str).str.upper() == upper]
        if not ticker_matches.empty:
            row = ticker_matches.iloc[0]
            parsed = pd.to_datetime(row.get("earnings_date"), errors="coerce")
            return upper, str(row.get("company") or upper), parsed.date() if not pd.isna(parsed) else date.today()
        if "company" in cio_df:
            company_matches = cio_df[cio_df["company"].astype(str).str.lower() == text.lower()]
            if company_matches.empty:
                company_matches = cio_df[cio_df["company"].astype(str).str.lower().str.contains(text.lower(), regex=False)]
            if not company_matches.empty:
                row = company_matches.iloc[0]
                parsed = pd.to_datetime(row.get("earnings_date"), errors="coerce")
                ticker = str(row.get("ticker") or upper).upper()
                return ticker, str(row.get("company") or text), parsed.date() if not pd.isna(parsed) else date.today()
    return upper, text, date.today()


def infer_fmp_transcript_period(event_date: str) -> tuple[int, int]:
    """Approximate the fiscal quarter whose call is reported around an event date."""
    parsed = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(parsed):
        today = date.today()
        return today.year, (today.month - 1) // 3 + 1

    month = int(parsed.month)
    year = int(parsed.year)
    if month <= 3:
        return year - 1, 4
    if month <= 6:
        return year, 1
    if month <= 9:
        return year, 2
    return year, 3


def render_transcript_status(payload: dict[str, Any]) -> None:
    status = payload.get("transcript_status") or {}
    if not status:
        return

    requested_year = status.get("requested_year")
    requested_quarter = status.get("requested_quarter")
    period = f"{requested_year} Q{requested_quarter}" if requested_year and requested_quarter else "N/A"
    cols = st.columns(3)
    cols[0].metric("Transcript period", period)
    cols[1].metric("Transcript", "Loaded" if status.get("loaded") else "Not found")
    cols[2].metric("Guidance", "Extracted" if status.get("guidance_extracted") else "Not found")
    if status.get("message"):
        st.caption(str(status.get("message")))


def render_agent_output(title: str, payload: Any) -> None:
    st.write(f"### {title}")
    if not payload:
        st.info(f"No cached {title} output found for this ticker.")
        return

    if isinstance(payload, dict):
        cols = st.columns(4)
        short = payload.get("short_term_stance")
        long = payload.get("long_term_stance")
        stance = payload.get("stance")
        if short:
            cols[0].markdown(f"Short<br>{stance_badge(short)}", unsafe_allow_html=True)
        elif stance:
            cols[0].markdown(f"Stance<br>{stance_badge(stance)}", unsafe_allow_html=True)
        if long:
            cols[1].markdown(f"Long<br>{stance_badge(long)}", unsafe_allow_html=True)
        if payload.get("confidence_score") is not None:
            cols[2].metric("Confidence", f"{float(payload.get('confidence_score') or 0):.2f}")
        elif payload.get("confidence") is not None:
            cols[2].metric("Confidence", f"{float(payload.get('confidence') or 0):.2f}")
        if payload.get("thesis_impact"):
            cols[3].metric("Thesis impact", str(payload.get("thesis_impact")))
        render_transcript_status(payload)

        summary = (
            payload.get("news_summary")
            or payload.get("summary")
            or payload.get("core_judgment")
            or payload.get("core_thesis")
            or payload.get("stance_rationale")
            or payload.get("rationale")
        )
        if summary:
            st.write(summary)
        if payload.get("stance_rationale"):
            st.write("**Stance rationale:**")
            st.write(payload.get("stance_rationale"))
        if payload.get("confidence_reasoning") or payload.get("confidence_rationale"):
            st.write("**Confidence reasoning:**")
            st.write(payload.get("confidence_reasoning") or payload.get("confidence_rationale"))
        if payload.get("thesis_impact_reasoning"):
            st.write("**Thesis impact reasoning:**")
            st.write(payload.get("thesis_impact_reasoning"))
        render_signal_list("Key signals", payload.get("key_signals"))
        render_signal_list("Positive signals", payload.get("positive_signals"))
        render_signal_list("Negative signals", payload.get("negative_signals"))
        render_signal_list("Key evidence", payload.get("key_evidence"))
        render_signal_list("Risks", payload.get("risks"), limit=5)
        render_signal_list("Missing information", payload.get("missing_information"), limit=6)
        render_citations(payload)
    else:
        st.write(payload)


def run_agents_from_ui(
    ticker: str,
    company: str,
    earnings_date: str,
    cache_root: Path,
    force: bool,
    use_llm_debate: bool,
    use_llm_decision: bool,
    llm_provider: str,
    vertex_project: str,
    vertex_location: str,
    vertex_model: str,
    openai_model: str,
) -> dict[str, Any]:
    from openclam.agents.cio import cio_agent
    from openclam.agents.fundamental import fundamental_agent
    from openclam.agents.market_technical import market_technical_agent
    from openclam.agents.news_macro import news_macro_agent
    from openclam.evaluation import q4_earnings_cache as q4

    ticker = ticker.upper().strip()
    vertex_project = vertex_project.strip() or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or ""
    vertex_location = vertex_location.strip() or os.getenv("VERTEX_LOCATION", "us-central1")
    company = company.strip() or ticker
    root = q4.ensure_cache_dirs(cache_root)
    paths = q4.cached_ticker_paths(ticker, root)

    progress = st.progress(0)
    status = st.empty()
    details = st.empty()
    agent_outputs: dict[str, Any] = {}
    agent_errors: dict[str, str] = {}
    active_vertex_project = ""

    def step(percent: int, message: str, extra: str = "") -> None:
        progress.progress(percent)
        status.markdown(
            f"""
            <div style="
                background:#f4f6f8;
                border:1px solid #d9dee7;
                color:#334155;
                border-radius:6px;
                padding:0.7rem 0.9rem;
                font-weight:600;
            ">{message}</div>
            """,
            unsafe_allow_html=True,
        )
        if extra:
            details.caption(extra)

    if not force and paths["packets"].exists() and paths["cio"].exists():
        step(100, f"Loaded cached outputs for {ticker}.")
        return q4.load_cached_ticker(ticker, root)

    step(5, "Initializing LLM provider.", "Checking Vertex/OpenAI configuration.")
    if vertex_project:
        try:
            q4.VertexTextGenerator(vertex_project, vertex_location, vertex_model)
            active_vertex_project = vertex_project
        except Exception as exc:
            agent_errors["vertex_generator"] = q4._vertex_error_hint(exc)

    def run_news_macro_agent() -> Any:
        context = news_macro_agent.collect_context(
            ticker=ticker,
            company=company,
            event_date=earnings_date,
            lookback_days=14,
            max_news=10,
            news_mode="event_window",
            news_sources=["finnhub", "newsapi", "yfinance"],
            use_sample_if_empty=False,
            news_end_offset_days=0,
        )
        q4.save_json(paths["context"], context)
        news_report = news_macro_agent.generate_report(
            context,
            provider="auto",
            model=os.getenv("NEWS_MODEL", openai_model),
            gemini_model=vertex_model,
            vertex_project=active_vertex_project or None,
            vertex_location=vertex_location,
        )
        q4.save_json(paths["news_macro"], news_report)
        return news_report

    def run_market_agent() -> Any:
        if active_vertex_project:
            vertex_generator = q4.VertexTextGenerator(active_vertex_project, vertex_location, vertex_model)
            market_llm = q4.VertexLangChainCompatibleLLM(vertex_generator, temperature=1.0)
        else:
            market_llm = market_technical_agent.create_market_llm(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=openai_model,
                temperature=q4._openai_temperature(openai_model, 0.2),
            )
        market_query = f"{company} ({ticker}) technical analysis as of {earnings_date.replace('-', '/')}"
        market_report = market_technical_agent.run_market_analysis(market_query, llm=market_llm, ticker=ticker)
        q4.save_json(paths["market_technical"], market_report)
        return market_report

    def run_fundamental_agent() -> Any:
        transcript_year, transcript_quarter = infer_fmp_transcript_period(earnings_date)
        kwargs: dict[str, Any] = {
            "ticker": ticker,
            "transcript_year": transcript_year,
            "transcript_quarter": transcript_quarter,
            "require_transcript": False,
        }
        if active_vertex_project:
            vertex_generator = q4.VertexTextGenerator(active_vertex_project, vertex_location, vertex_model)
            kwargs.update(
                {
                    "model_name": vertex_model,
                    "temperature": 1.0,
                    "client": q4.VertexOpenAICompatibleClient(vertex_generator),
                }
            )
        else:
            kwargs.update(
                {
                    "model_name": openai_model,
                    "temperature": q4._openai_temperature(openai_model, 0.0),
                }
            )
        fundamental_report = fundamental_agent.run_fundamental_analysis(**kwargs)
        q4.save_json(paths["fundamental"], fundamental_report)
        return fundamental_report

    agent_jobs = {
        "news_macro": {
            "label": "News/Macro Agent",
            "details": "Fetching event-window news, company context, macro proxies, and generating the report.",
            "runner": run_news_macro_agent,
        },
        "market_technical": {
            "label": "Market Technical Agent",
            "details": "Computing price, momentum, volume, and event-reaction signals.",
            "runner": run_market_agent,
        },
        "fundamental": {
            "label": "Fundamental Agent",
            "details": "Reviewing financial metrics, valuation, transcript data, and missing information.",
            "runner": run_fundamental_agent,
        },
    }

    step(15, "Starting all three agents.", "News/Macro, Market Technical, and Fundamental are running in parallel.")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(job["runner"]): name for name, job in agent_jobs.items()}
        completed = 0
        for future in as_completed(futures):
            name = futures[future]
            job = agent_jobs[name]
            completed += 1
            try:
                agent_outputs[name] = future.result()
                step(
                    15 + completed * 20,
                    f"{job['label']} finished.",
                    f"{completed}/3 agents complete. {job['details']}",
                )
            except Exception as exc:
                agent_errors[name] = repr(exc)
                step(
                    15 + completed * 20,
                    f"{job['label']} failed; continuing.",
                    repr(exc),
                )

    step(82, "Building CIO input packets.", "Standardizing agent outputs into the CIO schema.")
    packets = []
    if "news_macro" in agent_outputs:
        packets.append(news_macro_agent.to_cio_agent_input(agent_outputs["news_macro"]))
    if "market_technical" in agent_outputs:
        packets.append(cio_agent.to_cio_packet_from_market(agent_outputs["market_technical"]))
    if "fundamental" in agent_outputs:
        packets.append(cio_agent.to_cio_packet_from_fundamental(agent_outputs["fundamental"], company=company))
    q4.save_json(paths["packets"], packets)
    q4.save_json(paths["errors"], agent_errors)

    step(90, "Running CIO synthesis and debate.", "Checking agreement, disagreement, and final stance.")
    if packets:
        cio_llm_provider = "vertex" if llm_provider == "vertex" and active_vertex_project else "openai"
        workflow = cio_agent.run_cio_workflow(
            packets,
            use_llm_debate=use_llm_debate,
            use_llm_decision=use_llm_decision,
            llm_provider=cio_llm_provider,
            debate_model=vertex_model if cio_llm_provider == "vertex" else openai_model,
            decision_model=vertex_model if cio_llm_provider == "vertex" else openai_model,
            vertex_project=active_vertex_project or None,
            vertex_location=vertex_location,
        )
        q4.save_json(cache_root / "cio" / f"{ticker}.json", workflow)

    packets_by_ticker = q4.load_cached_packets_by_ticker(cache_root)
    packets_by_ticker[ticker] = packets
    q4.save_json(cache_root / "tables" / "packets_by_ticker.json", packets_by_ticker)
    step(100, f"Finished {ticker}.", "Saved all available outputs to cache.")
    return {
        "ticker": ticker,
        "company": company,
        "earnings_date": earnings_date,
        "paths": {key: str(value) for key, value in paths.items()},
        "packets": packets,
        "errors": agent_errors,
    }


def row_from_cio_payload(ticker: str, cio_payload: dict[str, Any], company: str = "") -> pd.Series:
    decision = cio_payload.get("final_decision") or {}
    return pd.Series(
        {
            "ticker": ticker,
            "company": company or decision.get("company", ""),
            "bucket": "custom_run",
            "cio_short_term_stance": decision.get("final_short_term_stance"),
            "cio_long_term_stance": decision.get("final_long_term_stance"),
            "cio_confidence": decision.get("confidence", 0),
            "cio_reason": decision.get("core_thesis"),
            "post_7d_return": None,
            "abnormal_vs_qqq": None,
            "post_30d_return": None,
            "abnormal_30d_vs_qqq": None,
            "cio_short_direction_match": None,
            "cio_long_direction_match": None,
        }
    )


def main() -> None:
    st.set_page_config(page_title="OpenClam CIO Demo", layout="wide")
    inject_ui_styles()
    st.title("OpenClam Multi-Agent Investment Advisory")
    st.caption("Cached agent-output viewer and single-ticker runner")

    with st.sidebar:
        st.header("Demo Panel")
        root = DEFAULT_CACHE_ROOT
        cache_root = str(root)
        price_df, cio_df, strategy_df = load_tables(cache_root)
        tickers = sorted(cio_df["ticker"].dropna().astype(str).unique()) if "ticker" in cio_df else []
        selected = st.session_state.get("selected_ticker")
        if selected and selected not in tickers:
            tickers = sorted([*tickers, selected])
        if not tickers:
            tickers = ["TSLA"]
        default_ticker = selected or ("TSLA" if "TSLA" in tickers else (tickers[0] if tickers else "TSLA"))
        default_index = tickers.index(default_ticker) if default_ticker in tickers else 0
        ticker = st.selectbox("Choose a saved case", tickers, index=default_index if tickers else None)

        st.divider()
        st.subheader("Analyze a New Case")
        run_query = st.text_input("Ticker or company name", value=str(ticker or "TSLA"))
        run_ticker, inferred_company, inferred_date = infer_ticker_company_date(run_query, cio_df)
        run_date = st.date_input("Earnings or event date", value=inferred_date)
        st.caption(
            f"Resolved as {run_ticker or 'N/A'} / {inferred_company or 'N/A'}. "
            "Cached names auto-fill their date; for new names, enter a ticker and choose the event date."
        )
        force = st.checkbox("Refresh existing saved result", value=False)
        use_llm_debate = st.checkbox("Let agents debate", value=True)
        use_llm_decision = st.checkbox("Let CIO write final decision", value=True)
        llm_provider = st.selectbox("Model provider", ["vertex", "openai", "auto"], index=0)
        vertex_project = st.text_input("Google Cloud project", value=os.getenv("VERTEX_PROJECT", ""))
        vertex_location = st.text_input("Vertex location", value=os.getenv("VERTEX_LOCATION", "us-central1"))
        vertex_model = st.text_input("Vertex/Gemini model", value=os.getenv("VERTEX_MODEL", "gemini-2.5-flash"))
        openai_model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-nano"))

        if st.button("Run analysis", type="primary"):
            if not run_ticker:
                st.error("Ticker or company is required.")
            else:
                with st.spinner(f"Running agents for {run_ticker}..."):
                    try:
                        result = run_agents_from_ui(
                            ticker=run_ticker,
                            company=inferred_company or run_ticker,
                            earnings_date=str(run_date),
                            cache_root=root,
                            force=force,
                            use_llm_debate=use_llm_debate,
                            use_llm_decision=use_llm_decision,
                            llm_provider=llm_provider,
                            vertex_project=vertex_project,
                            vertex_location=vertex_location,
                            vertex_model=vertex_model,
                            openai_model=openai_model,
                        )
                        load_tables.clear()
                        st.session_state["selected_ticker"] = run_ticker.upper()
                        st.success(f"Saved cache for {run_ticker.upper()}.")
                        errors = result.get("errors") or {}
                        if errors:
                            st.warning(f"Completed with agent errors: {errors}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Run failed: {exc}")

        st.divider()
        st.write("Start this app:")
        st.code("streamlit run app.py", language="bash")

    if not tickers:
        st.error("No cached evaluation table found. Use the sidebar to run a ticker or check the cache path.")
        return

    root = Path(cache_root)
    cio_payload = load_json(root / "cio" / f"{ticker}.json", default={})
    news_payload = load_json(root / "news_macro" / f"{ticker}.json", default={})
    market_payload = load_json(root / "market_technical" / f"{ticker}.json", default={})
    fundamental_payload = load_json(root / "fundamental" / f"{ticker}.json", default={})
    if "ticker" in cio_df and ticker in set(cio_df["ticker"].astype(str)):
        row = cio_df[cio_df["ticker"].astype(str) == ticker].iloc[0]
    else:
        row = row_from_cio_payload(ticker, cio_payload, company=ticker)

    header_cols = st.columns([2, 1, 1, 1])
    header_cols[0].subheader(f"{row.get('ticker')} | {row.get('company', '')}")
    header_cols[1].markdown(f"Short stance<br>{stance_badge(row.get('cio_short_term_stance'))}", unsafe_allow_html=True)
    header_cols[2].markdown(f"Long stance<br>{stance_badge(row.get('cio_long_term_stance'))}", unsafe_allow_html=True)
    header_cols[3].metric("CIO confidence", f"{float(row.get('cio_confidence', 0) or 0):.2f}")

    tab_summary, tab_news, tab_market, tab_fundamental, tab_debate = st.tabs(
        ["CIO Decision", "News Macro", "Market Technical", "Fundamental", "Debate"]
    )

    with tab_summary:
        decision = cio_payload.get("final_decision") or {}
        st.write("### Final CIO View")
        st.write(decision.get("core_thesis") or row.get("cio_reason") or "No cached thesis found.")
        st.write("**Why now:**")
        st.write(decision.get("why_now") or "N/A")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Supporting evidence**")
            for item in decision.get("key_supporting_evidence", [])[:6]:
                st.write(f"- {item}")
        with col2:
            st.write("**Key risks**")
            for item in decision.get("key_risks", [])[:6]:
                st.write(f"- {item}")

        if news_payload:
            st.write("### News/Macro Snapshot")
            st.write(news_payload.get("news_summary", ""))
            st.caption(news_payload.get("core_insight", ""))

    with tab_news:
        render_agent_output("News Macro Agent Output", news_payload)

    with tab_market:
        render_agent_output("Market Technical Agent Output", market_payload)

    with tab_fundamental:
        render_agent_output("Fundamental Agent Output", fundamental_payload)

    with tab_debate:
        st.write("### Debate Diagnostics")
        render_debate(cio_payload)


if __name__ == "__main__":
    main()
