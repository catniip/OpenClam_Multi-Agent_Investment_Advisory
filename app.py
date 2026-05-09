from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "agent_outputs" / "q4_2025_ai_tech"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
    text = str(label or "Neutral")
    color = {
        "Bullish": "#0b8f4d",
        "Bearish": "#b42318",
        "Neutral": "#596579",
    }.get(text, "#596579")
    return (
        f"<span style='background:{color}; color:white; padding:0.20rem 0.55rem; "
        f"border-radius:999px; font-size:0.84rem; font-weight:700'>{text}</span>"
    )


def render_metric_row(row: pd.Series) -> None:
    cols = st.columns(4)
    cols[0].metric("7D return", format_pct(row.get("post_7d_return")))
    cols[1].metric("7D abnormal vs QQQ", format_pct(row.get("abnormal_vs_qqq")))
    cols[2].metric("30D return", format_pct(row.get("post_30d_return")))
    cols[3].metric("30D abnormal vs QQQ", format_pct(row.get("abnormal_30d_vs_qqq")))


def render_agent_packets(packets: list[dict[str, Any]]) -> None:
    rows = []
    for packet in packets:
        rows.append(
            {
                "agent": packet.get("agent_name"),
                "short": packet.get("short_term_stance"),
                "long": packet.get("long_term_stance"),
                "confidence": packet.get("confidence"),
                "rationale": packet.get("stance_rationale") or packet.get("rationale"),
            }
        )
    if not rows:
        st.info("No cached agent packets found for this ticker.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
        response_rows = [
            {
                "type": item.get("response_type"),
                "revised short": item.get("revised_short_term_stance"),
                "revised long": item.get("revised_long_term_stance"),
                "confidence": item.get("revised_confidence"),
                "evidence needed": item.get("evidence_needed"),
            }
            for item in responses
        ]
        st.dataframe(pd.DataFrame(response_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="OpenClam CIO Demo", layout="wide")
    st.title("OpenClam Multi-Agent Investment Advisory")
    st.caption("Cached Q4 2025 AI/Tech earnings evaluation viewer")

    with st.sidebar:
        st.header("Controls")
        cache_root = st.text_input("Cache root", value=str(DEFAULT_CACHE_ROOT))
        price_df, cio_df, strategy_df = load_tables(cache_root)
        tickers = sorted(cio_df["ticker"].dropna().astype(str).unique()) if "ticker" in cio_df else []
        default_index = tickers.index("TSLA") if "TSLA" in tickers else 0
        ticker = st.selectbox("Ticker", tickers, index=default_index if tickers else None)
        st.divider()
        st.write("Run locally:")
        st.code("streamlit run app.py", language="bash")

    if not tickers:
        st.error("No cached evaluation table found. Run notebook 05 or check the cache path.")
        return

    root = Path(cache_root)
    row = cio_df[cio_df["ticker"].astype(str) == ticker].iloc[0]
    packets = load_json(root / "packets" / f"{ticker}.json", default=[])
    cio_payload = load_json(root / "cio" / f"{ticker}.json", default={})
    news_payload = load_json(root / "news_macro" / f"{ticker}.json", default={})

    header_cols = st.columns([2, 1, 1, 1])
    header_cols[0].subheader(f"{row.get('ticker')} | {row.get('company', '')}")
    header_cols[1].markdown(f"Short stance<br>{stance_badge(row.get('cio_short_term_stance'))}", unsafe_allow_html=True)
    header_cols[2].markdown(f"Long stance<br>{stance_badge(row.get('cio_long_term_stance'))}", unsafe_allow_html=True)
    header_cols[3].metric("CIO confidence", f"{float(row.get('cio_confidence', 0) or 0):.2f}")

    render_metric_row(row)

    tab_summary, tab_agents, tab_debate, tab_eval = st.tabs(
        ["CIO Decision", "Agent Inputs", "Debate", "Evaluation"]
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

    with tab_agents:
        st.write("### CIO-Ready Agent Packets")
        render_agent_packets(packets)

    with tab_debate:
        st.write("### Debate Diagnostics")
        render_debate(cio_payload)

    with tab_eval:
        st.write("### Result for Selected Ticker")
        eval_cols = [
            "ticker",
            "bucket",
            "cio_short_term_stance",
            "cio_long_term_stance",
            "cio_short_direction_match",
            "cio_long_direction_match",
            "abnormal_vs_qqq",
            "abnormal_30d_vs_qqq",
        ]
        st.dataframe(pd.DataFrame([row])[eval_cols], use_container_width=True, hide_index=True)

        if not strategy_df.empty:
            st.write("### Strategy Summary")
            show = strategy_df.copy()
            for col in ["short_accuracy", "long_accuracy"]:
                if col in show:
                    show[col] = show[col].map(lambda value: f"{value:.1%}")
            st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
