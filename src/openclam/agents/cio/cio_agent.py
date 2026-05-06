from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Pandas requires version .* of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd


STANCE_SCORES = {
    "Bullish": 1.0,
    "Neutral": 0.0,
    "Bearish": -1.0,
}

DEFAULT_SHORT_WEIGHTS = {
    "market_technical": 0.55,
    "news_macro": 0.35,
    "fundamental": 0.10,
}

DEFAULT_LONG_WEIGHTS = {
    "fundamental": 0.45,
    "news_macro": 0.35,
    "market_technical": 0.20,
}


BUCKET_WEIGHT_PROFILES = {
    "mega_cap_platform": {
        "short": {"news_macro": 0.55, "market_technical": 0.35, "fundamental": 0.10},
        "long": {"news_macro": 0.45, "fundamental": 0.35, "market_technical": 0.20},
        "rationale": "Mega-cap platform earnings tend to be dominated by headline interpretation, guidance, and capex narrative.",
    },
    "ai_semis": {
        "short": {"market_technical": 0.45, "news_macro": 0.40, "fundamental": 0.15},
        "long": {"news_macro": 0.40, "fundamental": 0.35, "market_technical": 0.25},
        "rationale": "AI semiconductor moves depend on technical confirmation plus second-order supply-chain and capex signals.",
    },
    "ai_infrastructure": {
        "short": {"news_macro": 0.45, "market_technical": 0.40, "fundamental": 0.15},
        "long": {"news_macro": 0.40, "fundamental": 0.35, "market_technical": 0.25},
        "rationale": "AI infrastructure is driven by second-order demand signals, but price confirmation still matters after earnings.",
    },
    "power_infrastructure": {
        "short": {"market_technical": 0.50, "news_macro": 0.40, "fundamental": 0.10},
        "long": {"news_macro": 0.40, "fundamental": 0.35, "market_technical": 0.25},
        "rationale": "Power infrastructure is a thematic second-order trade where technical confirmation helps avoid crowded entries.",
    },
    "data_center_reit": {
        "short": {"market_technical": 0.45, "news_macro": 0.40, "fundamental": 0.15},
        "long": {"fundamental": 0.40, "news_macro": 0.35, "market_technical": 0.25},
        "rationale": "Data-center REITs require balancing capital-cost fundamentals with AI demand and price confirmation.",
    },
    "data_center_operator": {
        "short": {"market_technical": 0.50, "news_macro": 0.40, "fundamental": 0.10},
        "long": {"news_macro": 0.40, "market_technical": 0.35, "fundamental": 0.25},
        "rationale": "Data-center operators are high-beta event trades where post-earnings price action carries high information value.",
    },
    "software_cloud": {
        "short": {"news_macro": 0.45, "market_technical": 0.40, "fundamental": 0.15},
        "long": {"news_macro": 0.40, "fundamental": 0.35, "market_technical": 0.25},
        "rationale": "Software/cloud reactions are often driven by growth durability, AI monetization narrative, and guidance quality.",
    },
}


def resolve_weight_profile(bucket: Any = None) -> dict[str, Any]:
    """Return bucket-aware CIO routing weights."""
    key = str(bucket or "").strip().lower()
    profile = BUCKET_WEIGHT_PROFILES.get(key)
    if not profile:
        return {
            "bucket": key or "default",
            "short": DEFAULT_SHORT_WEIGHTS,
            "long": DEFAULT_LONG_WEIGHTS,
            "rationale": "Default CIO weights are used because no bucket-specific routing rule matched.",
        }
    return {"bucket": key, **profile}


def normalize_stance(value: Any) -> str:
    """Normalize agent stance labels into Bullish/Neutral/Bearish."""
    text = str(value or "").strip().lower()
    if text in {"bull", "bullish", "long", "positive", "buy", "outperform"}:
        return "Bullish"
    if text in {"bear", "bearish", "short", "negative", "sell", "underperform"}:
        return "Bearish"
    return "Neutral"


def normalize_agent_name(value: Any) -> str:
    """Map different agent display names into stable CIO agent ids."""
    text = str(value or "").strip().lower().replace("&", "and")
    if "technical" in text or "market" in text:
        return "market_technical"
    if "fundamental" in text:
        return "fundamental"
    if "news" in text or "macro" in text:
        return "news_macro"
    return text.replace(" ", "_") or "unknown_agent"


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        score = float(value)
        if math.isnan(score):
            return default
        return max(0.0, min(1.0, score))
    except Exception:
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    return dict(value) if hasattr(value, "items") else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_agent_packet(packet: Any) -> dict[str, Any]:
    """Convert any agent result packet into the standard CIO schema."""
    raw = _as_dict(packet)
    agent_name = normalize_agent_name(raw.get("agent_name"))
    summary = raw.get("summary")
    if isinstance(summary, dict):
        summary_text = " | ".join(f"{key}: {value}" for key, value in summary.items() if value)
    else:
        summary_text = str(summary or raw.get("core_insight") or raw.get("core_judgment") or "")

    return {
        "ticker": str(raw.get("ticker") or "").upper(),
        "company": str(raw.get("company") or raw.get("company_name") or ""),
        "agent_name": agent_name,
        "short_term_stance": normalize_stance(raw.get("short_term_stance")),
        "long_term_stance": normalize_stance(raw.get("long_term_stance", raw.get("stance"))),
        "confidence": clamp_confidence(raw.get("confidence", raw.get("confidence_score", 0.5))),
        "confidence_rationale": str(
            raw.get("confidence_rationale")
            or raw.get("confidence_reasoning")
            or "No confidence rationale supplied."
        ),
        "stance_rationale": str(
            raw.get("stance_rationale")
            or raw.get("core_insight")
            or raw.get("core_judgment")
            or summary_text
            or "No stance rationale supplied."
        ),
        "key_signals": _as_list(raw.get("key_signals") or raw.get("positive_signals") or raw.get("key_evidence")),
        "risks": _as_list(raw.get("risks") or raw.get("negative_signals") or raw.get("missing_information")),
        "citations": _as_list(raw.get("citations")),
        "summary": summary_text,
        "raw_packet": raw,
    }


def to_cio_packet_from_market(market_output: Any) -> dict[str, Any]:
    """Adapter for the Market & Technical agent dictionary output."""
    return normalize_agent_packet(market_output)


def to_cio_packet_from_fundamental(fundamental_output: Any, company: str | None = None) -> dict[str, Any]:
    """Adapter for FundamentalOutput without changing the fundamental agent."""
    raw = _as_dict(fundamental_output)
    packet = {
        "ticker": raw.get("ticker"),
        "company": company or raw.get("company") or raw.get("company_name") or "",
        "agent_name": "fundamental",
        "short_term_stance": "Neutral",
        "long_term_stance": raw.get("stance", "Neutral"),
        "confidence": raw.get("confidence", 0.5),
        "confidence_rationale": raw.get("confidence_reasoning", ""),
        "stance_rationale": raw.get("core_judgment", ""),
        "key_signals": _as_list(raw.get("positive_signals")) + _as_list(raw.get("key_evidence")),
        "risks": _as_list(raw.get("negative_signals")) + _as_list(raw.get("missing_information")),
        "summary": {
            "beat_or_miss": raw.get("beat_or_miss"),
            "guidance_change": raw.get("guidance_change"),
            "management_tone": raw.get("management_tone"),
            "thesis_impact": raw.get("thesis_impact"),
            "thesis_impact_reasoning": raw.get("thesis_impact_reasoning"),
        },
    }
    return normalize_agent_packet(packet)


def _stance_counts(packets: list[dict[str, Any]], horizon: str) -> dict[str, int]:
    key = f"{horizon}_term_stance"
    counts = {"Bullish": 0, "Neutral": 0, "Bearish": 0}
    for packet in packets:
        counts[normalize_stance(packet.get(key))] += 1
    return counts


def _shared_stance_sentence(counts: dict[str, int], horizon: str) -> str | None:
    stance, count = max(counts.items(), key=lambda item: item[1])
    if count >= 2:
        return f"{count} agents lean {stance.lower()} on the {horizon}-term horizon."
    return None


def synthesize_agent_views(agent_packets: list[Any]) -> dict[str, Any]:
    """Summarize agreement, disagreement, and missing information across agents."""
    packets = [normalize_agent_packet(packet) for packet in agent_packets]
    ticker = next((packet["ticker"] for packet in packets if packet["ticker"]), "")
    company = next((packet["company"] for packet in packets if packet["company"]), "")
    short_counts = _stance_counts(packets, "short")
    long_counts = _stance_counts(packets, "long")

    agreements = [
        item
        for item in (
            _shared_stance_sentence(short_counts, "short"),
            _shared_stance_sentence(long_counts, "long"),
        )
        if item
    ]
    disagreements: list[str] = []
    if short_counts["Bullish"] and short_counts["Bearish"]:
        disagreements.append("Short-term views contain both bullish and bearish calls.")
    if long_counts["Bullish"] and long_counts["Bearish"]:
        disagreements.append("Long-term views contain both bullish and bearish calls.")

    missing_information = []
    for packet in packets:
        raw_missing = packet["raw_packet"].get("missing_information")
        for item in _as_list(raw_missing):
            if item and item not in missing_information:
                missing_information.append(str(item))

    return {
        "ticker": ticker,
        "company": company,
        "agent_view_summary": [
            {
                "agent_name": packet["agent_name"],
                "short_term_stance": packet["short_term_stance"],
                "long_term_stance": packet["long_term_stance"],
                "confidence": packet["confidence"],
                "rationale": packet["stance_rationale"],
            }
            for packet in packets
        ],
        "areas_of_agreement": agreements,
        "areas_of_disagreement": disagreements,
        "missing_information": missing_information,
        "short_stance_counts": short_counts,
        "long_stance_counts": long_counts,
        "agent_packets": packets,
    }


def should_trigger_debate(synthesis: dict[str, Any], high_confidence: float = 0.7) -> dict[str, Any]:
    """Decide whether CIO should ask agents to respond to each other's evidence."""
    packets = synthesis.get("agent_packets", [])
    reasons: list[str] = []
    conflict_level = "low"

    if synthesis.get("areas_of_disagreement"):
        reasons.extend(synthesis["areas_of_disagreement"])
        conflict_level = "high"

    for horizon in ("short", "long"):
        key = f"{horizon}_term_stance"
        high_conf_stances = {
            packet[key]
            for packet in packets
            if packet.get("confidence", 0.0) >= high_confidence and packet.get(key) != "Neutral"
        }
        if len(high_conf_stances) > 1:
            reasons.append(f"High-confidence {horizon}-term agents disagree.")
            conflict_level = "high"

    if not reasons and len(packets) < 3:
        reasons.append("Only partial agent coverage is available, so CIO should keep uncertainty explicit.")
        conflict_level = "medium"

    return {
        "debate_required": conflict_level == "high",
        "conflict_level": conflict_level,
        "debate_reason": reasons or ["No material cross-agent disagreement."],
    }


def build_debate_prompt(agent_packet: dict[str, Any], other_packets: list[dict[str, Any]]) -> str:
    """Build a prompt asking one agent to reconsider its view after seeing peer evidence."""
    return json.dumps(
        {
            "role": "You are revising your agent view for a CIO committee. Use only the supplied evidence.",
            "your_original_view": agent_packet,
            "other_agent_views": other_packets,
            "task": [
                "Identify what evidence from other agents you agree with.",
                "Identify what evidence you disagree with or believe has lower weight.",
                "Decide whether to maintain or revise short_term_stance, long_term_stance, and confidence.",
                "Return JSON with keys: response_type, agreement, disagreement, revised_short_term_stance, revised_long_term_stance, revised_confidence, evidence_needed.",
            ],
        },
        indent=2,
        default=str,
    )


def _call_openai_json(prompt: str, model: str, api_key: str | None = None) -> dict[str, Any] | None:
    """Small optional OpenAI JSON helper. CIO still works without this dependency/key."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=resolved_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not include markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return None


def _call_vertex_json(
    prompt: str,
    model: str,
    project: str | None = None,
    location: str | None = None,
) -> dict[str, Any] | None:
    """Optional Vertex Gemini JSON helper. Falls back cleanly if Vertex is unavailable."""
    resolved_project = project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    resolved_location = location or os.getenv("VERTEX_LOCATION", "us-central1")
    if not resolved_project:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(vertexai=True, project=resolved_project, location=resolved_location)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text or "{}")
    except Exception:
        pass

    try:
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        vertexai.init(project=resolved_project, location=resolved_location)
        vertex_model = GenerativeModel(model)
        response = vertex_model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text or "{}")
    except Exception:
        return None


def _call_llm_json(
    prompt: str,
    model: str,
    provider: str = "auto",
    api_key: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
) -> dict[str, Any] | None:
    provider = (provider or "auto").lower()
    if provider in {"auto", "vertex"}:
        vertex_payload = _call_vertex_json(
            prompt,
            model=model,
            project=vertex_project,
            location=vertex_location,
        )
        if vertex_payload or provider == "vertex":
            return vertex_payload
    if provider in {"auto", "openai"}:
        return _call_openai_json(prompt, model=model, api_key=api_key)
    return None


def _fallback_debate_response(
    agent_packet: dict[str, Any],
    other_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministic debate fallback used when no LLM is configured."""
    direct_opposition = False
    for other in other_packets:
        short_opposes = {agent_packet["short_term_stance"], other["short_term_stance"]} == {"Bullish", "Bearish"}
        long_opposes = {agent_packet["long_term_stance"], other["long_term_stance"]} == {"Bullish", "Bearish"}
        direct_opposition = direct_opposition or short_opposes or long_opposes

    confidence_haircut = 0.15 if direct_opposition else 0.05
    return {
        "agent_name": agent_packet["agent_name"],
        "response_type": "Maintain",
        "agreement": "No LLM debate was run; CIO fallback preserves the agent view but explicitly accounts for peer disagreement.",
        "disagreement": "Direct opposition from another agent lowers confidence." if direct_opposition else "No direct bull/bear opposition.",
        "revised_short_term_stance": agent_packet["short_term_stance"],
        "revised_long_term_stance": agent_packet["long_term_stance"],
        "revised_confidence": max(0.0, round(agent_packet["confidence"] - confidence_haircut, 3)),
        "evidence_needed": "Run LLM debate or add timestamped evidence if this conflict is investment-critical.",
    }


def run_cio_debate(
    agent_packets: list[Any],
    use_llm: bool = False,
    model: str = "gpt-5.4-nano",
    provider: str = "auto",
    api_key: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
) -> dict[str, Any]:
    """Run one debate round when agents materially disagree."""
    packets = [normalize_agent_packet(packet) for packet in agent_packets]
    synthesis = synthesize_agent_views(packets)
    trigger = should_trigger_debate(synthesis)
    if not trigger["debate_required"]:
        return {
            "triggered": False,
            "trigger": trigger,
            "synthesis": synthesis,
            "debate_responses": [],
            "revised_packets": packets,
        }

    responses: list[dict[str, Any]] = []
    revised_packets: list[dict[str, Any]] = []
    for idx, packet in enumerate(packets):
        others = [other for other_idx, other in enumerate(packets) if other_idx != idx]
        response = None
        if use_llm:
            response = _call_llm_json(
                build_debate_prompt(packet, others),
                model=model,
                provider=provider,
                api_key=api_key,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
            )
        if not response:
            response = _fallback_debate_response(packet, others)

        revised = dict(packet)
        revised["short_term_stance"] = normalize_stance(response.get("revised_short_term_stance"))
        revised["long_term_stance"] = normalize_stance(response.get("revised_long_term_stance"))
        revised["confidence"] = clamp_confidence(response.get("revised_confidence"), packet["confidence"])
        revised["debate_response"] = response
        responses.append(response)
        revised_packets.append(revised)

    return {
        "triggered": True,
        "trigger": trigger,
        "synthesis": synthesis,
        "debate_responses": responses,
        "revised_packets": revised_packets,
    }


def _weighted_stance_score(
    packets: list[dict[str, Any]],
    horizon: str,
    weights: dict[str, float],
) -> dict[str, Any]:
    stance_key = f"{horizon}_term_stance"
    weighted_sum = 0.0
    total_weight = 0.0
    votes = []
    for packet in packets:
        weight = weights.get(packet["agent_name"], 0.25)
        confidence = packet.get("confidence", 0.5)
        stance = normalize_stance(packet.get(stance_key))
        contribution = weight * confidence * STANCE_SCORES[stance]
        weighted_sum += contribution
        total_weight += weight * confidence
        votes.append(
            {
                "agent_name": packet["agent_name"],
                "stance": stance,
                "confidence": confidence,
                "weight": weight,
                "contribution": round(contribution, 4),
            }
        )

    score = weighted_sum / total_weight if total_weight else 0.0
    if score > 0.25:
        final_stance = "Bullish"
    elif score < -0.25:
        final_stance = "Bearish"
    else:
        final_stance = "Neutral"
    return {
        "score": round(score, 4),
        "stance": final_stance,
        "votes": votes,
    }


def _summarize_evidence(packets: list[dict[str, Any]], field: str, limit: int = 6) -> list[str]:
    items: list[str] = []
    for packet in packets:
        for item in _as_list(packet.get(field)):
            if isinstance(item, dict):
                text = item.get("evidence") or item.get("signal") or json.dumps(item, default=str)
            else:
                text = str(item)
            text = text.strip()
            if text and text not in items:
                items.append(text)
            if len(items) >= limit:
                return items
    return items


def generate_cio_decision(
    agent_packets: list[Any],
    debate_result: dict[str, Any] | None = None,
    use_llm: bool = False,
    model: str = "gpt-5.4-nano",
    provider: str = "auto",
    api_key: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    bucket: Any = None,
) -> dict[str, Any]:
    """Produce the CIO final decision from original or debate-revised packets."""
    packets = (
        debate_result.get("revised_packets", [])
        if debate_result
        else [normalize_agent_packet(packet) for packet in agent_packets]
    )
    synthesis = debate_result.get("synthesis") if debate_result else synthesize_agent_views(packets)
    trigger = debate_result.get("trigger") if debate_result else should_trigger_debate(synthesis)
    weight_profile = resolve_weight_profile(bucket)
    short_score = _weighted_stance_score(packets, "short", weight_profile["short"])
    long_score = _weighted_stance_score(packets, "long", weight_profile["long"])

    average_confidence = sum(packet["confidence"] for packet in packets) / len(packets) if packets else 0.0
    conviction = max(abs(short_score["score"]), abs(long_score["score"]))
    conflict_penalty = 0.15 if trigger["conflict_level"] == "high" else 0.05 if trigger["conflict_level"] == "medium" else 0.0
    final_confidence = max(0.0, min(1.0, average_confidence * 0.55 + conviction * 0.45 - conflict_penalty))

    if short_score["stance"] == "Bullish" and long_score["stance"] == "Bullish":
        action = "Long"
    elif short_score["stance"] == "Bearish" and long_score["stance"] == "Bearish":
        action = "Short / Avoid"
    elif short_score["stance"] == "Bearish" and long_score["stance"] == "Bullish":
        action = "Wait for pullback"
    elif short_score["stance"] == "Bullish" and long_score["stance"] == "Bearish":
        action = "Tactical long only"
    else:
        action = "Watch / No Trade"

    if final_confidence >= 0.72 and action not in {"Watch / No Trade"}:
        position_size = "Normal"
    elif final_confidence >= 0.55 and action not in {"Watch / No Trade"}:
        position_size = "Small"
    else:
        position_size = "No new position"

    supporting = _summarize_evidence(packets, "key_signals")
    risks = _summarize_evidence(packets, "risks")
    ticker = synthesis.get("ticker") or next((packet["ticker"] for packet in packets if packet["ticker"]), "")
    company = synthesis.get("company") or next((packet["company"] for packet in packets if packet["company"]), "")

    deterministic_decision = {
        "ticker": ticker,
        "company": company,
        "final_short_term_stance": short_score["stance"],
        "final_long_term_stance": long_score["stance"],
        "investment_action": action,
        "position_size_hint": position_size,
        "confidence": round(final_confidence, 3),
        "core_thesis": (
            f"CIO view for {ticker}: short-term {short_score['stance'].lower()}, "
            f"long-term {long_score['stance'].lower()}, based on weighted cross-agent evidence."
        ),
        "why_now": "Decision is based on the current event-window evidence packet supplied by the agents.",
        "key_supporting_evidence": supporting,
        "key_risks": risks,
        "agent_view_summary": synthesis.get("agent_view_summary", []),
        "debate_summary": {
            "debate_triggered": bool(debate_result and debate_result.get("triggered")),
            "conflict_level": trigger["conflict_level"],
            "debate_reason": trigger["debate_reason"],
            "debate_responses": debate_result.get("debate_responses", []) if debate_result else [],
        },
        "routing_profile": weight_profile,
        "reason_for_final_decision": (
            f"Short score {short_score['score']} and long score {long_score['score']} after "
            f"{'debate-adjusted' if debate_result and debate_result.get('triggered') else 'initial'} "
            f"confidence weighting using {weight_profile['bucket']} routing."
        ),
        "agent_votes": {
            "short": short_score["votes"],
            "long": long_score["votes"],
        },
        "scores": {
            "short": short_score["score"],
            "long": long_score["score"],
        },
    }
    if not use_llm:
        return deterministic_decision

    prompt = json.dumps(
        {
            "role": "You are the CIO of a multi-agent investment advisory committee.",
            "task": [
                "Synthesize all agent packets and debate results.",
                "Return the final investment view for the main ticker only.",
                "Use the deterministic weighted decision as a guardrail, but override it if the evidence clearly justifies doing so.",
                "Return JSON with keys: final_short_term_stance, final_long_term_stance, investment_action, position_size_hint, confidence, core_thesis, why_now, key_supporting_evidence, key_risks, reason_for_final_decision.",
            ],
            "allowed_stances": ["Bullish", "Neutral", "Bearish"],
            "agent_packets": packets,
            "debate_result": debate_result,
            "deterministic_guardrail": deterministic_decision,
        },
        indent=2,
        default=str,
    )
    llm_decision = _call_llm_json(
        prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    if not llm_decision:
        return deterministic_decision

    merged = dict(deterministic_decision)
    for key in (
        "investment_action",
        "position_size_hint",
        "core_thesis",
        "why_now",
        "key_supporting_evidence",
        "key_risks",
        "reason_for_final_decision",
    ):
        if key in llm_decision:
            merged[key] = llm_decision[key]
    merged["final_short_term_stance"] = normalize_stance(
        llm_decision.get("final_short_term_stance", deterministic_decision["final_short_term_stance"])
    )
    merged["final_long_term_stance"] = normalize_stance(
        llm_decision.get("final_long_term_stance", deterministic_decision["final_long_term_stance"])
    )
    merged["confidence"] = clamp_confidence(llm_decision.get("confidence"), deterministic_decision["confidence"])
    merged["llm_decision_used"] = True
    return merged


def run_cio_workflow(
    agent_packets: list[Any],
    use_llm_debate: bool = False,
    use_llm_decision: bool = False,
    llm_provider: str = "auto",
    debate_model: str = "gpt-5.4-nano",
    decision_model: str = "gpt-5.4-nano",
    api_key: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    bucket: Any = None,
) -> dict[str, Any]:
    """Full CIO flow: synthesize agent views, debate if needed, then decide."""
    normalized_packets = [normalize_agent_packet(packet) for packet in agent_packets]
    synthesis = synthesize_agent_views(normalized_packets)
    debate = run_cio_debate(
        normalized_packets,
        use_llm=use_llm_debate,
        model=debate_model,
        provider=llm_provider,
        api_key=api_key,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    final_decision = generate_cio_decision(
        normalized_packets,
        debate_result=debate,
        use_llm=use_llm_decision,
        model=decision_model,
        provider=llm_provider,
        api_key=api_key,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        bucket=bucket,
    )
    return {
        "synthesis": synthesis,
        "debate": debate,
        "final_decision": final_decision,
    }


def _stance_to_direction(stance: Any) -> str | None:
    normalized = normalize_stance(stance)
    if normalized == "Bullish":
        return "up"
    if normalized == "Bearish":
        return "down"
    return None


def _direction_match(
    predicted: str | None,
    realized: Any,
    abnormal: Any,
    neutral_band: float,
) -> bool | None:
    if pd.isna(abnormal) or realized is None:
        return None
    abnormal_value = float(abnormal)
    if predicted is None:
        return abs(abnormal_value) <= neutral_band
    return predicted == realized


def _direction_match_reason(
    predicted: str | None,
    realized: Any,
    abnormal: Any,
    neutral_band: float,
) -> str:
    if pd.isna(abnormal) or realized is None:
        return "missing realized abnormal return"
    abnormal_value = float(abnormal)
    if predicted is None:
        if abs(abnormal_value) <= neutral_band:
            return "neutral matched: abnormal return stayed inside neutral band"
        return "neutral missed: abnormal return moved outside neutral band"
    return "matched" if predicted == realized else "missed"


def run_cio_eval(
    summary_df: pd.DataFrame,
    packets_by_ticker: dict[str, list[Any]],
    long_post_trading_days: int = 30,
    neutral_band: float = 0.02,
    use_llm_debate: bool = False,
    use_llm_decision: bool = False,
    llm_provider: str = "auto",
    debate_model: str = "gpt-5.4-nano",
    decision_model: str = "gpt-5.4-nano",
    api_key: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate CIO decisions against short and long abnormal returns versus QQQ."""
    long_abnormal_col = f"abnormal_{long_post_trading_days}d_vs_qqq"
    long_direction_col = f"realized_{long_post_trading_days}d_direction_vs_qqq"
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}

    for _, row in summary_df.iterrows():
        ticker = str(row["ticker"]).upper()
        packets = packets_by_ticker.get(ticker, [])
        base = row.to_dict()
        if not packets:
            base.update(
                {
                    "cio_ready": False,
                    "cio_short_term_stance": None,
                    "cio_long_term_stance": None,
                    "cio_action": None,
                    "cio_confidence": None,
                    "cio_debate_triggered": None,
                    "cio_short_direction_match": None,
                    "cio_short_direction_match_reason": "missing CIO agent packets",
                    "cio_long_direction_match": None,
                    "cio_long_direction_match_reason": "missing CIO agent packets",
                }
            )
            rows.append(base)
            continue

        workflow = run_cio_workflow(
            packets,
            use_llm_debate=use_llm_debate,
            use_llm_decision=use_llm_decision,
            llm_provider=llm_provider,
            debate_model=debate_model,
            decision_model=decision_model,
            api_key=api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            bucket=row.get("bucket"),
        )
        decision = workflow["final_decision"]
        results[ticker] = workflow

        short_predicted = _stance_to_direction(decision["final_short_term_stance"])
        long_predicted = _stance_to_direction(decision["final_long_term_stance"])
        short_match = _direction_match(
            short_predicted,
            row.get("realized_direction_vs_qqq"),
            row.get("abnormal_vs_qqq"),
            neutral_band,
        )
        long_match = _direction_match(
            long_predicted,
            row.get(long_direction_col),
            row.get(long_abnormal_col),
            neutral_band,
        )
        base.update(
            {
                "cio_ready": True,
                "cio_short_term_stance": decision["final_short_term_stance"],
                "cio_long_term_stance": decision["final_long_term_stance"],
                "cio_action": decision["investment_action"],
                "cio_confidence": decision["confidence"],
                "cio_debate_triggered": workflow["debate"]["triggered"],
                "cio_debate_conflict_level": workflow["debate"]["trigger"]["conflict_level"],
                "cio_routing_bucket": decision.get("routing_profile", {}).get("bucket"),
                "cio_routing_rationale": decision.get("routing_profile", {}).get("rationale"),
                "cio_reason": decision["reason_for_final_decision"],
                "cio_short_direction_match": short_match,
                "cio_short_direction_match_reason": _direction_match_reason(
                    short_predicted,
                    row.get("realized_direction_vs_qqq"),
                    row.get("abnormal_vs_qqq"),
                    neutral_band,
                ),
                "cio_long_direction_match": long_match,
                "cio_long_direction_match_reason": _direction_match_reason(
                    long_predicted,
                    row.get(long_direction_col),
                    row.get(long_abnormal_col),
                    neutral_band,
                ),
            }
        )
        rows.append(base)

    return pd.DataFrame(rows), results


def evaluate_cio_decisions(cio_eval: pd.DataFrame) -> dict[str, Any]:
    """Summarize CIO eval accuracy and debate trigger rate."""
    def _accuracy(column: str) -> dict[str, Any]:
        evaluable = cio_eval[cio_eval[column].notna()] if column in cio_eval else pd.DataFrame()
        matched = int((evaluable[column] == True).sum()) if not evaluable.empty else 0
        total = int(len(evaluable))
        return {
            f"{column}_evaluable": total,
            f"{column}_matched": matched,
            f"{column}_accuracy": matched / total if total else None,
        }

    return {
        "cases": int(len(cio_eval)),
        "cio_ready_cases": int((cio_eval.get("cio_ready") == True).sum()) if "cio_ready" in cio_eval else 0,
        "debate_trigger_rate": float((cio_eval.get("cio_debate_triggered") == True).mean())
        if "cio_debate_triggered" in cio_eval and len(cio_eval)
        else None,
        **_accuracy("cio_short_direction_match"),
        **_accuracy("cio_long_direction_match"),
    }


def aggregate_agent_packets(agent_packets: list[Any]) -> dict[str, Any]:
    """Backward-compatible name for deterministic CIO aggregation."""
    workflow = run_cio_workflow(agent_packets, use_llm_debate=False)
    return workflow["final_decision"]


def run_cio_debate_round(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper around run_cio_debate."""
    return run_cio_debate(*args, **kwargs)
