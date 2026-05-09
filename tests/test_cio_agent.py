from openclam.agents.cio import cio_agent


def test_normalize_agent_packet_handles_minimal_packet():
    packet = cio_agent.normalize_agent_packet(
        {
            "agent_name": "Market Technical",
            "short_term_stance": "long",
            "long_term_stance": "underperform",
            "confidence": 1.4,
            "stance_rationale": "Momentum is positive, but long-term risk is elevated.",
        }
    )

    assert packet["agent_name"] == "market_technical"
    assert packet["short_term_stance"] == "Bullish"
    assert packet["long_term_stance"] == "Bearish"
    assert packet["confidence"] == 1.0


def test_cio_debate_trigger_detects_high_conflict():
    synthesis = cio_agent.synthesize_agent_views(
        [
            {
                "ticker": "TSLA",
                "company": "Tesla",
                "agent_name": "news_macro",
                "short_term_stance": "Bullish",
                "long_term_stance": "Bullish",
                "confidence": 0.9,
            },
            {
                "ticker": "TSLA",
                "company": "Tesla",
                "agent_name": "market_technical",
                "short_term_stance": "Bearish",
                "long_term_stance": "Neutral",
                "confidence": 0.8,
            },
            {
                "ticker": "TSLA",
                "company": "Tesla",
                "agent_name": "fundamental",
                "short_term_stance": "Neutral",
                "long_term_stance": "Bullish",
                "confidence": 0.5,
            },
        ]
    )
    trigger = cio_agent.should_trigger_debate(synthesis)

    assert trigger["debate_required"] is True
    assert trigger["conflict_level"] == "high"
    assert any("Short-term" in reason for reason in trigger["debate_reason"])


def test_weight_profile_falls_back_for_unknown_bucket():
    profile = cio_agent.resolve_weight_profile("unknown_bucket")

    assert profile["bucket"] == "unknown_bucket"
    assert set(profile["short"]) == {"market_technical", "news_macro", "fundamental"}
    assert set(profile["long"]) == {"market_technical", "news_macro", "fundamental"}
