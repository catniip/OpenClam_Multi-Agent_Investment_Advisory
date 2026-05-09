import pandas as pd

from openclam.evaluation import q4_earnings_cache as q4


def test_direction_match_scores_neutral_band():
    assert q4.direction_match_label(None, "up", 0.01, neutral_band=0.02) is True
    assert q4.direction_match_label(None, "up", 0.03, neutral_band=0.02) is False
    assert q4.direction_match_label("up", "up", 0.03, neutral_band=0.02) is True
    assert q4.direction_match_label("down", "up", 0.03, neutral_band=0.02) is False


def test_score_stance_columns_adds_short_and_long_matches():
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "short_stance": "Bullish",
                "long_stance": "Bearish",
                "abnormal_vs_qqq": 0.05,
                "realized_direction_vs_qqq": "up",
                "abnormal_30d_vs_qqq": -0.04,
                "realized_30d_direction_vs_qqq": "down",
            }
        ]
    )

    scored = q4.score_stance_columns(
        frame,
        short_stance_col="short_stance",
        long_stance_col="long_stance",
        prefix="demo",
        long_post_trading_days=30,
        neutral_band=0.02,
    )

    assert scored.loc[0, "demo_short_predicted_direction"] == "up"
    assert scored.loc[0, "demo_long_predicted_direction"] == "down"
    assert bool(scored.loc[0, "demo_short_direction_match"]) is True
    assert bool(scored.loc[0, "demo_long_direction_match"]) is True


def test_raw_aggregation_supports_bucket_and_uniform_weights():
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "bucket": "mega_cap_platform",
                "abnormal_vs_qqq": 0.03,
                "realized_direction_vs_qqq": "up",
                "abnormal_30d_vs_qqq": -0.04,
                "realized_30d_direction_vs_qqq": "down",
            }
        ]
    )
    packets = {
        "AAA": [
            {
                "agent_name": "news_macro",
                "short_term_stance": "Bullish",
                "long_term_stance": "Bearish",
                "confidence": 0.9,
            },
            {
                "agent_name": "market_technical",
                "short_term_stance": "Bearish",
                "long_term_stance": "Bullish",
                "confidence": 0.9,
            },
            {
                "agent_name": "fundamental",
                "short_term_stance": "Bearish",
                "long_term_stance": "Bearish",
                "confidence": 0.9,
            },
        ]
    }

    scored = q4.add_raw_aggregation_baseline(
        frame,
        packets,
        prefix="bucket_raw",
        weight_mode="bucket",
    )
    scored = q4.add_raw_aggregation_baseline(
        scored,
        packets,
        prefix="uniform_raw",
        weight_mode="uniform",
    )

    assert scored.loc[0, "bucket_raw_short_stance"] == "Neutral"
    assert scored.loc[0, "uniform_raw_short_stance"] == "Bearish"
