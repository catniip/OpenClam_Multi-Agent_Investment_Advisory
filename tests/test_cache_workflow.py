from openclam.evaluation import q4_earnings_cache as q4


def test_q4_universe_contains_expected_buckets():
    universe = q4.q4_2025_combined_cio_advantage_df()

    assert len(universe) == 55
    assert {"ticker", "company", "earnings_date", "bucket"}.issubset(universe.columns)
    assert {"mega_cap_platform", "ai_semis", "software_cloud"}.issubset(set(universe["bucket"]))


def test_json_cache_round_trip(tmp_path):
    root = q4.ensure_cache_dirs(tmp_path / "cache")
    paths = q4.cached_ticker_paths("NVDA", root)

    payload = {"ticker": "NVDA", "status": "ok"}
    q4.save_json(paths["news_macro"], payload)

    loaded = q4.load_cached_ticker("NVDA", root)
    assert loaded["ticker"] == "NVDA"
    assert loaded["news_macro"] == payload
    assert loaded["market_technical"] is None
