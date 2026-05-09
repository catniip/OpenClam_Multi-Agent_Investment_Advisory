# Troubleshooting Guide

## Import Errors

### `ModuleNotFoundError: No module named 'openclam'`

Install the package from the repository root:

```bash
pip install -e .
```

If running from a notebook, restart the kernel after installation.

### `ModuleNotFoundError: No module named 'news_macro_agent'`

Use the package import path:

```python
from openclam.agents.news_macro import news_macro_agent
```

The old direct import only works if the source folder is manually added to `PYTHONPATH`.

## API and Credential Issues

### Vertex AI authentication fails

For local runs:

```bash
gcloud auth application-default login
```

Then set:

```text
VERTEX_PROJECT=your-project-id
VERTEX_LOCATION=us-central1
```

For Colab:

```python
from google.colab import auth
auth.authenticate_user()
```

### NewsAPI returns HTTP 426 or browser CORS errors

NewsAPI Developer plans restrict some browser and historical usage. The project handles this by:

- using Finnhub and Yahoo Finance where available,
- recording the NewsAPI error in `data_notes`,
- continuing the run instead of failing the whole agent.

### Missing API keys

Missing optional keys should not crash the workflow. The affected source is skipped and a note is written to the context or error file.

## Cache and Reproducibility

### Notebook keeps reusing old outputs

Cached outputs are stored under:

```text
data/agent_outputs/q4_2025_ai_tech/
```

Use `force=True` or `FORCE_RERUN=True` only when you intentionally want to regenerate outputs.

### A run was interrupted

Rerun the cache-aware cells. The orchestration code loads existing ticker files and only needs to regenerate missing outputs when `force=False`.

## Evaluation Issues

### CIO result exists for only one ticker

This usually means the last run used a one-row `summary_df`. Rebuild from cache using the full summary table:

```python
cio_eval, cio_results, cio_summary = q4.rebuild_cio_eval_from_cache(
    summary_df,
    cache_root=CACHE_ROOT,
)
```

### Long-horizon returns are missing

The long horizon uses trading days after the earnings-date close. If the evaluation date is too recent, there may not be enough future price data. Reduce `LONG_POST_TRADING_DAYS` or wait until enough data exists.

## Tests

Run the test suite from the repository root:

```bash
pytest
```

The current tests avoid live API calls and focus on deterministic helpers, packet normalization, debate triggering, and evaluation scoring.
