# Rubric Checklist

| Rubric criterion | Where to review |
| --- | --- |
| Clean and organized code structure | `src/openclam/agents/`, `src/openclam/evaluation/`, `notebooks/`, `docs/`, `tests/` |
| Comprehensive documentation | `README.md`, `docs/system_design.md`, `docs/evaluation_plan.md`, `docs/proposal_summary.md` |
| Unit tests and error handling | `tests/`, cache-aware rerun helpers, provider fallbacks, per-ticker error files |
| Code optimization | Cached agent outputs, no-debate direct aggregation, incremental ticker reruns |
| Successful implementation of proposed features | News/Macro, Market/Technical, Fundamental, and CIO agents |
| Robust error handling | Missing-key handling, LLM fallback paths, `errors/` cache folder, troubleshooting guide |
| Performance optimization | Reusable cache under `data/agent_outputs/q4_2025_ai_tech/` |
| Integration testing | `tests/test_cache_workflow.py`, `tests/test_cio_agent.py`, full workflow notebooks |
| Reproducible experiments | `notebooks/05_q4_2025_expanded_eval.ipynb`, saved CSV/JSON outputs |
| Well-documented experimental setup | `docs/evaluation_plan.md` |
| Clear presentation of results | cached summary CSVs under `data/agent_outputs/q4_2025_ai_tech/tables/` |
| Analysis scripts and notebooks | five notebooks under `notebooks/` |
| Clear installation instructions | `README.md` Quick Start |
| Environment setup guide | `.env.example`, `environment.yml`, `requirements.txt`, README setup section |
| Usage examples and demonstrations | notebooks `01` through `05`, plus `app.py` Streamlit UI |
| Troubleshooting guide | `docs/troubleshooting.md` |

## Verification

Run:

```bash
pytest
```

Expected current result:

```text
8 passed
```
