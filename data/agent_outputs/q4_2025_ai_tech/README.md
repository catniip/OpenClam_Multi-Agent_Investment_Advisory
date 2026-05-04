# Q4 2025 AI/Tech Agent Output Cache

This folder stores reusable outputs from the expanded Q4 2025 earnings evaluation.

Generated layout:

- `news_macro/`: News & Macro agent reports by ticker.
- `market_technical/`: Market & Technical agent reports by ticker.
- `fundamental/`: Fundamental agent reports by ticker.
- `contexts/`: News contexts used by the News & Macro agent.
- `packets/`: Standardized CIO-ready packets by ticker.
- `cio/`: CIO workflow outputs, including synthesis, debate, and final decision.
- `errors/`: Per-ticker agent error logs.
- `tables/`: Universe, price summary, price paths, CIO eval, and aggregate summaries.

The notebook `notebooks/05_q4_2025_expanded_eval.ipynb` can generate these files.
Set `FORCE_RERUN=False` to reuse cached outputs.
