# Product Review — Sentiment & Action Dashboard

Quick scaffold for a 6-hour PoC that ingests product reviews, clusters themes, computes sentiment, and auto-suggests improvements.

Structure:
- `app/dashboard/` — Streamlit app entrypoint
- `app/ingestion/` — CSV/API ingestion helpers
- `app/embeddings/` — embeddings wrapper
- `app/clustering/` — clustering logic
- `app/agent/` — LLM agent to propose improvements
- `app/observability/` — monitoring helpers (sentiment drift, topic distribution)
- `deployment/` — Dockerfile + Azure deploy hints
- `sample_data/reviews.csv` — example input

Quick start

1. Create and activate Python venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Set environment variables (example):

```powershell
$env:OPENAI_API_KEY = "<your-key>"
$env:AZURE_APPINSIGHTS_KEY = "<your-key>"
```

3. Run Streamlit demo:

```powershell
streamlit run app/dashboard/streamlit_app.py
```

Model selection

- The scaffold uses `gpt-5-mini` by default. To override, set `AZURE_OPENAI_MODEL` in your environment or `.env`.

Notes
- This repo is scaffolded for fast iteration; replace placeholder implementations with production-grade modules (auth, rate-limiting, robust sentiment models, etc.).
- See `deployment/Dockerfile` for a simple container.
