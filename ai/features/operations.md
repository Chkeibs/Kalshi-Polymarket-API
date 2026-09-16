# Operations

## Safe setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
```

Only Kalshi read-only WebSocket access needs local credentials. Never commit `.env`. No OpenAI or Polymarket wallet key is required for the default API path.

## Common commands

```bash
uvicorn api.main:app --reload
python3 -m compileall api pipeline tests main.py
python3 -m pytest -q tests/test_fetch_api_compat.py tests/test_scanner_api_compat.py tests/test_arbitrage_engine.py
python3 -m pytest -q
python3 pipeline/run_clustering_pipeline.py
```

`pipeline/run_clustering_pipeline.py` makes no LLM call. It uses existing normalized data; `--fetch` adds an external market-data refresh.

Avoid `pipeline/run_full_pipeline.py`, `/pipeline/refresh`, paid verifier modes, and long live sessions unless required by the task. They can call external services or rewrite data artifacts.

## Git and security

- Active collaboration branch: `main`.
- Canonical remote: `https://github.com/Chkeibs/Kalshi-Polymarket-Arbitrage-Simulation.git`.
- Do not force-push or overwrite `main`.
- Inspect `git status --short` before edits and before committing.
- Scan tracked files for secrets before sharing.
