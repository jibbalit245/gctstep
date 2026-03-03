# ──────────────────────────────────────────────────────────────────────────────
# GC Gold Futures AI Trading System — API Keys & Config
# Copy this file to .env and fill in your values
# NEVER commit .env to git
# ──────────────────────────────────────────────────────────────────────────────

# Databento (https://databento.com → API Keys)
DATABENTO_API_KEY=db-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# FRED Macro Data (https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# TopstepX / ProjectX API
# Get from: TopstepX dashboard → Subscriptions → ProjectX API Access
TOPSTEP_USERNAME=your_topstep_username
TOPSTEP_API_KEY=your_projectx_api_key

# Trading Mode: "demo" (paper trading) or "live" (real money)
TRADING_MODE=demo

# QuestDB (defaults work if running locally via Docker)
QUESTDB_HOST=localhost
QUESTDB_HTTP_PORT=9000
QUESTDB_PG_PORT=8812
QUESTDB_USER=admin
QUESTDB_PASSWORD=quest

# Model Storage (local path to save/load trained models)
MODEL_DIR=./models/saved
