# GC Gold Futures AI Trading System — Dependencies
# Install with: pip install -r requirements.txt

# ── Data ──────────────────────────────────────────────
databento>=0.38.0
pandas>=2.0.0
numpy>=1.24.0
fredapi>=0.5.0
cot-reports>=0.6.0
yfinance>=0.2.40
requests>=2.31.0

# ── QuestDB ───────────────────────────────────────────
questdb>=1.2.0
psycopg2-binary>=2.9.0       # QuestDB Postgres wire protocol

# ── Feature Engineering ───────────────────────────────
pandas-ta>=0.3.14b0
scikit-learn>=1.3.0
scipy>=1.11.0

# ── ML Models ─────────────────────────────────────────
lightgbm>=4.3.0
torch>=2.2.0
optuna>=3.5.0                # Hyperparameter tuning
shap>=0.44.0

# ── Backtesting ───────────────────────────────────────
backtrader>=1.9.78.123

# ── Execution / API ───────────────────────────────────
project_x_py>=3.3.4          # TopstepX / ProjectX SDK
httpx>=0.26.0                # Async HTTP (used by project_x_py)
websockets>=12.0
python-dotenv>=1.0.0

# ── Utilities ─────────────────────────────────────────
joblib>=1.3.0                # Model serialization
python-dateutil>=2.8.0
pytz>=2024.1
tqdm>=4.66.0
loguru>=0.7.0                # Better logging
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0               # For Paperspace notebooks
ipywidgets>=8.0.0
