# GC Gold Futures AI Trading System

A full stack gold futures trading system using:
- **Databento** — CME GC continuous contract data
- **QuestDB** — tick storage + feature store
- **LightGBM + PyTorch LSTM** — signal generation
- **Backtrader** — backtesting
- **Topstep / ProjectX API** — live execution

---

## Project Structure

```
gc_trading/
├── README.md                    ← You are here
├── requirements.txt             ← All Python dependencies
├── .env.example                 ← API keys template (copy to .env)
├── config.py                    ← Central config (timeframes, thresholds, etc.)
│
├── data/
│   ├── fetch_databento.py       ← Pull GC historical + live data
│   ├── ingest_questdb.py        ← Write ticks into QuestDB
│   └── fetch_alt_data.py        ← FRED macro + COT reports
│
├── models/
│   ├── features.py              ← Feature engineering pipeline
│   ├── train_lgbm.py            ← LightGBM walk-forward training
│   ├── train_lstm.py            ← PyTorch LSTM training
│   ├── ensemble.py              ← Combined signal generation
│   └── shap_analysis.py        ← Feature importance / debugging
│
├── strategies/
│   └── backtrader_strategy.py  ← Backtrader backtesting harness
│
├── execution/
│   ├── topstep_bot.py           ← Live trading bot (ProjectX API)
│   └── risk_guard.py            ← Topstep rule enforcement
│
├── utils/
│   └── questdb_client.py        ← QuestDB query helpers
│
└── notebooks/
    └── research.ipynb           ← Paperspace Gradient notebook
```

---

## Setup Instructions

### Step 1 — Get Your API Keys

1. **Databento**: Sign up at [databento.com](https://databento.com) → API Keys → Create key. You get $125 free credits.
2. **Topstep**: Subscribe to a Trading Combine at [topstep.com](https://topstep.com) ($49–$149/mo)
3. **TopstepX API**: In your TopstepX dashboard → Subscriptions → ProjectX API Access ($14.50/mo with code `topstep`)
4. **FRED**: Free API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

### Step 2 — Set Up Paperspace Gradient (Training)

Paperspace is used for **weekly model retraining only**. Live execution runs on your local machine.

1. Go to [paperspace.com](https://paperspace.com) and create an account
2. Click **Notebooks** → **Create Notebook**
3. Select runtime: **PyTorch 2.x** (has CUDA pre-installed)
4. Select machine: **A100** for full training, **P4000** ($0.45/hr) for development
5. Click **Start Notebook**
6. In the Jupyter terminal, run:

```bash
# Clone or upload this project
git clone https://github.com/YOUR_USERNAME/gc_trading.git
cd gc_trading

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
nano .env
```

7. Run the training pipeline:
```bash
python data/fetch_databento.py      # ~5 min, pulls 5 years of GC hourly bars
python data/fetch_alt_data.py       # Pulls FRED macro + COT data
python models/train_lgbm.py         # ~20 min on A100
python models/train_lstm.py         # ~45 min on A100
```

8. Download the trained models to your local machine:
   - In Paperspace file browser: right-click `models/saved/` → Download
   - Or use the Gradient storage API

**Estimated training cost: ~$4/week on A100**

---

### Step 3 — Set Up QuestDB (Local)

QuestDB runs locally on your machine (free, open source).

**Mac/Linux:**
```bash
# Docker (easiest)
docker run -p 9000:9000 -p 8812:8812 \
  -v $(pwd)/questdb_data:/root/.questdb \
  questdb/questdb:latest
```

**Windows:**
```bash
docker run -p 9000:9000 -p 8812:8812 ^
  -v %cd%\questdb_data:/root/.questdb ^
  questdb/questdb:latest
```

Open [http://localhost:9000](http://localhost:9000) — QuestDB console.

Run the schema setup SQL from `utils/questdb_client.py` → `create_tables()`.

---

### Step 4 — Local Environment (Live Trading)

**⚠️ Topstep prohibits VPS/cloud execution. All live trading must run on your own device.**

```bash
# Install locally
cd gc_trading
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your API keys

# Place downloaded model files in models/saved/
# gc_lgbm_vX.pkl
# gc_lstm_vX.pth

# Test the system (paper trading mode)
python execution/topstep_bot.py --mode demo
```

---

### Step 5 — Go Live

1. Pass the Topstep Trading Combine (6% profit target, respect drawdown rules)
2. Switch bot to live mode:
```bash
python execution/topstep_bot.py --mode live
```

---

## Weekly Workflow

```
Sunday night:
  1. Open Paperspace notebook
  2. Run: python data/fetch_databento.py --incremental
  3. Run: python models/train_lgbm.py
  4. Run: python models/train_lstm.py
  5. Download new models to local machine
  6. Cost: ~$4

Monday–Friday (8:30 AM – 3:58 PM CT):
  1. Start: python execution/topstep_bot.py --mode live
  2. Monitor QuestDB dashboard at localhost:9000
  3. Bot auto-closes all positions by 3:58 PM CT
```

---

## Topstep Rules (Hardcoded in risk_guard.py)

| Rule | Value ($100K account) |
|---|---|
| Daily loss limit | $2,000 |
| Max trailing drawdown | $4,000 |
| Max contracts | 10 |
| No overnight positions | Close by 3:58 PM CT |
| No weekend positions | Close by Friday 3:58 PM CT |
| Consistency target | No single day > 50% of total profit |

Violations = automatic account closure. The `TopstepRiskGuard` class enforces all of these.

---

## Monthly Costs

| Item | Cost |
|---|---|
| Databento Standard | $179 |
| Topstep $100K Combine | $99 |
| TopstepX API | $14.50 |
| Paperspace GPU training | ~$18 |
| QuestDB, FRED, COT data | Free |
| **Total** | **~$310/month** |
