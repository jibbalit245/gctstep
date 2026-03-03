#!/bin/bash
# setup_local.sh
# Sets up the project on your LOCAL machine for live trading.
# Topstep prohibits VPS/cloud execution — the bot must run here.
#
# Usage:
#   bash setup_local.sh

set -e

echo "============================================"
echo " GC Gold Futures AI — Local Setup"
echo "============================================"

# 1. Python check
echo ""
echo "[1/6] Checking Python version..."
python3 --version
PYTHON_OK=$(python3 -c "import sys; print('ok' if sys.version_info >= (3,10) else 'fail')")
if [ "$PYTHON_OK" != "ok" ]; then
    echo "  ✗ Python 3.10+ required"
    exit 1
fi
echo "  ✓ Python OK"

# 2. Install deps
echo ""
echo "[2/6] Installing Python dependencies..."
pip install -r requirements.txt -q
echo "  ✓ Done"

# 3. Create dirs
echo ""
echo "[3/6] Creating directories..."
mkdir -p data/cache models/saved outputs
echo "  ✓ Done"

# 4. .env setup
echo ""
echo "[4/6] Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  ⚠️  .env created. Fill in ALL keys before running the bot:"
    echo ""
    echo "    DATABENTO_API_KEY   → https://databento.com → API Keys"
    echo "    FRED_API_KEY        → https://fred.stlouisfed.org/docs/api/api_key.html"
    echo "    TOPSTEP_USERNAME    → Your TopstepX login username"
    echo "    TOPSTEP_API_KEY     → TopstepX dashboard → Subscriptions → ProjectX API"
    echo "    TRADING_MODE        → Start with 'demo' !"
    echo ""
    echo "    nano .env   (or open in any text editor)"
else
    echo "  ✓ .env already exists"
fi

# 5. Docker / QuestDB check
echo ""
echo "[5/6] Checking Docker for QuestDB..."
if command -v docker &> /dev/null; then
    echo "  ✓ Docker found"
    
    # Check if questdb container already running
    if docker ps | grep -q questdb; then
        echo "  ✓ QuestDB already running"
    else
        echo "  Starting QuestDB..."
        docker run -d --name questdb \
            -p 9000:9000 -p 8812:8812 \
            -v "$(pwd)/questdb_data:/root/.questdb" \
            questdb/questdb:latest
        echo "  ✓ QuestDB started at http://localhost:9000"
        sleep 3
        
        # Create tables
        python3 -c "
from utils.questdb_client import QuestDBClient
qdb = QuestDBClient()
if qdb.health_check():
    qdb.create_tables()
"
    fi
else
    echo "  ✗ Docker not found. Install Docker Desktop: https://docker.com"
    echo "    Then re-run this script, or start QuestDB manually."
fi

# 6. Model files check
echo ""
echo "[6/6] Checking trained models..."
LGBM_EXISTS=$([ -f "models/saved/gc_lgbm_latest.pkl" ] && echo "yes" || echo "no")
LSTM_EXISTS=$([ -f "models/saved/gc_lstm_latest.pth"  ] && echo "yes" || echo "no")

if [ "$LGBM_EXISTS" = "yes" ]; then
    echo "  ✓ LightGBM model found"
else
    echo "  ✗ LightGBM model missing"
    echo "    Train on Paperspace → download → place in models/saved/"
fi

if [ "$LSTM_EXISTS" = "yes" ]; then
    echo "  ✓ LSTM model found"
else
    echo "  ✗ LSTM model missing"
    echo "    Train on Paperspace → download → place in models/saved/"
fi

echo ""
echo "============================================"
echo " Local setup complete!"
echo "============================================"
echo ""
echo "Once models are in models/saved/, run the bot:"
echo ""
echo "  Demo (paper trading):  python execution/topstep_bot.py --mode demo"
echo "  Live (real money):     python execution/topstep_bot.py --mode live"
echo ""
echo "Monitor QuestDB:         http://localhost:9000"
echo ""
