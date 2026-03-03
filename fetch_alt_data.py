#!/bin/bash
# setup_paperspace.sh
# One-command setup for Paperspace Gradient.
# Run this in the Paperspace terminal after uploading the project.
#
# Usage:
#   bash setup_paperspace.sh

set -e

echo "============================================"
echo " GC Gold Futures AI — Paperspace Setup"
echo "============================================"

# 1. Check GPU
echo ""
echo "[1/5] Checking GPU..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 2. Install dependencies
echo ""
echo "[2/5] Installing Python dependencies (~2 min)..."
pip install -r requirements.txt -q
echo "     Done."

# 3. Create directories
echo ""
echo "[3/5] Creating directories..."
mkdir -p data/cache models/saved outputs notebooks
echo "     Done."

# 4. Check .env
echo ""
echo "[4/5] Checking .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  ⚠️  .env file created from template."
    echo "  You MUST edit it and add your API keys before continuing:"
    echo ""
    echo "    nano .env"
    echo ""
    echo "  Keys needed:"
    echo "    DATABENTO_API_KEY   → https://databento.com → API Keys"
    echo "    FRED_API_KEY        → https://fred.stlouisfed.org/docs/api/api_key.html"
    echo ""
    echo "  (TopstepX keys only needed for local bot, not Paperspace)"
else
    echo "     .env already exists."
fi

# 5. Verify imports
echo ""
echo "[5/5] Verifying imports..."
python3 -c "
import databento, lightgbm, torch, backtrader, shap, pandas_ta, questdb
print('  ✓ databento', databento.__version__)
print('  ✓ lightgbm', lightgbm.__version__)
print('  ✓ torch', torch.__version__)
print('  ✓ backtrader', backtrader.__version__)
print('  ✓ shap', shap.__version__)
print('  ✓ All imports OK')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys:   nano .env"
echo "  2. Fetch data:                     python data/fetch_databento.py"
echo "  3. Fetch macro data:               python data/fetch_alt_data.py"
echo "  4. Train LightGBM:                 python models/train_lgbm.py"
echo "  5. Train LSTM:                     python models/train_lstm.py"
echo "  6. Check SHAP:                     python models/shap_analysis.py"
echo "  7. Backtest:                       python strategies/backtrader_strategy.py"
echo "  8. Download models from models/saved/ to your local machine"
echo ""
echo "  Or open notebooks/research.ipynb and run all cells."
echo ""
