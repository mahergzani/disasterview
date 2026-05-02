#!/usr/bin/env bash
# DisasterView pipeline one-time setup
# Usage: bash setup.sh [--cuda]   (pass --cuda to install GPU torch)
set -euo pipefail

CUDA=${1:-""}

echo "==> Creating virtual environment (if not present)…"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip…"
pip install --upgrade pip setuptools wheel -q

echo "==> Installing PyTorch…"
if [[ "$CUDA" == "--cuda" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
else
    pip install torch torchvision -q
fi

echo "==> Installing core dependencies…"
pip install \
    tqdm numpy Pillow opencv-python \
    yt-dlp \
    "scenedetect[opencv]" \
    supervision \
    roboflow \
    python-dotenv \
    -q

echo "==> Installing CLIP…"
pip install git+https://github.com/openai/CLIP.git -q

echo "==> Installing supervision (for annotation utilities)…"
pip install supervision -q

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then set your Roboflow key before stage 6:"
echo "  export ROBOFLOW_API_KEY=<your-key>"
echo ""
echo "Run the full pipeline:"
echo "  python pipeline.py --all"
echo ""
echo "Or individual stages:"
echo "  python pipeline.py --stage 1"
