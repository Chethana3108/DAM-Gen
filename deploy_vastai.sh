#!/usr/bin/env bash
# ==============================================================================
# One-Click Deployment Script for Vast.ai GPU Server (PyTorch Docker template)
# ==============================================================================

set -e

echo "🚀 Starting Sitecore DAM GenAI Brand Campaign Engine on GPU..."

# 1. Install GPU dependencies
echo "📦 Installing PyTorch Diffusers & Model Acceleration packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Check GPU status
echo "🔍 Checking NVIDIA GPU status..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 3. Launch FastAPI server with GPU acceleration
echo "🔥 Launching server on port 8000..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
