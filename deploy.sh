#!/bin/bash
set -e

# Move to project directory
cd $DOM_PROJECT_ROOT

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Copy model file if uploaded
if [ -f "best_model.pth" ]; then
    echo "Model file detected"
else
    echo "ERROR: best_model.pth missing!"
fi

# Kill previous running app
pkill -f "uvicorn" || true

# Start FastAPI app (daemon mode)
nohup venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
echo "Backend running on port 8000"

exit 0
