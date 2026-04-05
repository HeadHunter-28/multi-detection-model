#!/bin/bash
set -e

echo "Starting backend at repo root..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --port 8000
