"""
SmartPrice Engine — Local Runner
Run this file to train the model and start the server.

Usage:
    python run.py

Requirements:
    pip install -r requirements.txt
    Place dataset at: data/dynamic_pricing.csv
"""

import os
import sys
import subprocess

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "dynamic_pricing.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "price_model.joblib")
def check_data():
    if not os.path.exists(DATA_PATH):
        print("=" * 55)
        print("  ❌  Dataset not found!")
        print("=" * 55)
        print("\n  1. Go to:")
        print("     https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset")
        print("\n  2. Download  dynamic_pricing.csv")
        print("\n  3. Place it at:")
        print(f"     {DATA_PATH}")
        print("\n" + "=" * 55)
        sys.exit(1)
    print("  ✅  Dataset found.")

def train():
    if os.path.exists(MODEL_PATH):
        print("  ✅  Model already trained — skipping.")
        return
    print("  🔄  Training model (first time only)...")
    result = subprocess.run(
        [sys.executable, "models/train.py"],
        cwd=os.path.dirname(__file__)
    )
    if result.returncode != 0:
        print("  ❌  Training failed. Check the error above.")
        sys.exit(1)
    print("  ✅  Model trained and saved.")

def serve():
    print("\n" + "=" * 55)
    print("  🚀  SmartPrice Engine is running!")
    print("  🌐  Open: http://localhost:8000")
    print("=" * 55 + "\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=os.path.dirname(__file__))

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ⚡  SmartPrice Engine — Starting Up")
    print("=" * 55)
    check_data()
    train()
    serve()
