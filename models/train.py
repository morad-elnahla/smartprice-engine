"""
SmartPrice Engine - Model Training Script
Dataset: Dynamic Pricing Dataset (Kaggle - arashnic/dynamic-pricing-dataset)
Target: Historical_Cost_of_Ride
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/dynamic_pricing.csv")
OUTPUT_DIR  = os.path.dirname(__file__)

# ── Load Data ──────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows | Columns: {list(df.columns)}")
    return df

# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    df = df.copy()

    # Demand-supply ratio
    df["demand_supply_ratio"] = df["Number_of_Riders"] / (df["Number_of_Drivers"] + 1)

    # Loyalty encoding
    loyalty_map = {"Silver": 0, "Regular": 1, "Gold": 2}
    df["loyalty_encoded"] = df["Customer_Loyalty_Status"].map(loyalty_map).fillna(1)

    # Location encoding
    location_map = {"Rural": 0, "Suburban": 1, "Urban": 2}
    df["location_encoded"] = df["Location_Category"].map(location_map).fillna(1)

    # Vehicle encoding
    le_vehicle = LabelEncoder()
    df["vehicle_encoded"] = le_vehicle.fit_transform(df["Vehicle_Type"].astype(str))

    # Time of booking encoding
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    df["time_encoded"] = df["Time_of_Booking"].map(time_map).fillna(1)

    return df, le_vehicle

# ── Feature Columns ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Number_of_Riders",
    "Number_of_Drivers",
    "demand_supply_ratio",
    "loyalty_encoded",
    "location_encoded",
    "vehicle_encoded",
    "time_encoded",
    "Number_of_Past_Rides",
    "Average_Ratings",
    "Expected_Ride_Duration",
]

TARGET_COL = "Historical_Cost_of_Ride"

# ── Train ──────────────────────────────────────────────────────────────────────
def train():
    df = load_data()
    df, le_vehicle = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"[RESULT] MAE={mae:.2f} | R²={r2:.4f}")

    # Save artifacts
    joblib.dump(model,      os.path.join(OUTPUT_DIR, "model.joblib"))
    joblib.dump(scaler,     os.path.join(OUTPUT_DIR, "scaler.joblib"))
    joblib.dump(le_vehicle, os.path.join(OUTPUT_DIR, "le_vehicle.joblib"))
    print("[INFO] Artifacts saved ✅")

    return mae, r2

if __name__ == "__main__":
    train()
