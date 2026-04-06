from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import joblib, json, os
from pathlib import Path

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartPrice Engine",
    description="Dynamic Pricing API — Revenue Optimization using ML",
    version="2.0.0"
)

BASE = Path(__file__).parent.parent
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

# ─── Load Models ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODELS = BASE / "models"

price_model   = joblib.load(MODELS / "price_model.joblib")
scaler        = joblib.load(MODELS / "scaler.joblib")
demand_model  = joblib.load(MODELS / "demand_model.joblib")
demand_scaler = joblib.load(MODELS / "demand_scaler.joblib")

with open(MODELS / "model_meta.json", encoding="utf-8") as f:
    meta = json.load(f)

# ─── Mappings ─────────────────────────────────────────────────────────────────
LOYALTY_MAP  = {"Silver": 0, "Regular": 1, "Gold": 2}
LOCATION_MAP = {"Rural": 0, "Suburban": 1, "Urban": 2}
TIME_MAP     = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
VEHICLE_MAP  = {"Economy": 0, "Premium": 1}

# ─── Request Schema ───────────────────────────────────────────────────────────
class RideContext(BaseModel):
    number_of_riders:        int   = Field(50,  ge=1,   le=200)
    number_of_drivers:       int   = Field(25,  ge=1,   le=200)
    number_of_past_rides:    int   = Field(50,  ge=0,   le=500)
    average_ratings:         float = Field(4.0, ge=1.0, le=5.0)
    expected_ride_duration:  int   = Field(60,  ge=5,   le=300)
    customer_loyalty_status: str   = Field("Regular")   # Silver / Regular / Gold
    location_category:       str   = Field("Urban")     # Rural / Suburban / Urban
    vehicle_type:            str   = Field("Economy")   # Economy / Premium
    time_of_booking:         str   = Field("Morning")   # Morning / Afternoon / Evening / Night
    # Constraints
    min_price:               float = Field(10.0,  ge=0)
    max_price:               float = Field(600.0, ge=0)
    competitor_price:        float = Field(None,  ge=0)

# ─── Feature Builder ──────────────────────────────────────────────────────────
def build_features(ctx: RideContext) -> np.ndarray:
    riders  = ctx.number_of_riders
    drivers = ctx.number_of_drivers
    dur     = ctx.expected_ride_duration
    rating  = ctx.average_ratings
    loyalty = LOYALTY_MAP.get(ctx.customer_loyalty_status, 1)
    loc     = LOCATION_MAP.get(ctx.location_category, 1)
    time    = TIME_MAP.get(ctx.time_of_booking, 1)
    vehicle = VEHICLE_MAP.get(ctx.vehicle_type, 0)

    ds_ratio = riders / (drivers + 1)

    return np.array([
        riders, drivers,
        ctx.number_of_past_rides,
        rating, dur,
        loyalty, loc, vehicle, time,
        ds_ratio,
        riders  / (dur + 1),
        drivers / (dur + 1),
        rating  * loyalty,
        rating  * dur,
        int(ds_ratio > 2.0),
        int(ds_ratio > 2.0 and time >= 2),
        ctx.number_of_past_rides * rating,
    ]).reshape(1, -1)

# ─── Dynamic Sensitivity ──────────────────────────────────────────────────────
def get_sensitivity(ctx: RideContext) -> float:
    """
    بدل sensitivity ثابت — بيتغير حسب الـ context
    Rush hour + Urban → الناس تقبل سعر أعلى → sensitivity أقل
    """
    base = 0.008
    if TIME_MAP.get(ctx.time_of_booking, 1) >= 2:   # Evening / Night
        base *= 0.75
    if LOCATION_MAP.get(ctx.location_category, 1) == 2:  # Urban
        base *= 0.85
    if ctx.number_of_riders / (ctx.number_of_drivers + 1) > 2:  # High demand
        base *= 0.80
    return base

# ─── Optimizer ────────────────────────────────────────────────────────────────
def optimize_price(ctx: RideContext) -> dict:
    feat       = build_features(ctx)
    feat_sc    = scaler.transform(feat)
    naive_log  = price_model.predict(feat_sc)[0]
    naive_price = float(np.expm1(naive_log))

    # Apply constraints
    max_allowed = ctx.max_price
    if ctx.competitor_price:
        max_allowed = min(max_allowed, ctx.competitor_price * 1.1)

    sensitivity = get_sensitivity(ctx)
    base_riders = ctx.number_of_riders

    # Price sweep
    sweep       = np.linspace(ctx.min_price, max_allowed, 300)
    best_rev    = -np.inf
    best_price  = naive_price
    best_demand = base_riders

    for p in sweep:
        demand  = base_riders * np.exp(-sensitivity * p)
        revenue = p * demand
        if revenue > best_rev:
            best_rev    = revenue
            best_price  = p
            best_demand = demand

    # Clamp naive price to constraints
    naive_price   = min(max(naive_price, ctx.min_price), max_allowed)
    naive_demand  = base_riders * np.exp(-sensitivity * naive_price)
    naive_revenue = naive_price * naive_demand
    uplift        = best_rev - naive_revenue

    return {
        "optimal_price"      : round(best_price,   2),
        "estimated_demand"   : round(best_demand,  1),
        "projected_revenue"  : round(best_rev,     2),
        "naive_price"        : round(naive_price,  2),
        "naive_revenue"      : round(naive_revenue,2),
        "revenue_uplift"     : round(uplift,        2),
        "uplift_pct"         : round(uplift / max(naive_revenue, 1) * 100, 1),
        "demand_supply_ratio": round(ctx.number_of_riders / (ctx.number_of_drivers + 1), 2),
        "price_sensitivity"  : round(sensitivity,  4),
        "constraints_applied": {
            "min_price"        : ctx.min_price,
            "max_price"        : max_allowed,
            "competitor_price" : ctx.competitor_price,
        }
    }

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    BASE = Path(__file__).parent.parent
    return FileResponse(BASE / "static" / "index.html")

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model"  : meta["model_type"],
        "R2"     : meta["metrics"]["R2"],
        "MAPE"   : meta["metrics"]["MAPE"],
    }
@app.post("/predict")
def predict_price(ctx: RideContext):
    try:
        result = optimize_price(ctx)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return meta