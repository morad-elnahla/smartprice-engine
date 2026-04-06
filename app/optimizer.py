
import numpy as np
import joblib, json

class PriceOptimizer:
    """
    SmartPrice Optimization Engine
    --------------------------------
    بيدخله context → بيطلع:
    - السعر الأمثل لأعلى Revenue
    - الطلب المتوقع
    - الـ Revenue المتوقع
    - مقارنة الـ naive price vs optimal
    """

    def __init__(self, model_path="models/price_model.joblib",
                       scaler_path="models/scaler.joblib",
                       meta_path="models/model_meta.json"):
        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(meta_path) as f:
            self.meta = json.load(f)

    def _build_features(self, ctx: dict) -> np.ndarray:
        riders  = ctx.get("Number_of_Riders", 50)
        drivers = ctx.get("Number_of_Drivers", 25)
        dur     = ctx.get("Expected_Ride_Duration", 60)
        rating  = ctx.get("Average_Ratings", 4.0)
        loyalty = ctx.get("loyalty_encoded", 1)
        return np.array([
            riders, drivers,
            ctx.get("Number_of_Past_Rides", 50),
            rating, dur, loyalty,
            ctx.get("location_encoded", 1),
            ctx.get("vehicle_encoded", 0),
            ctx.get("time_encoded", 1),
            riders / (drivers + 1),
            riders / (dur + 1),
            drivers / (dur + 1),
            rating * loyalty,
            rating * dur,
            int(riders / (drivers + 1) > 2.0),
            int((riders / (drivers + 1) > 2.0) and ctx.get("time_encoded", 1) >= 2),
            ctx.get("Number_of_Past_Rides", 50) * rating,
        ]).reshape(1, -1)

    def _predict_price(self, ctx: dict) -> float:
        feat = self._build_features(ctx)
        feat_sc = self.scaler.transform(feat)
        return float(np.expm1(self.model.predict(feat_sc)[0]))

    def optimize(self, ctx: dict, price_min=10, price_max=600,
                 price_sensitivity=0.008, n_sweep=200) -> dict:
        base_riders = ctx.get("Number_of_Riders", 50)
        naive_price = self._predict_price(ctx)

        # Sweep
        sweep   = np.linspace(price_min, price_max, n_sweep)
        best_rev, best_price, best_demand = -np.inf, naive_price, base_riders

        for p in sweep:
            ctx_p  = ctx.copy()
            pred_p = self._predict_price(ctx_p)
            demand = base_riders * np.exp(-price_sensitivity * pred_p)
            rev    = pred_p * demand
            if rev > best_rev:
                best_rev, best_price, best_demand = rev, pred_p, demand

        naive_demand  = base_riders * np.exp(-price_sensitivity * naive_price)
        naive_revenue = naive_price * naive_demand
        revenue_uplift = best_rev - naive_revenue

        return {
            "optimal_price"    : round(best_price, 2),
            "estimated_demand" : round(best_demand, 1),
            "projected_revenue": round(best_rev, 2),
            "naive_price"      : round(naive_price, 2),
            "naive_revenue"    : round(naive_revenue, 2),
            "revenue_uplift"   : round(revenue_uplift, 2),
            "uplift_pct"       : round(revenue_uplift / max(naive_revenue, 1) * 100, 1),
            "demand_supply_ratio": round(base_riders / (ctx.get("Number_of_Drivers",25) + 1), 2),
        }
