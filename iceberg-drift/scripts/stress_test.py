#!/usr/bin/env python

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iceberg_drift.utils import load_model_for_inference

def haversine(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def run_scenario(name, wind_u, wind_v, curr_u, curr_v, predictor, lat=-60.0, lon=-55.0):
    print(f"\n{'='*50}\nSCENARIO: {name}\n{'='*50}")
    
    # 72 hours of history, 6h freq -> 12 steps + current = 13 steps
    base_time = datetime.now(timezone.utc)
    hist_times = pd.date_range(base_time - timedelta(hours=72), base_time, freq="6h")
    n_hist = len(hist_times)
    
    # Stationary historical positions
    hist_lats = np.full(n_hist, lat)
    hist_lons = np.full(n_hist, lon)
    
    # Set constant forcing
    h_wind_u = np.full(n_hist, wind_u)
    h_wind_v = np.full(n_hist, wind_v)
    h_curr_u = np.full(n_hist, curr_u)
    h_curr_v = np.full(n_hist, curr_v)
    
    result = predictor.predict(
        timestamps=hist_times,
        latitudes=hist_lats,
        longitudes=hist_lons,
        current_u=h_curr_u,
        current_v=h_curr_v,
        wind_u=h_wind_u,
        wind_v=h_wind_v,
        iceberg_length=2000,
        iceberg_width=1000,
    )
    
    pred_lats = result["latitudes"]
    pred_lons = result["longitudes"]
    phys_lats = result.get("physics_latitudes", [])
    phys_lons = result.get("physics_longitudes", [])
    
    if len(pred_lats) > 1:
        pred_dist = haversine(pred_lats[0], pred_lons[0], pred_lats[-1], pred_lons[-1])
        print(f"PINN Predicted Drift Distance (72h): {pred_dist:.2f} km")
        print(f"Final Position: Lat {pred_lats[-1]:.3f}, Lon {pred_lons[-1]:.3f}")
    
    if len(phys_lats) > 1:
        phys_dist = haversine(phys_lats[0], phys_lons[0], phys_lats[-1], phys_lons[-1])
        print(f"Physics Baseline Drift Distance: {phys_dist:.2f} km")
        print(f"Physics Final Position: Lat {phys_lats[-1]:.3f}, Lon {phys_lons[-1]:.3f}")

if __name__ == "__main__":
    model_path = "output/checkpoints/pinn_model.pt"
    print(f"Loading model from {model_path}...")
    predictor = load_model_for_inference(
        model_path,
        prediction_horizon_hours=72,
        device="cpu",
    )
    
    # 1. Calm Waters
    run_scenario("Calm Waters (0 wind, 0 current)", 0.0, 0.0, 0.0, 0.0, predictor)
    
    # 2. Hurricane Forcing
    # 30 m/s wind is ~ 108 km/h. Moving North-East
    run_scenario("Hurricane (30m/s NE wind, 0 current)", 21.2, 21.2, 0.0, 0.0, predictor)
    
    # 3. Ocean River
    # 2 m/s current is very strong for ocean. Moving East
    run_scenario("Ocean River (0 wind, 2m/s E current)", 0.0, 0.0, 2.0, 0.0, predictor)
