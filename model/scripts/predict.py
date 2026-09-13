#!/usr/bin/env python
"""
Inference script for iceberg drift prediction.

Usage:
    python scripts/predict.py --model output/checkpoints/best_model.pt --lat -65.0 --lon 0.0
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceberg_drift.utils import DriftPredictor, load_model_for_inference


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(description="Predict iceberg drift trajectory")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--lat", type=float, required=True, help="Initial latitude")
    parser.add_argument("--lon", type=float, required=True, help="Initial longitude")
    parser.add_argument("--horizon", type=int, default=72, help="Prediction horizon (hours)")
    parser.add_argument("--length", type=float, default=2000, help="Iceberg length (m)")
    parser.add_argument("--width", type=float, default=1000, help="Iceberg width (m)")
    parser.add_argument("--output", type=str, default="output/prediction", help="Output directory")
    parser.add_argument("--format", type=str, default="all", choices=["geojson", "netcdf", "csv", "gpx", "all"])
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor
    logger.info(f"Loading model from {args.model}")
    predictor = load_model_for_inference(
        args.model,
        prediction_horizon_hours=args.horizon,
        device=args.device,
    )

    # For demo, generate synthetic environmental forcing
    # In production, this would come from actual forecasts
    logger.info("Generating synthetic environmental forcing...")
    # Historical timestamps
    base_time = datetime.now(timezone.utc)
    hist_times = pd.date_range(base_time - timedelta(hours=24), base_time, freq="6h")
    n_hist = len(hist_times)
    fut_times = pd.date_range(base_time + timedelta(hours=6), base_time + timedelta(hours=args.horizon), freq="6h")

    # Synthetic historical positions (stationary for simplicity)
    hist_lats = np.full(n_hist, args.lat)
    hist_lons = np.full(n_hist, args.lon)

    # Synthetic environmental forcing
    # In reality, these would come from ERA5, Copernicus forecasts
    np.random.seed(42)
    hist_current_u = np.random.normal(0.05, 0.1, n_hist)
    hist_current_v = np.random.normal(-0.02, 0.1, n_hist)
    hist_wind_u = np.random.normal(5, 8, n_hist)
    hist_wind_v = np.random.normal(-3, 8, n_hist)

    # Run prediction
    logger.info(f"Predicting trajectory for iceberg at ({args.lat}, {args.lon})")
    result = predictor.predict(
        timestamps=hist_times,
        latitudes=hist_lats,
        longitudes=hist_lons,
        current_u=hist_current_u,
        current_v=hist_current_v,
        wind_u=hist_wind_u,
        wind_v=hist_wind_v,
        iceberg_length=args.length,
        iceberg_width=args.width,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Initial position: ({args.lat:.3f}, {args.lon:.3f})")
    logger.info(f"Final position: ({result['latitudes'][-1]:.3f}, {result['longitudes'][-1]:.3f})")

    # Calculate total drift distance if possible
    total_distance_km = "N/A"
    if 'latitudes' in result and 'longitudes' in result:
        lats = result['latitudes']
        lons = result['longitudes']
        if len(lats) > 1 and len(lons) > 1:
            # Haversine formula to calculate distance between consecutive points
            from math import radians, cos, sin, asin, sqrt
            def haversine(lat1, lon1, lat2, lon2):
                # Convert decimal degrees to radians
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                # Radius of earth in kilometers
                r = 6371
                return c * r

            total_distance = 0.0
            for i in range(1, len(lats)):
                total_distance += haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            total_distance_km = f"{total_distance:.1f}"

    logger.info(f"Total drift distance: {total_distance_km} km")
    logger.info(f"Prediction horizon: {args.horizon} hours")

    # Export results
    logger.info("=" * 60)
    logger.info("EXPORTING RESULTS")
    logger.info("=" * 60)

    from iceberg_drift.utils import export_to_geojson, export_to_netcdf, export_to_csv, export_route_for_navigation

    trajectories = {
        "predicted": {
            "latitudes": result["latitudes"],
            "longitudes": result["longitudes"],
            "timestamps": result["timestamps"],
            "velocities_u": result["velocities_u"],
            "velocities_v": result["velocities_v"],
        }
    }

    if "physics_latitudes" in result:
        trajectories["physics"] = {
            "latitudes": result["physics_latitudes"],
            "longitudes": result["physics_longitudes"],
            "timestamps": result["timestamps"],
            "velocities_u": result["physics_velocities_u"],
            "velocities_v": result["physics_velocities_v"],
        }

    if "uncertainty_km" in result:
        trajectories["predicted"]["uncertainty_km"] = result["uncertainty_km"]

    if args.format in ["geojson", "all"]:
        export_to_geojson(
            trajectories,
            str(output_dir / "trajectory.geojson"),
            properties={
                "model": args.model,
                "init_lat": args.lat,
                "init_lon": args.lon,
                "horizon_hours": args.horizon,
                "iceberg_length_m": args.length,
                "iceberg_width_m": args.width,
            },
        )

    if args.format in ["netcdf", "all"]:
        export_to_netcdf(
            {"u": result["velocities_u"][None, :], "v": result["velocities_v"][None, :]},
            [{"init_lat": args.lat, "init_lon": args.lon, "iceberg_id": "pred_001"}],
            str(output_dir / "trajectory.nc"),
        )

    if args.format in ["csv", "all"]:
        export_to_csv(
            {"u": result["velocities_u"][None, :], "v": result["velocities_v"][None, :]},
            [{"init_lat": args.lat, "init_lon": args.lon, "iceberg_id": "pred_001"}],
            str(output_dir / "trajectory.csv"),
        )

    if args.format in ["gpx", "all"]:
        export_route_for_navigation(
            trajectories["predicted"],
            str(output_dir / "route.gpx"),
            format="gpx",
        )

    # Also save a summary JSON
    import json
    summary = {
        "init_lat": args.lat,
        "init_lon": args.lon,
        "final_lat": float(result["latitudes"][-1]),
        "final_lon": float(result["longitudes"][-1]),
        "horizon_hours": args.horizon,
        "iceberg_length_m": args.length,
        "iceberg_width_m": args.width,
        "n_steps": len(result["latitudes"]) - 1,
        "timestamps": [str(t) for t in result["timestamps"]],
        "latitudes": result["latitudes"].tolist(),
        "longitudes": result["longitudes"].tolist(),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()