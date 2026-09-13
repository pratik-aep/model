"""Export utilities for iceberg drift predictions."""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import json

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# GeoJSON Export
# =============================================================================

def export_to_geojson(
    trajectories: Dict[str, Any],
    output_path: str,
    properties: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export trajectory predictions to GeoJSON format.

    Args:
        trajectories: Dict with keys:
            - 'latitudes', 'longitudes': arrays
            - 'timestamps': array of datetime strings
            - 'velocities_u', 'velocities_v': optional
            - 'uncertainty_km': optional
        output_path: Output file path
        properties: Additional properties for the feature collection

    Returns:
        Path to saved file
    """
    features = []

    for traj_name, traj_data in trajectories.items():
        lats = traj_data.get('latitudes', [])
        lons = traj_data.get('longitudes', [])
        times = traj_data.get('timestamps', [])
        u = traj_data.get('velocities_u', [])
        v = traj_data.get('velocities_v', [])
        uncertainty = traj_data.get('uncertainty_km', [])

        # LineString geometry
        coordinates = [[float(lon), float(lat)] for lat, lon in zip(lats, lons)]

        # Properties for each point
        point_properties = []
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            props = {
                "trajectory": traj_name,
                "sequence": i,
                "latitude": float(lat),
                "longitude": float(lon),
            }
            if i < len(times):
                props["timestamp"] = str(times[i])
            if i < len(u):
                props["velocity_u"] = float(u[i])
                props["velocity_v"] = float(v[i])
                props["speed"] = float(np.sqrt(u[i]**2 + v[i]**2))
            if i < len(uncertainty):
                props["uncertainty_km"] = float(uncertainty[i])
            point_properties.append(props)

        # LineString feature
        line_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "trajectory": traj_name,
                "n_points": len(lats),
                "start_time": str(times[0]) if len(times) > 0 else None,
                "end_time": str(times[-1]) if len(times) > 0 else None,
                "total_distance_km": _compute_path_distance(lats, lons),
            },
        }
        features.append(line_feature)

        # Point features for each timestep
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            point_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": point_properties[i],
            }
            features.append(point_feature)

    # Feature collection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": properties or {
            "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "description": "Iceberg drift trajectory predictions",
        },
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"GeoJSON exported to {output_path} ({len(features)} features)")
    return output_path


def _compute_path_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    """Compute total path distance in km."""
    if len(lats) < 2:
        return 0.0

    dlat = np.diff(lats) * 111.0
    dlon = np.diff(lons) * 111.0 * np.cos(np.deg2rad(lats[:-1]))
    distances = np.sqrt(dlat**2 + dlon**2)
    return float(np.sum(distances))


# =============================================================================
# NetCDF Export (CF-compliant)
# =============================================================================

def export_to_netcdf(
    predictions: Dict[str, np.ndarray],
    metadata: List[Dict],
    output_path: str,
    include_physics: bool = False,
    physics_predictions: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    Export predictions to NetCDF (CF-compliant).

    Args:
        predictions: Dict with 'u', 'v' arrays (n_samples, n_steps)
        metadata: List of dicts with trajectory info
        output_path: Output file path
        include_physics: Whether to include physics baseline
        physics_predictions: Dict with 'u', 'v' for physics baseline

    Returns:
        Path to saved file
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("xarray required for NetCDF export. Install with: pip install xarray netcdf4")

    n_samples, n_steps = predictions['u'].shape

    # Create dataset with proper dimension coordinates
    ds = xr.Dataset(
        coords={
            'trajectory': np.arange(n_samples),
            'step': np.arange(n_steps),
        }
    )

    # Predicted velocities
    ds['predicted_u'] = (('trajectory', 'step'), predictions['u'])
    ds['predicted_v'] = (('trajectory', 'step'), predictions['v'])
    ds['predicted_u'].attrs = {
        'units': 'm/s',
        'long_name': 'Predicted eastward velocity',
        'standard_name': 'eastward_sea_ice_velocity',
    }
    ds['predicted_v'].attrs = {
        'units': 'm/s',
        'long_name': 'Predicted northward velocity',
        'standard_name': 'northward_sea_ice_velocity',
    }

    # Predicted positions (integrated)
    pred_lats, pred_lons = _integrate_all_trajectories(
        metadata, predictions['u'], predictions['v']
    )
    ds['predicted_latitude'] = (('trajectory', 'step'), pred_lats)
    ds['predicted_longitude'] = (('trajectory', 'step'), pred_lons)
    ds['predicted_latitude'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
    ds['predicted_longitude'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}

    # Physics baseline if provided
    if include_physics and physics_predictions:
        ds['physics_u'] = (('trajectory', 'step'), physics_predictions['u'])
        ds['physics_v'] = (('trajectory', 'step'), physics_predictions['v'])
        phys_lats, phys_lons = _integrate_all_trajectories(
            metadata, physics_predictions['u'], physics_predictions['v']
        )
        ds['physics_latitude'] = (('trajectory', 'step'), phys_lats)
        ds['physics_longitude'] = (('trajectory', 'step'), phys_lons)

    # Metadata as trajectory attributes
    traj_attrs = {}
    for key in ['init_lat', 'init_lon', 'length_m', 'width_m', 'iceberg_id']:
        values = [m.get(key, np.nan) for m in metadata]
        traj_attrs[key] = ('trajectory', np.array(values))

    for key, (dim, vals) in traj_attrs.items():
        ds[key] = (dim, vals)
        ds[key].attrs['units'] = 'meters' if 'length' in key or 'width' in key else 'degrees'

    # Global attributes
    ds.attrs['title'] = 'Iceberg Drift Predictions'
    ds.attrs['institution'] = 'Antarctic Navigation DSS'
    ds.attrs['source'] = 'AI-enabled drift prediction model'
    ds.attrs['history'] = f'Created {datetime.now(timezone.utc).isoformat()}Z'
    ds.attrs['conventions'] = 'CF-1.8'

    # Save
    ds.to_netcdf(output_path, format='NETCDF4')
    logger.info(f"NetCDF exported to {output_path}")

    return output_path


def _extract_timestamps(metadata: List[Dict], n_steps: int) -> np.ndarray:
    """Extract timestamps from metadata."""
    timestamps = np.full((len(metadata), n_steps), '', dtype=object)

    for i, meta in enumerate(metadata):
        init_time = meta.get('init_time')
        if init_time is not None:
            if isinstance(init_time, str):
                base = pd.Timestamp(init_time)
            else:
                base = init_time
            # Assume 6-hourly steps
            step_times = [base + pd.Timedelta(hours=6*j) for j in range(n_steps)]
            timestamps[i, :] = [t.isoformat() for t in step_times]

    return timestamps


def _integrate_all_trajectories(
    metadata: List[Dict],
    u_array: np.ndarray,
    v_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate velocities to get lat/lon for all trajectories."""
    n_samples, n_steps = u_array.shape
    lats = np.zeros((n_samples, n_steps + 1))
    lons = np.zeros((n_samples, n_steps + 1))

    for i in range(n_samples):
        init_lat = metadata[i].get('init_lat', -65.0)
        init_lon = metadata[i].get('init_lon', 0.0)

        lats[i, 0] = init_lat
        lons[i, 0] = init_lon

        lat, lon = init_lat, init_lon
        for step in range(n_steps):
            dt = 6 * 3600  # 6 hours in seconds
            lat += (v_array[i, step] * dt) / 111000.0
            lon += (u_array[i, step] * dt) / (111000.0 * np.cos(np.radians(lat)))
            lats[i, step + 1] = lat
            lons[i, step + 1] = lon

    return lats[:, 1:], lons[:, 1:]  # Exclude initial position


# =============================================================================
# CSV Export (Simple)
# =============================================================================

def export_to_csv(
    predictions: Dict[str, np.ndarray],
    metadata: List[Dict],
    output_path: str,
    include_positions: bool = True,
) -> str:
    """
    Export predictions to CSV format.

    Args:
        predictions: Dict with 'u', 'v' arrays
        metadata: List of trajectory metadata
        output_path: Output file path
        include_positions: Whether to include integrated lat/lon

    Returns:
        Path to saved file
    """
    n_samples, n_steps = predictions['u'].shape

    rows = []
    for i in range(n_samples):
        meta = metadata[i]
        for step in range(n_steps):
            row = {
                'trajectory_id': i,
                'iceberg_id': meta.get('iceberg_id', f'traj_{i}'),
                'step': step,
                'pred_u': predictions['u'][i, step],
                'pred_v': predictions['v'][i, step],
                'pred_speed': np.sqrt(predictions['u'][i, step]**2 + predictions['v'][i, step]**2),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if include_positions:
        pred_lats, pred_lons = _integrate_all_trajectories(metadata, predictions['u'], predictions['v'])
        df['pred_lat'] = pred_lats.flatten()
        df['pred_lon'] = pred_lons.flatten()

    df.to_csv(output_path, index=False)
    logger.info(f"CSV exported to {output_path} ({len(df)} rows)")

    return output_path


# =============================================================================
# Route Export (for navigation)
# =============================================================================

def export_route_for_navigation(
    trajectory: Dict[str, Any],
    output_path: str,
    format: str = "gpx",
    vessel_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export trajectory as navigation route (GPX, KML, or CSV waypoints).

    Args:
        trajectory: Single trajectory dict with latitudes, longitudes, timestamps
        output_path: Output file path
        format: "gpx", "kml", or "csv"
        vessel_info: Optional vessel information

    Returns:
        Path to saved file
    """
    lats = trajectory['latitudes']
    lons = trajectory['longitudes']
    times = trajectory.get('timestamps', [])

    if format.lower() == "gpx":
        return _export_gpx(lats, lons, times, output_path, vessel_info)
    elif format.lower() == "kml":
        return _export_kml(lats, lons, times, output_path, vessel_info)
    elif format.lower() == "csv":
        return _export_waypoints_csv(lats, lons, times, output_path, vessel_info)
    else:
        raise ValueError(f"Unknown format: {format}")


def _export_gpx(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    output_path: str,
    vessel_info: Optional[Dict] = None,
) -> str:
    """Export as GPX route."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    gpx = Element('gpx', {
        'version': '1.1',
        'creator': 'Iceberg Drift DSS',
        'xmlns': 'http://www.topografix.com/GPX/1/1',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd',
    })

    # Metadata
    meta = SubElement(gpx, 'metadata')
    name = SubElement(meta, 'name')
    name.text = 'Iceberg Avoidance Route'
    time_elem = SubElement(meta, 'time')
    time_elem.text = datetime.now(timezone.utc).isoformat() + 'Z'

    # Route
    rte = SubElement(gpx, 'rte')
    rte_name = SubElement(rte, 'name')
    rte_name.text = 'Predicted Safe Route'

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        rtept = SubElement(rte, 'rtept', {'lat': str(lat), 'lon': str(lon)})
        if i < len(times):
            rtept_time = SubElement(rtept, 'time')
            rtept_time.text = str(times[i])
        rtept_name = SubElement(rtept, 'name')
        rtept_name.text = f'WP{i+1}'

    # Pretty print
    rough = tostring(gpx, 'utf-8')
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ")

    with open(output_path, 'w') as f:
        f.write(pretty)

    logger.info(f"GPX route exported to {output_path}")
    return output_path


def _export_kml(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    output_path: str,
    vessel_info: Optional[Dict] = None,
) -> str:
    """Export as KML."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    kml = Element('kml', {'xmlns': 'http://www.opengis.net/kml/2.2'})
    doc = SubElement(kml, 'Document')

    name = SubElement(doc, 'name')
    name.text = 'Iceberg Drift Route'

    # Style for route line
    style = SubElement(doc, 'Style', {'id': 'routeStyle'})
    line_style = SubElement(style, 'LineStyle')
    color = SubElement(line_style, 'color')
    color.text = 'ff0000ff'  # Red (ABGR)
    width = SubElement(line_style, 'width')
    width.text = '3'

    # Placemark for route
    pm = SubElement(doc, 'Placemark')
    pm_name = SubElement(pm, 'name')
    pm_name.text = 'Predicted Route'
    pm_style = SubElement(pm, 'styleUrl')
    pm_style.text = '#routeStyle'

    line_string = SubElement(pm, 'LineString')
    tessellate = SubElement(line_string, 'tessellate')
    tessellate.text = '1'
    coords = SubElement(line_string, 'coordinates')

    coord_str = ' '.join([f'{lon},{lat},0' for lat, lon in zip(lats, lons)])
    coords.text = coord_str

    # Waypoints
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        wp = SubElement(doc, 'Placemark')
        wp_name = SubElement(wp, 'name')
        wp_name.text = f'WP{i+1}'
        if i < len(times):
            wp_time = SubElement(wp, 'TimeStamp')
            when = SubElement(wp_time, 'when')
            when.text = str(times[i])

        wp_point = SubElement(wp, 'Point')
        wp_coords = SubElement(wp_point, 'coordinates')
        wp_coords.text = f'{lon},{lat},0'

    rough = tostring(kml, 'utf-8')
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ")

    with open(output_path, 'w') as f:
        f.write(pretty)

    logger.info(f"KML route exported to {output_path}")
    return output_path


def _export_waypoints_csv(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    output_path: str,
    vessel_info: Optional[Dict] = None,
) -> str:
    """Export as CSV waypoints."""
    df = pd.DataFrame({
        'waypoint': range(1, len(lats) + 1),
        'latitude': lats,
        'longitude': lons,
    })

    if len(times) > 0:
        df['timestamp'] = [str(t) for t in times]

    if vessel_info:
        for key, val in vessel_info.items():
            df[key] = val

    df.to_csv(output_path, index=False)
    logger.info(f"Waypoints CSV exported to {output_path}")
    return output_path