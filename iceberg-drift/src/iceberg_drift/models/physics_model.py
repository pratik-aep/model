"""Physics-based iceberg drift model.

Implements the standard drift model:
- Iceberg velocity = Current velocity + Wind drift factor * Wind velocity + Coriolis correction
- Wind drift factor typically ~1-2% for large icebergs
- Coriolis effect deflects trajectory to the left in Southern Hemisphere
"""

import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Physical Constants
# =============================================================================

EARTH_ROTATION_RATE = 7.2921150e-5  # rad/s
SEAWATER_DENSITY = 1028.0  # kg/m^3
ICE_DENSITY = 917.0  # kg/m^3
AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2


@dataclass
class IcebergParameters:
    """Physical parameters of an iceberg."""
    length_m: float = 2000.0
    width_m: float = 1000.0
    height_m: float = 300.0      # Above waterline (freeboard)
    draft_m: float = 270.0       # Below waterline
    density: float = ICE_DENSITY

    # Drag coefficients
    Cd_water: float = 1.0        # Form drag in water
    Cd_air: float = 1.2          # Form drag in air
    Cw_water: float = 0.02       # Skin friction in water
    Cw_air: float = 0.0012       # Skin friction in air

    @property
    def area_above_water(self) -> float:
        """Cross-sectional area above water (m^2)."""
        return self.length_m * self.height_m

    @property
    def area_below_water(self) -> float:
        """Cross-sectional area below water (m^2)."""
        return self.length_m * self.draft_m

    @property
    def waterline_area(self) -> float:
        """Waterline area (m^2)."""
        return self.length_m * self.width_m

    @property
    def mass(self) -> float:
        """Mass of iceberg (kg)."""
        volume = self.length_m * self.width_m * (self.height_m + self.draft_m)
        return volume * self.density

    @property
    def submerged_volume(self) -> float:
        """Submerged volume (m^3)."""
        return self.length_m * self.width_m * self.draft_m


@dataclass
class EnvironmentalForcing:
    """Environmental forcing at a given time/location."""
    # Ocean current (m/s)
    current_u: float = 0.0
    current_v: float = 0.0

    # Wind at 10m (m/s)
    wind_u: float = 0.0
    wind_v: float = 0.0

    # Location
    latitude: float = -65.0  # degrees
    longitude: float = 0.0

    # Sea state (optional)
    wave_height: float = 0.0
    wave_period: float = 0.0


def compute_coriolis_parameter(latitude: float) -> float:
    """Compute Coriolis parameter f = 2 * Ω * sin(lat)."""
    return 2 * EARTH_ROTATION_RATE * np.sin(np.radians(latitude))


def compute_wind_drag_factor(iceberg: IcebergParameters) -> float:
    """
    Compute wind drift factor (fraction of wind velocity transferred to iceberg).

    Standard approximation: 1-2% for large tabular icebergs.
    More detailed: depends on area ratio, drag coefficients, density ratio.
    """
    # Force balance: F_air = F_water (at steady state)
    # 0.5 * rho_air * Cd_air * A_air * |U_wind - U_iceberg| * (U_wind - U_iceberg)
    # = 0.5 * rho_water * Cd_water * A_water * |U_iceberg - U_current| * (U_iceberg - U_current)

    # Simplified: wind_factor ≈ (rho_air/rho_water) * (Cd_air/Cd_water) * (A_air/A_water)
    area_ratio = iceberg.area_above_water / iceberg.area_below_water
    wind_factor = (AIR_DENSITY / SEAWATER_DENSITY) * (iceberg.Cd_air / iceberg.Cd_water) * area_ratio

    # Typical values: 0.01 to 0.03 (1-3%)
    return np.clip(wind_factor, 0.005, 0.05)


def compute_physics_drift(
    iceberg: IcebergParameters,
    forcing: EnvironmentalForcing,
    dt: float = 3600.0,
) -> Tuple[float, float]:
    """
    Compute iceberg drift velocity using physics-based model.

    Returns:
        (u_drift, v_drift) in m/s (eastward, northward)
    """
    # Wind drift factor
    wind_factor = compute_wind_drag_factor(iceberg)

    # Coriolis parameter
    f = compute_coriolis_parameter(forcing.latitude)

    # Water-relative velocity (iceberg moves with current + wind drift)
    # This is the steady-state solution ignoring acceleration terms
    u_relative = forcing.current_u + wind_factor * forcing.wind_u
    v_relative = forcing.current_v + wind_factor * forcing.wind_v

    # Coriolis deflection (in Southern Hemisphere, f < 0, deflects LEFT)
    # The steady-state with Coriolis: u = u_relative - (f/g) * v_relative * h
    # But for icebergs, we use the standard drift law:
    # u_drift = u_current + wind_factor * u_wind - (f/|f|) * alpha * v_wind
    # v_drift = v_current + wind_factor * v_wind + (f/|f|) * alpha * u_wind
    # where alpha ≈ 0.01-0.02 for typical icebergs

    # Simplified: add Coriolis correction to wind-driven component
    alpha = 0.015  # Coriolis deflection coefficient
    hem_sign = -1.0 if forcing.latitude < 0 else 1.0  # Southern = -1, Northern = +1

    u_drift = (forcing.current_u +
               wind_factor * forcing.wind_u +
               hem_sign * alpha * forcing.wind_v)

    v_drift = (forcing.current_v +
               wind_factor * forcing.wind_v -
               hem_sign * alpha * forcing.wind_u)

    return u_drift, v_drift


def compute_physics_trajectory(
    iceberg: IcebergParameters,
    initial_lat: float,
    initial_lon: float,
    forcing_history: List[EnvironmentalForcing],
    dt: float = 3600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full trajectory by integrating physics drift.

    Args:
        iceberg: IcebergParameters
        initial_lat: Starting latitude
        initial_lon: Starting longitude
        forcing_history: List of EnvironmentalForcing for each timestep
        dt: Timestep in seconds

    Returns:
        (lats, lons) arrays of trajectory
    """
    lats = [initial_lat]
    lons = [initial_lon]

    lat, lon = initial_lat, initial_lon

    for forcing in forcing_history:
        forcing.latitude = lat
        forcing.longitude = lon

        u_drift, v_drift = compute_physics_drift(iceberg, forcing, dt)

        # Convert to lat/lon change
        # 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 km * cos(lat)
        lat += (v_drift * dt) / 111000.0
        lon += (u_drift * dt) / (111000.0 * np.cos(np.radians(lat)))
        # Wrap longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180

        lats.append(lat)
        lons.append(lon)

    return np.array(lats), np.array(lons)


class PhysicsDriftModel:
    """
    Physics-based iceberg drift model.

    Can be used as:
    1. Standalone predictor (pure physics)
    2. Baseline for ML models
    3. Physics component in PINN
    """

    def __init__(self, iceberg_params: Optional[IcebergParameters] = None):
        self.iceberg_params = iceberg_params or IcebergParameters()
        self.wind_factor = compute_wind_drag_factor(self.iceberg_params)

    def predict_step(
        self,
        lat: float,
        lon: float,
        current_u: float,
        current_v: float,
        wind_u: float,
        wind_v: float,
        dt: float = 3600.0,
    ) -> Tuple[float, float]:
        """Predict one step drift velocity."""
        forcing = EnvironmentalForcing(
            current_u=current_u,
            current_v=current_v,
            wind_u=wind_u,
            wind_v=wind_v,
            latitude=lat,
            longitude=lon,
        )
        return compute_physics_drift(self.iceberg_params, forcing, dt)

    def predict_trajectory(
        self,
        init_lat: float,
        init_lon: float,
        current_u_series: np.ndarray,
        current_v_series: np.ndarray,
        wind_u_series: np.ndarray,
        wind_v_series: np.ndarray,
        dt: float = 3600.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict full trajectory given forcing time series."""
        n_steps = len(current_u_series)
        forcing_history = []

        for i in range(n_steps):
            forcing = EnvironmentalForcing(
                current_u=current_u_series[i],
                current_v=current_v_series[i],
                wind_u=wind_u_series[i],
                wind_v=wind_v_series[i],
                latitude=init_lat,  # Will be updated in compute_physics_trajectory
                longitude=init_lon,
            )
            forcing_history.append(forcing)

        return compute_physics_trajectory(
            self.iceberg_params, init_lat, init_lon, forcing_history, dt
        )

    def get_wind_factor(self) -> float:
        """Get the computed wind drift factor."""
        return self.wind_factor


# =============================================================================
# Vectorized version for batch processing
# =============================================================================

def compute_physics_drift_batch(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    wind_factor: float = 0.02,
    coriolis_alpha: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized physics drift computation for batch processing.

    Args:
        All inputs are arrays of same shape
        wind_factor: Fraction of wind velocity (typically 0.01-0.03)
        coriolis_alpha: Coriolis deflection coefficient (typically 0.01-0.02)

    Returns:
        (u_drift, v_drift) arrays
    """
    # Coriolis parameter
    f = 2 * EARTH_ROTATION_RATE * np.sin(np.radians(latitudes))
    hem_sign = np.where(latitudes < 0, -1.0, 1.0)

    # Drift velocity
    u_drift = (current_u +
               wind_factor * wind_u +
               hem_sign * coriolis_alpha * wind_v)

    v_drift = (current_v +
               wind_factor * wind_v -
               hem_sign * coriolis_alpha * wind_u)

    return u_drift, v_drift


def integrate_trajectory_batch(
    init_lats: np.ndarray,
    init_lons: np.ndarray,
    u_drift: np.ndarray,  # (n_trajectories, n_steps)
    v_drift: np.ndarray,  # (n_trajectories, n_steps)
    dt: float = 3600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate trajectories for a batch of initial positions.

    Args:
        init_lats: (n_trajectories,) initial latitudes
        init_lons: (n_trajectories,) initial longitudes
        u_drift: (n_trajectories, n_steps) eastward velocity (m/s)
        v_drift: (n_trajectories, n_steps) northward velocity (m/s)
        dt: Timestep in seconds

    Returns:
        (lats, lons) arrays of shape (n_trajectories, n_steps + 1)
    """
    n_traj, n_steps = u_drift.shape
    lats = np.zeros((n_traj, n_steps + 1))
    lons = np.zeros((n_traj, n_steps + 1))

    lats[:, 0] = init_lats
    lons[:, 0] = init_lons

    for step in range(n_steps):
        # Current position
        lat = lats[:, step]
        lon = lons[:, step]

        # Velocity at this step
        u = u_drift[:, step]
        v = v_drift[:, step]

        # Position update
        lats[:, step + 1] = lat + (v * dt) / 111000.0
        lons[:, step + 1] = lon + (u * dt) / (111000.0 * np.cos(np.radians(lat)))
        # Wrap longitude to [-180, 180]
        lons[:, step + 1] = ((lons[:, step + 1] + 180) % 360) - 180

    return lats, lons