"""Visualization utilities for iceberg drift prediction."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    logging.warning("Cartopy not available - geographic plots will use simple projection")

logger = logging.getLogger(__name__)


# =============================================================================
# Color Schemes
# =============================================================================

# Custom colormaps for drift visualization
DRIFT_CMAP = LinearSegmentedColormap.from_list(
    "drift",
    ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]
)

UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list(
    "uncertainty",
    ["#00429d", "#3265c8", "#5d88e3", "#86abf2", "#b3cdf9", "#e0e0e0", "#f9c8a8", "#f1a36b", "#e67e22", "#d35400", "#c0392b"]
)


# =============================================================================
# Static Trajectory Plots
# =============================================================================

def plot_trajectory(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Iceberg Drift Trajectory",
    figsize: Tuple[int, int] = (12, 10),
    projection: str = "polar",
    show_start_end: bool = True,
    color_by: Optional[np.ndarray] = None,
    color_label: str = "Time",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single iceberg trajectory.

    Args:
        latitudes: Latitude array (degrees)
        longitudes: Longitude array (degrees)
        timestamps: Optional timestamps for each point
        title: Plot title
        figsize: Figure size
        projection: "polar" (SouthPolarStereo) or "platecarree"
        show_start_end: Mark start/end points
        color_by: Values to color trajectory by (e.g., time, speed, uncertainty)
        color_label: Label for colorbar
        output_path: Save path

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    if projection == "polar" and CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        transform = None

    # Plot trajectory
    if color_by is not None and len(color_by) == len(latitudes):
        # Color by value - use scatter for continuous coloring
        scatter = ax.scatter(
            longitudes, latitudes,
            c=color_by, cmap=DRIFT_CMAP,
            s=30, alpha=0.8, edgecolors='white', linewidth=0.5,
            transform=transform,
        )
        plt.colorbar(scatter, ax=ax, label=color_label, shrink=0.8)
        # Also plot line
        ax.plot(longitudes, latitudes, 'k-', alpha=0.3, linewidth=1, transform=transform)
    else:
        ax.plot(longitudes, latitudes, 'b-', linewidth=2, alpha=0.7, transform=transform)

    # Mark start and end
    if show_start_end:
        ax.plot(longitudes[0], latitudes[0], 'go', markersize=12, label='Start',
               transform=transform, zorder=5)
        ax.plot(longitudes[-1], latitudes[-1], 'ro', markersize=12, label='End',
               transform=transform, zorder=5)
        ax.legend()

    # Add timestamps as annotations if provided
    if timestamps is not None:
        n_annotations = min(10, len(timestamps))
        indices = np.linspace(0, len(timestamps) - 1, n_annotations, dtype=int)
        for idx in indices:
            if projection == "polar" and CARTOPY_AVAILABLE:
                ax.annotate(
                    str(timestamps[idx])[:10],
                    (longitudes[idx], latitudes[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7,
                    transform=transform,
                )

    ax.set_title(title, fontsize=14, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Trajectory plot saved to {output_path}")

    return fig


def plot_trajectory_comparison(
    true_latitudes: np.ndarray,
    true_longitudes: np.ndarray,
    pred_latitudes: np.ndarray,
    pred_longitudes: np.ndarray,
    physics_latitudes: Optional[np.ndarray] = None,
    physics_longitudes: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Trajectory Comparison: Predicted vs True",
    figsize: Tuple[int, int] = (14, 10),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted vs true trajectories with optional physics baseline.

    Args:
        true_latitudes, true_longitudes: Ground truth trajectory
        pred_latitudes, pred_longitudes: Predicted trajectory
        physics_latitudes, physics_longitudes: Physics-only baseline (optional)
        timestamps: Optional timestamps
        title: Plot title
        figsize: Figure size
        output_path: Save path

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        transform = None

    # True trajectory
    ax.plot(true_longitudes, true_latitudes, 'b-', linewidth=3, alpha=0.8,
           label='True', transform=transform, zorder=4)

    # Predicted trajectory
    ax.plot(pred_longitudes, pred_latitudes, 'r--', linewidth=2, alpha=0.8,
           label='Predicted (ML+Physics)', transform=transform, zorder=3)

    # Physics baseline
    if physics_latitudes is not None and physics_longitudes is not None:
        ax.plot(physics_longitudes, physics_latitudes, 'g:', linewidth=2, alpha=0.7,
               label='Physics Only', transform=transform, zorder=2)

    # Start/end markers
    ax.plot(true_longitudes[0], true_latitudes[0], 'go', markersize=14,
           label='Start', transform=transform, zorder=5)
    ax.plot(true_longitudes[-1], true_latitudes[-1], 'bo', markersize=10,
           label='True End', transform=transform, zorder=5)
    ax.plot(pred_longitudes[-1], pred_latitudes[-1], 'ro', markersize=10,
           label='Pred End', transform=transform, zorder=5)

    if physics_latitudes is not None:
        ax.plot(physics_longitudes[-1], physics_latitudes[-1], 'go', markersize=10,
               label='Physics End', transform=transform, zorder=5)

    # Error vectors (connect true and pred at each timestep)
    if len(true_latitudes) == len(pred_latitudes):
        for i in range(0, len(true_latitudes), max(1, len(true_latitudes) // 20)):
            ax.plot(
                [true_longitudes[i], pred_longitudes[i]],
                [true_latitudes[i], pred_latitudes[i]],
                'k-', alpha=0.2, linewidth=0.5, transform=transform
            )

    ax.legend(loc='lower left', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add error statistics as text
    position_errors = np.sqrt(
        (true_longitudes - pred_longitudes)**2 +
        (true_latitudes - pred_latitudes)**2
    ) * 111  # Approximate km

    stats_text = (
        f"Mean Error: {np.mean(position_errors):.1f} km\n"
        f"Median Error: {np.median(position_errors):.1f} km\n"
        f"Max Error: {np.max(position_errors):.1f} km\n"
        f"Final Error: {position_errors[-1]:.1f} km"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {output_path}")

    return fig


def plot_error_heatmap(
    error_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    title: str = "Position Error Heatmap",
    figsize: Tuple[int, int] = (12, 10),
    vmin: float = 0,
    vmax: Optional[float] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot spatial heatmap of prediction errors.

    Args:
        error_grid: 2D array of errors (km)
        lat_grid: 2D latitude grid
        lon_grid: 2D longitude grid
        title: Plot title
        figsize: Figure size
        vmin, vmax: Color scale limits
        output_path: Save path

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(draw_labels=True)
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        transform = None

    if vmax is None:
        vmax = np.nanpercentile(error_grid, 95)

    im = ax.pcolormesh(
        lon_grid, lat_grid, error_grid,
        cmap=UNCERTAINTY_CMAP,
        vmin=vmin, vmax=vmax,
        shading='auto',
        transform=transform,
    )

    plt.colorbar(im, ax=ax, label='Position Error (km)', shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error heatmap saved to {output_path}")

    return fig


def plot_skill_scores(
    horizon_hours: List[int],
    skill_persistence: List[float],
    skill_physics: Optional[List[float]] = None,
    title: str = "Model Skill Score by Prediction Horizon",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot skill scores vs persistence and physics baselines."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(horizon_hours, skill_persistence, 'o-', linewidth=2, markersize=8,
           label='vs Persistence', color='#2c3e50')

    if skill_physics is not None:
        ax.plot(horizon_hours, skill_physics, 's-', linewidth=2, markersize=8,
               label='vs Physics', color='#e74c3c')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Perfect')

    ax.set_xlabel('Prediction Horizon (hours)', fontsize=12)
    ax.set_ylabel('Skill Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)

    # Add interpretation bands
    ax.axhspan(0, 1, alpha=0.1, color='green', label='Better than baseline')
    ax.axhspan(-0.5, 0, alpha=0.1, color='red', label='Worse than baseline')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Skill scores plot saved to {output_path}")

    return fig


def plot_error_by_horizon(
    horizon_hours: List[int],
    metrics_by_horizon: Dict[int, Dict[str, float]],
    metric_name: str = "rmse_position_km",
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot error metric as function of prediction horizon."""
    if title is None:
        title = f"{metric_name} by Prediction Horizon"

    fig, ax = plt.subplots(figsize=figsize)

    horizons = sorted(metrics_by_horizon.keys())
    values = [metrics_by_horizon[h].get(metric_name, np.nan) for h in horizons]

    ax.plot(horizons, values, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.fill_between(horizons, 0, values, alpha=0.1, color='#3498db')

    ax.set_xlabel('Prediction Horizon (hours)', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Horizon error plot saved to {output_path}")

    return fig


# =============================================================================
# Animation
# =============================================================================

def create_trajectory_animation(
    trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]],
    timestamps: np.ndarray,
    title: str = "Iceberg Drift Prediction",
    figsize: Tuple[int, int] = (12, 10),
    fps: int = 5,
    interval: int = 200,
    output_path: Optional[str] = None,
    show_uncertainty: Optional[Dict[str, np.ndarray]] = None,
) -> animation.FuncAnimation:
    """
    Create animated trajectory comparison.

    Args:
        trajectories: Dict of name -> (lats, lons) arrays
        timestamps: Array of timestamps for each frame
        title: Animation title
        figsize: Figure size
        fps: Frames per second
        interval: Milliseconds between frames
        output_path: Save path (as .mp4 or .gif)
        show_uncertainty: Dict of name -> uncertainty radius (km) per frame

    Returns:
        Matplotlib animation object
    """
    if not CARTOPY_AVAILABLE:
        logger.warning("Cartopy required for animated geographic plots")
        return None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    transform = ccrs.PlateCarree()

    # Colors for different trajectories
    colors = {
        'True': 'blue',
        'Predicted': 'red',
        'Physics': 'green',
        'Persistence': 'orange',
        'Ensemble_Mean': 'purple',
    }

    # Initialize lines
    lines = {}
    uncertainty_patches = {}

    for name, (lats, lons) in trajectories.items():
        color = colors.get(name, 'black')
        line, = ax.plot([], [], '-', color=color, linewidth=2, alpha=0.8,
                       label=name, transform=transform, zorder=3)
        lines[name] = line

        if show_uncertainty and name in show_uncertainty:
            # Uncertainty circle at current position
            circle = plt.Circle((0, 0), 1, color=color, alpha=0.2, transform=transform)
            ax.add_patch(circle)
            uncertainty_patches[name] = circle

    # Current position markers
    markers = {}
    for name in trajectories:
        color = colors.get(name, 'black')
        marker, = ax.plot([], [], 'o', color=color, markersize=10, transform=transform, zorder=5)
        markers[name] = marker

    ax.legend(loc='lower left')
    title_text = ax.set_title(f"{title} - {timestamps[0]}", fontsize=14, fontweight='bold')

    n_frames = len(timestamps)

    def init():
        for line in lines.values():
            line.set_data([], [])
        for marker in markers.values():
            marker.set_data([], [])
        for patch in uncertainty_patches.values():
            patch.center = (0, 0)
            patch.set_radius(0)
        return list(lines.values()) + list(markers.values()) + list(uncertainty_patches.values())

    def animate(frame):
        for name, (lats, lons) in trajectories.items():
            # Trajectory up to current frame
            lines[name].set_data(lons[:frame+1], lats[:frame+1])
            # Current position marker
            markers[name].set_data([lons[frame]], [lats[frame]])

            # Uncertainty
            if name in uncertainty_patches and frame < len(show_uncertainty[name]):
                radius_km = show_uncertainty[name][frame]
                # Convert km to degrees (approximate)
                radius_deg = radius_km / 111.0
                uncertainty_patches[name].center = (lons[frame], lats[frame])
                uncertainty_patches[name].set_radius(radius_deg)

        title_text.set_text(f"{title} - {str(timestamps[frame])[:16]}")

        return list(lines.values()) + list(markers.values()) + list(uncertainty_patches.values())

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=interval, blit=True, repeat=True
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == '.mp4':
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_path, writer=writer)
        elif output_path.suffix == '.gif':
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
        else:
            # Default to mp4
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_path.with_suffix('.mp4'), writer=writer)

        logger.info(f"Animation saved to {output_path}")

    return anim


# =============================================================================
# Ensemble Visualization
# =============================================================================

def plot_ensemble_trajectories(
    ensemble_trajectories: np.ndarray,  # (n_members, n_steps, 2) - (lat, lon)
    true_trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    mean_trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    percentile_trajectories: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    title: str = "Ensemble Trajectories",
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ensemble of trajectories with uncertainty bounds.

    Args:
        ensemble_trajectories: (n_members, n_steps, 2) lat/lon
        true_trajectory: (lats, lons) ground truth
        mean_trajectory: (lats, lons) ensemble mean
        percentile_trajectories: Dict of percentile -> (lats, lons) e.g., {10: ..., 90: ...}
        title: Plot title
        figsize: Figure size
        output_path: Save path
    """
    fig = plt.figure(figsize=figsize)

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        transform = None

    n_members, n_steps, _ = ensemble_trajectories.shape

    # Plot individual members (light)
    for i in range(min(n_members, 50)):  # Limit for visibility
        lats = ensemble_trajectories[i, :, 0]
        lons = ensemble_trajectories[i, :, 1]
        ax.plot(lons, lats, '-', color='gray', alpha=0.1, linewidth=0.5, transform=transform)

    # Plot percentile bounds
    if percentile_trajectories:
        for pct, (lats, lons) in sorted(percentile_trajectories.items()):
            alpha = 0.3 if pct in [10, 90] else 0.5
            ax.plot(lons, lats, '--', color='purple', alpha=alpha, linewidth=1.5,
                   label=f'{pct}th percentile', transform=transform)

    # Plot mean
    if mean_trajectory:
        mean_lats, mean_lons = mean_trajectory
        ax.plot(mean_lons, mean_lats, '-', color='purple', linewidth=3, alpha=0.9,
               label='Ensemble Mean', transform=transform, zorder=4)

    # Plot true
    if true_trajectory:
        true_lats, true_lons = true_trajectory
        ax.plot(true_lons, true_lats, '-', color='blue', linewidth=3, alpha=0.9,
               label='True', transform=transform, zorder=5)
        ax.plot(true_lons[0], true_lats[0], 'go', markersize=12, label='Start', transform=transform, zorder=6)
        ax.plot(true_lons[-1], true_lats[-1], 'bo', markersize=10, label='True End', transform=transform, zorder=6)
        if mean_trajectory:
            ax.plot(mean_lons[-1], mean_lats[-1], 'mo', markersize=10, label='Mean End', transform=transform, zorder=6)

    ax.legend(loc='lower left', fontsize=9)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Ensemble plot saved to {output_path}")

    return fig


# =============================================================================
# Dashboard-style Summary Plot
# =============================================================================

def create_validation_dashboard(
    result: Any,  # ValidationResult
    figsize: Tuple[int, int] = (20, 16),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive validation dashboard."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Trajectory comparison (large, spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2], projection=ccrs.SouthPolarStereo() if CARTOPY_AVAILABLE else None)
    if CARTOPY_AVAILABLE:
        ax1.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax1.add_feature(cfeature.COASTLINE)
        ax1.gridlines(draw_labels=True)

    # 2. Error distribution
    ax2 = fig.add_subplot(gs[0, 2])

    # 3. Skill scores
    ax3 = fig.add_subplot(gs[1, 0])

    # 4. Error by horizon
    ax4 = fig.add_subplot(gs[1, 1])

    # 5. Direction error
    ax5 = fig.add_subplot(gs[1, 2])

    # 6. Regional errors (if available)
    ax6 = fig.add_subplot(gs[2, 0])

    # 7. Seasonal errors
    ax7 = fig.add_subplot(gs[2, 1])

    # 8. Size-based errors
    ax8 = fig.add_subplot(gs[2, 2])

    # Fill in plots if data available
    if hasattr(result, 'predictions') and hasattr(result, 'targets'):
        pred_u = result.predictions.get('u', np.array([]))
        pred_v = result.predictions.get('v', np.array([]))
        true_u = result.targets.get('u', np.array([]))
        true_v = result.targets.get('v', np.array([]))

        if len(pred_u) > 0:
            # Error distribution
            pos_errors = np.sqrt((pred_u - true_u)**2 + (pred_v - true_v)**2).flatten() * 111
            ax2.hist(pos_errors, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(pos_errors), color='red', linestyle='--')
            ax2.set_xlabel('Position Error (km)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Error Distribution')

            # Skill scores
            if hasattr(result, 'metrics') and result.metrics.horizon_metrics:
                horizons = sorted(result.metrics.horizon_metrics.keys())
                skills = [result.metrics.horizon_metrics[h].get('skill_vs_persistence', np.nan) for h in horizons]
                ax3.plot(horizons, skills, 'o-', label='vs Persistence')
                ax3.axhline(y=0, color='gray', linestyle='--')
                ax3.set_xlabel('Horizon (h)')
                ax3.set_ylabel('Skill Score')
                ax3.set_title('Skill Scores')
                ax3.legend()

            # Error by horizon
            if hasattr(result, 'metrics') and result.metrics.horizon_metrics:
                rmses = [result.metrics.horizon_metrics[h].get('rmse_position_km', np.nan) for h in horizons]
                ax4.plot(horizons, rmses, 's-', color='orange')
                ax4.set_xlabel('Horizon (h)')
                ax4.set_ylabel('RMSE (km)')
                ax4.set_title('RMSE by Horizon')

            # Direction error
            pred_dir = np.degrees(np.arctan2(pred_v, pred_u)) % 360
            true_dir = np.degrees(np.arctan2(true_v, true_u)) % 360
            dir_err = np.abs(pred_dir - true_dir)
            dir_err = np.minimum(dir_err, 360 - dir_err).flatten()
            ax5.hist(dir_err, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax5.axvline(np.mean(dir_err), color='red', linestyle='--')
            ax5.set_xlabel('Direction Error (deg)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Direction Error')

    fig.suptitle('Iceberg Drift Model Validation Dashboard', fontsize=16, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Dashboard saved to {output_path}")

    return fig