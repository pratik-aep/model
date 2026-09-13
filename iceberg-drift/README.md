# Iceberg Drift Prediction Model

AI-enabled Antarctic sea-ice and iceberg trajectory forecasting for navigation decision support.

## Overview

This project implements a physics-informed machine learning system for predicting iceberg drift trajectories in Antarctic waters. It combines:

- **Physics-based drift model**: Ocean current + wind drift (1-2%) + Coriolis deflection
- **ML correction model**: LSTM/GRU/Transformer learning residuals from historical data
- **PINN architecture**: Physics-informed neural network with physics-constrained loss
- **Ensemble methods**: Uncertainty quantification for safe navigation

## Features

- 📊 **Data Pipeline**: Automated download/processing of ERA5 wind, Copernicus currents, NIC iceberg positions
- 🧮 **Physics Model**: First-principles iceberg drift with configurable parameters
- 🤖 **ML Models**: LSTM, GRU, Transformer, MLP for sequence-to-sequence prediction
- ⚗️ **PINN**: Physics-informed loss combining data fidelity + physics constraints
- 📈 **Evaluation**: Multi-horizon metrics, bootstrap CIs, baseline comparisons
- 🗺️ **Visualization**: Trajectory plots, animations, error heatmaps, skill scores
- 📤 **Export**: GeoJSON, NetCDF (CF), CSV, GPX/KML for navigation systems

## Installation

```bash
# Create environment
conda create -n iceberg-drift python=3.10
conda activate iceberg-drift

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Train a Model

```bash
# Using synthetic data (for development)
python scripts/train.py --model-type pinn --epochs 50 --batch-size 32

# With custom config
python scripts/train.py --config config/default_config.yaml --model-type hybrid --epochs 100
```

### 2. Make Predictions

```bash
# Predict trajectory for an iceberg
python scripts/predict.py \
    --model output/checkpoints/best_model.pt \
    --lat -65.0 --lon 0.0 \
    --horizon 72 \
    --length 2000 --width 1000 \
    --output output/my_prediction \
    --format all
```

### 3. Use in Python

```python
from iceberg_drift import DriftPredictor, load_model_for_inference

# Load model
predictor = load_model_for_inference("output/checkpoints/best_model.pt")

# Predict with real environmental data
result = predictor.predict(
    timestamps=hist_timestamps,
    latitudes=hist_lats,
    longitudes=hist_lons,
    current_u=current_u, current_v=current_v,
    wind_u=wind_u, wind_v=wind_v,
    iceberg_length=2000, iceberg_width=1000,
)

# Access results
print(f"Final position: {result['latitudes'][-1]:.3f}, {result['longitudes'][-1]:.3f}")
```

## Project Structure

```
iceberg-drift/
├── config/
│   └── default_config.yaml    # Main configuration
├── src/iceberg_drift/
│   ├── config.py              # Configuration management
│   ├── data_processing/       # Data download, preprocessing, datasets
│   │   ├── download.py        # Satellite/oceanographic data download
│   │   ├── preprocessing.py   # Feature engineering, target creation
│   │   └── dataset.py         # PyTorch Dataset/DataLoader
│   ├── models/                # Model architectures
│   │   ├── physics_model.py   # Physics-based drift model
│   │   ├── ml_models.py       # LSTM, GRU, Transformer, MLP
│   │   ├── pinn.py            # Physics-informed neural networks
│   │   └── ensemble.py        # Ensemble methods
│   ├── evaluation/            # Training & validation
│   │   ├── metrics.py         # Drift-specific metrics
│   │   ├── trainer.py         # Training loop
│   │   └── validator.py       # Comprehensive validation
│   └── utils/                 # Inference & visualization
│       ├── inference.py       # High-level prediction interface
│       ├── visualization.py   # Plotting & animation
│       └── export.py          # GeoJSON, NetCDF, GPX export
├── scripts/
│   ├── train.py               # Training entry point
│   └── predict.py             # Inference entry point
├── data/
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Processed features
├── models/                    # Saved model checkpoints
├── output/                    # Training logs, predictions, plots
└── notebooks/                 # Jupyter notebooks for exploration
```

## Configuration

Edit `config/default_config.yaml` to customize:

- **Data sources**: ERA5, Copernicus, NIC, BYU, IBCSO
- **Model architecture**: Hidden dim, layers, dropout, horizon
- **Physics parameters**: Wind factor, Coriolis alpha, iceberg geometry
- **Training**: LR, scheduler, early stopping, AMP
- **Evaluation**: Metrics, bootstrap samples, horizons

## Data Sources

| Data | Source | Access |
|------|--------|--------|
| Iceberg positions | US NIC / BYU | FTP / API |
| Wind (ERA5) | CDS / Copernicus | CDS API |
| Ocean currents | CMEMS (GLORYS/HYCOM) | CMEMS API |
| Bathymetry | IBCSO v2 | PANGAEA |
| Sea ice | NSIDC / OSI-SAF | FTP / API |

## Model Details

### Physics Baseline
```
u_drift = u_current + α·u_wind + sign(lat)·β·v_wind
v_drift = v_current + α·v_wind - sign(lat)·β·u_wind

α ≈ 0.02 (wind factor, 1-3%)
β ≈ 0.015 (Coriolis deflection)
```

### PINN Loss
```
L = w_data·MSE(pred, true) + w_physics·MSE(pred, physics) + w_boundary·constraints
```

### Evaluation Metrics
- Position RMSE/MAE (km)
- Direction error (degrees)
- Along-track / Cross-track error
- Drift distance & direction error
- Skill score vs persistence & physics baselines

## Uncertainty Quantification

- **Monte Carlo Dropout**: Enable dropout at inference for epistemic uncertainty
- **Ensemble**: Multiple model seeds for aleatoric + epistemic uncertainty
- **Quantile Regression**: Direct prediction of prediction intervals

## Export Formats

| Format | Use Case |
|--------|----------|
| GeoJSON | Web mapping, QGIS, Kepler.gl |
| NetCDF | Scientific analysis, CF-compliant |
| CSV | Simple tabular exchange |
| GPX | GPS devices, navigation apps |
| KML | Google Earth |

## Citation

```bibtex
@software{iceberg_drift_2024,
  title = {AI-Enabled Antarctic Iceberg Drift Prediction},
  author = {Antarctic Navigation DSS Team},
  year = {2024},
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Roadmap

- [ ] Real data integration (CDS API, CMEMS)
- [ ] Sub-seasonal forecasting (2-6 weeks)
- [ ] Coastal/fast-ice specialized model
- [ ] Multi-iceberg interaction modeling
- [ ] Operational API service
- [ ] Bridge-ready mobile app

---

**Built for Antarctic research vessels and national Antarctic programs.**