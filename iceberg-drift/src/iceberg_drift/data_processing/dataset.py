"""PyTorch Dataset and DataLoader for iceberg drift prediction."""

import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def iceberg_drift_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for IcebergDriftDataset.

    - Stacks 'x' and 'y' tensors along batch dimension.
    - Keeps 'meta' as a list of dicts (one per sample).
    - Assumes each item in batch is a dict with keys 'x', 'y', 'meta'.
    """
    # Stack tensors
    x = torch.stack([item['x'] for item in batch])
    y = torch.stack([item['y'] for item in batch])

    # Keep metadata as list of dicts
    meta = [item['meta'] for item in batch]

    return {
        'x': x,
        'y': y,
        'meta': meta,
    }


class IcebergDriftDataset(Dataset):
    """
    Dataset for iceberg drift prediction.

    Supports both:
    - One-step prediction (next timestep)
    - Sequence-to-sequence (history -> future trajectory)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        sequence_length: int = 24,  # hours of history
        prediction_horizon: int = 24,  # hours to predict
        time_step_hours: int = 6,  # data frequency
        mode: str = "sequence",  # "sequence" or "one_step"
        iceberg_id_col: str = "iceberg_id",
        datetime_col: str = "datetime",
    ):
        """
        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            target_cols: List of target column names
            sequence_length: Number of historical timesteps to use
            prediction_horizon: Number of future timesteps to predict
            time_step_hours: Hours per timestep
            mode: "sequence" for seq2seq, "one_step" for next-step prediction
            iceberg_id_col: Column name for iceberg identifier
            datetime_col: Column name for timestamp
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.time_step_hours = time_step_hours
        self.mode = mode
        self.iceberg_id_col = iceberg_id_col
        self.datetime_col = datetime_col

        # Validate columns exist
        missing_features = [c for c in feature_cols if c not in df.columns]
        missing_targets = [c for c in target_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        # Build indices for each trajectory
        self.trajectory_indices = self._build_trajectory_indices()
        logger.info(f"Dataset created: {len(self)} samples, mode={mode}")

    def _build_trajectory_indices(self) -> List[Dict[str, Any]]:
        """Build index mapping for valid sequence windows."""
        indices = []

        for iceberg_id in self.df[self.iceberg_id_col].unique():
            mask = self.df[self.iceberg_id_col] == iceberg_id
            traj_df = self.df.loc[mask].reset_index(drop=True)

            n_steps = len(traj_df)
            min_required = self.sequence_length + self.prediction_horizon

            if n_steps < min_required:
                continue

            if self.mode == "sequence":
                # Seq2seq: each window predicts prediction_horizon steps ahead
                for start_idx in range(0, n_steps - min_required + 1, self.prediction_horizon):
                    # Check for NaN values in feature slice
                    feature_slice = traj_df.iloc[start_idx:start_idx + self.sequence_length][self.feature_cols]
                    if feature_slice.isna().any().any():
                        continue  # Skip window with NaN features
                    indices.append({
                        "iceberg_id": iceberg_id,
                        "start_idx": start_idx,
                        "end_idx": start_idx + self.sequence_length,
                        "target_start_idx": start_idx + self.sequence_length,
                        "target_end_idx": start_idx + self.sequence_length + self.prediction_horizon,
                    })
            elif self.mode == "one_step":
                # One-step: each timestep predicts next timestep
                for start_idx in range(self.sequence_length, n_steps - self.prediction_horizon + 1):
                    indices.append({
                        "iceberg_id": iceberg_id,
                        "start_idx": start_idx - self.sequence_length,
                        "end_idx": start_idx,
                        "target_start_idx": start_idx,
                        "target_end_idx": start_idx + self.prediction_horizon,
                    })
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        return indices

    def __len__(self) -> int:
        return len(self.trajectory_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.trajectory_indices[idx]
        iceberg_id = info["iceberg_id"]

        # Get trajectory data
        mask = self.df[self.iceberg_id_col] == iceberg_id
        traj_df = self.df.loc[mask].reset_index(drop=True)

        # Input sequence: features
        x = traj_df.iloc[info["start_idx"]:info["end_idx"]][self.feature_cols].values
        x = torch.tensor(x, dtype=torch.float32)  # (seq_len, n_features)

        # Target sequence
        y = traj_df.iloc[info["target_start_idx"]:info["target_end_idx"]][self.target_cols].values
        y = torch.tensor(y, dtype=torch.float32)  # (pred_horizon, n_targets)

        # Metadata
        last_row = traj_df.iloc[info["end_idx"] - 1]
        meta = {
            "iceberg_id": iceberg_id,
            "init_lat": last_row["lat"],
            "init_lon": last_row["lon"],
            "init_time": last_row[self.datetime_col],
        }
        # Carry raw environmental forcing at the last input timestep, so downstream
        # consumers (e.g. DriftValidator._compare_with_physics) have real values to
        # build a physics baseline from instead of silently defaulting to zero.
        for env_col in ("current_uo", "current_vo", "wind_u10", "wind_v10"):
            raw_col = f"raw_{env_col}"
            if raw_col in traj_df.columns:
                meta[raw_col] = last_row[raw_col]

        return {
            "x": x,
            "y": y,
            "meta": meta,
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    sequence_length: int = 24,
    prediction_horizon: int = 24,
    time_step_hours: int = 6,
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "sequence",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""

    train_dataset = IcebergDriftDataset(
        train_df, feature_cols, target_cols,
        sequence_length, prediction_horizon, time_step_hours, mode
    )
    val_dataset = IcebergDriftDataset(
        val_df, feature_cols, target_cols,
        sequence_length, prediction_horizon, time_step_hours, mode
    )
    test_dataset = IcebergDriftDataset(
        test_df, feature_cols, target_cols,
        sequence_length, prediction_horizon, time_step_hours, mode
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=iceberg_drift_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=iceberg_drift_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=iceberg_drift_collate_fn
    )

    logger.info(f"Dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader