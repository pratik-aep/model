"""Training pipeline for iceberg drift prediction models."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from ..models.ml_models import BaseDriftModel
from ..models.pinn import PINNDriftModel, HybridDriftModel
from .metrics import compute_all_metrics, bootstrap_confidence_interval

logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Training loop
    epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_loss"
    gradient_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 10
    save_interval: int = 10
    output_dir: str = "output/checkpoints"

    # Teacher forcing
    initial_teacher_forcing: float = 0.5
    final_teacher_forcing: float = 0.0
    teacher_forcing_decay: str = "linear"  # "linear", "exponential", "none"

    # Validation
    validate_every: int = 1
    metric_horizons: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 72])


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    epoch: int = 0
    step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


# =============================================================================
# Trainer Class
# =============================================================================

class DriftTrainer:
    """
    Trainer for iceberg drift prediction models.

    Supports:
    - Standard ML models (LSTM, GRU, Transformer, MLP)
    - Physics-informed models (PINN, Hybrid)
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "auto",
        loss_fn: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Loss function
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.state = TrainingState()

        # Teacher forcing schedule
        self.teacher_forcing_ratio = config.initial_teacher_forcing

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        if self.config.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.config.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.config.optimizer == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        scheduler_type = self.config.scheduler
        params = self.config.scheduler_params

        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=float(params.get("eta_min", 1e-6)),
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 30),
                gamma=params.get("gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
                min_lr=params.get("min_lr", 1e-6),
            )
        elif scheduler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def _update_teacher_forcing(self, epoch: int):
        """Update teacher forcing ratio based on schedule."""
        if self.config.teacher_forcing_decay == "linear":
            progress = epoch / self.config.epochs
            self.teacher_forcing_ratio = (
                self.config.initial_teacher_forcing * (1 - progress) +
                self.config.final_teacher_forcing * progress
            )
        elif self.config.teacher_forcing_decay == "exponential":
            decay_rate = (self.config.final_teacher_forcing / self.config.initial_teacher_forcing) ** (1 / self.config.epochs)
            self.teacher_forcing_ratio = self.config.initial_teacher_forcing * (decay_rate ** epoch)
        else:
            self.teacher_forcing_ratio = self.config.initial_teacher_forcing

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            metadata = batch["meta"]

            # Debug first batch
            if batch_idx == 0:
                logger.info(f"Debug batch {batch_idx}:")
                logger.info(f"  Input shape: {x.shape}, dtype: {x.dtype}")
                logger.info(f"  Input NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
                logger.info(f"  Input min/max: {x.min().item():.6f}/{x.max().item():.6f}")
                logger.info(f"  Target shape: {y.shape}")
                logger.info(f"  Target NaN: {torch.isnan(y).any()}, Inf: {torch.isinf(y).any()}")
                logger.info(f"  Target min/max: {y.min().item():.6f}/{y.max().item():.6f}")

            # Forward pass with mixed precision
            amp_context = autocast() if self.scaler else nullcontext()

            with amp_context:
                # Model-specific forward
                if isinstance(self.model, (PINNDriftModel, HybridDriftModel)):
                    output = self.model(
                        x,
                        targets=y,
                        teacher_forcing_ratio=self.teacher_forcing_ratio,
                        metadata=metadata,
                    )
                    predictions = output["predictions"]

                    # Debug predictions
                    if batch_idx == 0:
                        logger.info(f"  Predictions shape: {predictions.shape}")
                        logger.info(f"  Predictions NaN: {torch.isnan(predictions).any()}, Inf: {torch.isinf(predictions).any()}")
                        logger.info(f"  Predictions min/max: {predictions.min().item():.6f}/{predictions.max().item():.6f}")

                    # Compute loss
                    if hasattr(self.model, 'compute_loss'):
                        loss_dict = self.model.compute_loss(output, y, x, metadata)
                        loss = loss_dict["total_loss"]
                    else:
                        loss = self.loss_fn(predictions, y)

                    # Debug loss
                    if batch_idx == 0:
                        logger.info(f"  Loss: {loss.item()}")
                        if torch.isnan(loss):
                            logger.info("  *** LOSS IS NaN ***")
                else:
                    # Standard model
                    predictions = self.model(x, y, self.teacher_forcing_ratio)
                    loss = self.loss_fn(predictions, y)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []
        all_metadata = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                metadata = batch["meta"]

                if isinstance(self.model, (PINNDriftModel, HybridDriftModel)):
                    output = self.model(x, targets=None, teacher_forcing_ratio=0.0, metadata=metadata)
                    predictions = output["predictions"]

                    if hasattr(self.model, 'compute_loss'):
                        loss_dict = self.model.compute_loss(output, y, x, metadata)
                        loss = loss_dict["total_loss"]
                    else:
                        loss = self.loss_fn(predictions, y)
                else:
                    predictions = self.model(x, targets=None, teacher_forcing_ratio=0.0)
                    loss = self.loss_fn(predictions, y)

                total_loss += loss.item()
                num_batches += 1

                # Store for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_metadata.extend(metadata)

        avg_loss = total_loss / max(num_batches, 1)

        # Compute comprehensive metrics
        if all_predictions:
            predictions_np = np.concatenate(all_predictions, axis=0)
            targets_np = np.concatenate(all_targets, axis=0)

            # Convert to position format for metrics
            pred_dict = {"u": predictions_np[:, :, 0], "v": predictions_np[:, :, 1]}
            true_dict = {"u": targets_np[:, :, 0], "v": targets_np[:, :, 1]}

            metrics = compute_all_metrics(
                pred_dict, true_dict, all_metadata,
                horizon_hours=self.config.metric_horizons,
            )

            metrics_dict = {
                "val_loss": avg_loss,
                "position_rmse_km": metrics.rmse_position_km,
                "position_mae_km": metrics.mean_position_error_km,
                "direction_error_deg": metrics.mean_direction_error_deg,
                "skill_vs_persistence": metrics.skill_score_vs_persistence,
                "skill_vs_physics": metrics.skill_score_vs_physics,
            }

            # Add per-horizon metrics
            for horizon, h_metrics in metrics.horizon_metrics.items():
                for k, v in h_metrics.items():
                    metrics_dict[f"horizon_{horizon}h_{k}"] = v
        else:
            metrics_dict = {"val_loss": avg_loss}

        return metrics_dict

    def train(self) -> TrainingState:
        """Full training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        last_val_loss = float('inf')  # <-- add this

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            self._update_teacher_forcing(epoch)

            # Train
            train_metrics = self.train_epoch()
            self.state.train_losses.append(train_metrics["train_loss"])

            # Validate
            if (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validate()
                last_val_loss = val_metrics["val_loss"]  # <-- add this
                self.state.val_losses.append(val_metrics["val_loss"])
                self.state.val_metrics.append(val_metrics)

                # Check for improvement
                metric_val = val_metrics.get(self.config.early_stopping_metric, val_metrics["val_loss"])
                if metric_val < self.state.best_metric:
                    self.state.best_metric = metric_val
                    self.state.best_epoch = epoch
                    self.save_checkpoint("best_model.pt")

                pos_rmse = val_metrics.get("position_rmse_km")
                skill_persist = val_metrics.get("skill_vs_persistence")
                pos_rmse_str = f"{pos_rmse:.2f}" if pos_rmse is not None else "N/A"
                skill_persist_str = f"{skill_persist:.3f}" if skill_persist is not None else "N/A"

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Pos RMSE: {pos_rmse_str} km | "
                    f"Skill (persist): {skill_persist_str} | "
                    f"TF: {self.teacher_forcing_ratio:.3f}"
                )

                # Early stopping
                if epoch - self.state.best_epoch >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Learning rate step
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(last_val_loss)  # <-- use last_val_loss, not val_metrics[...]
                    else:
                        self.scheduler.step()

            self.state.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best epoch: {self.state.best_epoch + 1}, Best metric: {self.state.best_metric:.4f}")

        return self.state

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "epoch": self.state.epoch,
            "step": self.state.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "state": self.state,
            "config": self.config,
        }, path)
        logger.debug(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str) -> TrainingState:
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.state = checkpoint["state"]
        logger.info(f"Checkpoint loaded from {path}")
        return self.state


# =============================================================================
# Convenience Function
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    device: str = "auto",
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[nn.Module, TrainingState]:
    """
    Convenience function to train a model.

    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: TrainingConfig (uses defaults if None)
        device: Device to train on
        loss_fn: Loss function (uses MSE if None)

    Returns:
        (trained_model, training_state)
    """
    if config is None:
        config = TrainingConfig()

    trainer = DriftTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_fn=loss_fn,
    )

    state = trainer.train()
    return model, state