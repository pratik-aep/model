"""ML models for iceberg drift prediction.

Includes LSTM, GRU, Transformer, and MLP architectures for sequence-to-sequence prediction.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Base Model Class
# =============================================================================

class BaseDriftModel(nn.Module):
    """Base class for drift prediction models."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 24,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_horizon = prediction_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, prediction_horizon, output_dim)
        """
        raise NotImplementedError

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# LSTM Model
# =============================================================================

class LSTMDriftModel(BaseDriftModel):
    """LSTM-based sequence-to-sequence model for drift prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 24,
        bidirectional: bool = False,
        use_attention: bool = True,
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout, prediction_horizon)

        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=encoder_output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(encoder_output_dim)

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=output_dim,  # Teacher forcing: use previous target as input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Start token for decoder (learned)
        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            targets: (batch_size, pred_horizon, output_dim) for teacher forcing
            teacher_forcing_ratio: Probability of using ground truth as next input

        Returns:
            predictions: (batch_size, pred_horizon, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Encode
        encoder_output, (hidden, cell) = self.encoder(x)
        # encoder_output: (batch, seq_len, hidden_dim * num_directions)
        # hidden, cell: (num_layers * num_directions, batch, hidden_dim)

        # Apply attention if enabled
        if self.use_attention:
            # Self-attention on encoder output
            attn_output, _ = self.attention(encoder_output, encoder_output, encoder_output)
            encoder_output = self.attention_norm(encoder_output + attn_output)

        # Use last encoder hidden state as decoder initial state
        # Handle bidirectional: concatenate forward/backward hidden states
        if self.bidirectional:
            # Reshape: (num_layers, 2, batch, hidden) -> (num_layers, batch, 2*hidden)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
            cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)

        # Decoder initial hidden state (use last layer)
        dec_hidden = hidden[-self.num_layers:]
        dec_cell = cell[-self.num_layers:]

        # Decode autoregressively
        predictions = []
        decoder_input = self.start_token.expand(batch_size, 1, self.output_dim)

        for t in range(self.prediction_horizon):
            # Decoder step
            dec_output, (dec_hidden, dec_cell) = self.decoder(
                decoder_input, (dec_hidden, dec_cell)
            )
            # dec_output: (batch, 1, hidden_dim)

            # Project to output
            pred = self.output_proj(dec_output)  # (batch, 1, output_dim)
            predictions.append(pred)

            # Next input: teacher forcing or own prediction
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t:t+1, :]
            else:
                decoder_input = pred.detach()

        predictions = torch.cat(predictions, dim=1)  # (batch, pred_horizon, output_dim)
        return predictions


# =============================================================================
# GRU Model
# =============================================================================

class GRUDriftModel(BaseDriftModel):
    """GRU-based sequence-to-sequence model (lighter than LSTM)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 24,
        bidirectional: bool = False,
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout, prediction_horizon)

        self.bidirectional = bidirectional

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Encode
        encoder_output, hidden = self.encoder(x)
        # hidden: (num_layers * num_directions, batch, hidden_dim)

        # Handle bidirectional
        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)

        dec_hidden = hidden[-self.num_layers:]

        # Decode
        predictions = []
        decoder_input = self.start_token.expand(batch_size, 1, self.output_dim)

        for t in range(self.prediction_horizon):
            dec_output, dec_hidden = self.decoder(decoder_input, dec_hidden)
            pred = self.output_proj(dec_output)
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t:t+1, :]
            else:
                decoder_input = pred.detach()

        return torch.cat(predictions, dim=1)


# =============================================================================
# Transformer Model
# =============================================================================

class TransformerDriftModel(BaseDriftModel):
    """Transformer-based sequence-to-sequence model for drift prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        prediction_horizon: int = 24,
        num_heads: int = 8,
        ff_dim: int = 512,
        max_seq_len: int = 100,
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout, prediction_horizon)

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_seq_len)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Target embedding
        self.target_proj = nn.Linear(output_dim, hidden_dim)
        self.target_pos_encoder = PositionalEncoding(hidden_dim, dropout, prediction_horizon)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Encode
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        memory = self.encoder(x)  # (batch, seq_len, hidden_dim)

        # Prepare target sequence for decoder
        if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: use full target sequence
            tgt = self.target_proj(targets)
            tgt = self.target_pos_encoder(tgt)

            # Causal mask for autoregressive decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                self.prediction_horizon, device=device
            )

            dec_output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            predictions = self.output_proj(dec_output)
        else:
            # Autoregressive decoding
            predictions = []
            decoder_input = self.start_token.expand(batch_size, 1, self.hidden_dim)

            for t in range(self.prediction_horizon):
                tgt = self.target_pos_encoder(decoder_input)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    t + 1, device=device
                )
                dec_output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
                pred = self.output_proj(dec_output[:, -1:, :])
                predictions.append(pred)
                decoder_input = torch.cat([decoder_input, self.target_proj(pred)], dim=1)

            predictions = torch.cat(predictions, dim=1)

        return predictions


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# MLP Model (for one-step prediction)
# =============================================================================

class MLPDriftModel(BaseDriftModel):
    """MLP for one-step or direct multi-step prediction (no recurrence)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        prediction_horizon: int = 24,
        use_residual: bool = True,
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout, prediction_horizon)
        self.use_residual = use_residual
        # Projection from concatenated last timestep and mean-pooled features (2*input_dim) to input_dim
        self.input_adjust = nn.Linear(2 * input_dim, input_dim)

        # Input projection
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_residual and i > 0:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)

        # Output heads for each prediction step
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(prediction_horizon)
        ])

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            targets: unused (MLP predicts all horizon steps directly, no
                autoregressive decoding) — accepted only so the call signature
                matches the other drift models (LSTM/GRU/Transformer).
            teacher_forcing_ratio: unused, see above.
        """
        # Use last timestep + mean pooling over sequence
        x_last = x[:, -1, :]  # (batch, input_dim)
        x_mean = x.mean(dim=1)  # (batch, input_dim)
        x_combined = torch.cat([x_last, x_mean], dim=1)  # (batch, 2*input_dim)

        # Project to input_dim
        x_combined = self.input_adjust(x_combined)

        # Forward through backbone
        features = self.backbone(x_combined)  # (batch, hidden_dim)

        # Predict for each horizon step
        predictions = []
        for head in self.output_heads:
            predictions.append(head(features).unsqueeze(1))

        return torch.cat(predictions, dim=1)  # (batch, pred_horizon, output_dim)


class ResidualBlock(nn.Module):
    """Residual block with layer norm."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> BaseDriftModel:
    """Factory function to create models."""
    model_type = model_type.lower()

    if model_type == "lstm":
        return LSTMDriftModel(input_dim, output_dim, **kwargs)
    elif model_type == "gru":
        return GRUDriftModel(input_dim, output_dim, **kwargs)
    elif model_type == "transformer":
        return TransformerDriftModel(input_dim, output_dim, **kwargs)
    elif model_type == "mlp":
        return MLPDriftModel(input_dim, output_dim, **kwargs)
    elif model_type == "pinn":
        from .pinn import PINNDriftModel
        return PINNDriftModel(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Import numpy for PositionalEncoding
import numpy as np