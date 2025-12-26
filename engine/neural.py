# -*- coding: utf-8 -*-
"""
Grandmaster Engine v11.5 - Neural Network Architectures

Deep learning components:
- SwingTransformer: Attention-based model for flow pattern recognition
- TCN: Temporal Convolutional Network for sequential patterns
- TemporalBlock: Building block for TCN with dilated convolutions
"""

import torch
import torch.nn as nn

# =============================================================================
# SWING TRANSFORMER (v2)
# =============================================================================
class SwingTransformer(nn.Module):
    """
    Transformer-based model for swing trading signal detection.

    Uses self-attention to capture relationships between different
    days in the input sequence, identifying accumulation/distribution patterns.

    Architecture:
    - Linear embedding to d_model dimensions
    - Learnable positional encoding
    - TransformerEncoder with multiple layers
    - FC layers with sigmoid output (0-1 probability)
    """

    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3,
                 output_size=1, dropout=0.1):
        super(SwingTransformer, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 10, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embed input features
        x = self.embedding(x)

        # Add positional encoding
        if x.size(1) <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        out = self.sigmoid(out)

        return out


# =============================================================================
# TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# =============================================================================
class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolution + residual connection.

    Key features:
    - Causal padding: prevents information leakage from future
    - Dilated convolutions: exponentially increasing receptive field
    - Residual connection: enables gradient flow in deep networks
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation  # Causal padding

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection (downsample if dimensions differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        """Initialize convolution weights with small values."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Causal convolution: trim future values
        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.relu(self.dropout(out))
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.relu(self.dropout(out))

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequential pattern detection.

    Replaces XGBoost for architectural diversity.
    Advantages:
    - Variable-length sequences
    - Dilated convolutions for long-range patterns
    - Detects multi-day accumulation (e.g., LULU 446-day base)
    """

    def __init__(self, input_size, num_channels=[32, 32, 16],
                 kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation for large receptive field
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        out = self.network(x)

        # Global average pooling over time dimension
        out = out.mean(dim=2)
        out = self.fc(out)

        return self.sigmoid(out)


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    'SwingTransformer',
    'TemporalBlock',
    'TCN',
]
