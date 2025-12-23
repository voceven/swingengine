"""Temporal Convolutional Network for Variable-Length Sequences

Phase 3: ML Ensemble Diversity
Replaces XGBoost with TCN for temporal pattern detection.

Key Features:
- Handles variable-length sequences (NVDA: 730d vs Recent IPO: 180d)
- Parallel 1D convolutions (faster than LSTM)
- Captures multi-day accumulation patterns (446-day LULU base)
- Dilated causal convolutions preserve temporal ordering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolutions.
    
    Args:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation factor for temporal receptive field
        padding: Padding size
        dropout: Dropout probability
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated causal convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated causal convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor [batch, channels, sequence_length]
            
        Returns:
            Output tensor [batch, channels, sequence_length]
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TemporalCNN(nn.Module):
    """Temporal Convolutional Network for swing trading prediction.
    
    Architecture designed to capture institutional accumulation patterns
    across variable-length price histories (60-730 days).
    
    Args:
        num_inputs: Number of input features (e.g., gamma_velocity, oi_accel, price_momentum)
        num_channels: List of channel sizes for each TCN layer
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout probability (default: 0.2)
    
    Example:
        >>> tcn = TemporalCNN(num_inputs=4, num_channels=[32, 64, 128], kernel_size=3)
        >>> x = torch.randn(16, 4, 100)  # [batch=16, features=4, seq_len=100]
        >>> output = tcn(x)  # [batch=16, 1] - binary prediction
    """
    
    def __init__(self, num_inputs: int, num_channels: list = [32, 64, 128], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalCNN, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, 
                            stride=1, dilation=dilation_size, 
                            padding=padding, dropout=dropout)
            )
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling + classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN.
        
        Args:
            x: Input tensor [batch, features, sequence_length]
               Features: [gamma_velocity, oi_accel, price_momentum, volume_trend]
               Sequence: Variable length (60-730 days)
        
        Returns:
            Prediction tensor [batch, 1] with probability (0.0-1.0)
        """
        # TCN layers (preserve sequence length with causal padding)
        out = self.network(x)  # [batch, channels[-1], seq_len]
        
        # Global average pooling (handles variable sequence lengths)
        out = self.global_pool(out)  # [batch, channels[-1], 1]
        out = out.squeeze(-1)  # [batch, channels[-1]]
        
        # Classification head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out
    
    def predict_proba(self, X: np.ndarray, device: torch.device = None) -> np.ndarray:
        """Sklearn-compatible prediction interface.
        
        Args:
            X: Input array [batch, features, sequence_length]
            device: Torch device (CPU/GPU)
            
        Returns:
            Predictions [batch, 2] with [prob_class_0, prob_class_1]
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.forward(X_tensor).cpu().numpy().flatten()
            
            # Convert to sklearn format [prob_0, prob_1]
            prob_1 = preds
            prob_0 = 1 - preds
            return np.column_stack([prob_0, prob_1])
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 50, batch_size: int = 32, 
            learning_rate: float = 0.001,
            device: torch.device = None,
            verbose: bool = True) -> 'TemporalCNN':
        """Train TCN model.
        
        Args:
            X: Training sequences [n_samples, features, seq_len]
            y: Binary labels [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Adam optimizer learning rate
            device: Torch device (CPU/GPU)
            verbose: Print training progress
            
        Returns:
            self (trained model)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device)
        self.train()
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"  [TCN] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return self
