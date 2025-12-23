"""TabNet: Attentive Interpretable Tabular Learning

Phase 3: ML Ensemble Diversity
Replaces LightGBM with TabNet for feature interaction learning.

Key Features:
- Self-attention mechanism learns feature importance dynamically
- Captures non-linear interactions (high RSI + $1B flow = bullish)
- Sparse attention provides explainability
- Handles LULU-style edge cases trees miss

Reference: https://arxiv.org/abs/1908.07442
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class GhostBatchNorm(nn.Module):
    """Ghost Batch Normalization for TabNet.
    
    Splits batch into smaller "ghost" batches to reduce overfitting.
    """
    
    def __init__(self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        super(GhostBatchNorm, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) > self.virtual_batch_size:
            # Split into ghost batches
            chunks = x.chunk(int(np.ceil(x.size(0) / self.virtual_batch_size)), dim=0)
            return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)
        else:
            return self.bn(x)


class AttentiveTransformer(nn.Module):
    """Attention mechanism for feature selection.
    
    Learns which features are important for each prediction.
    """
    
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = 128):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size=virtual_batch_size)
    
    def forward(self, priors: torch.Tensor, processed_feat: torch.Tensor) -> torch.Tensor:
        """Compute attention mask.
        
        Args:
            priors: Prior attention from previous step [batch, features]
            processed_feat: Processed features [batch, features]
            
        Returns:
            Attention mask [batch, features] (sparse, sums to 1)
        """
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = x * priors  # Multiply by priors (sparsity enforcement)
        return F.sparsemax(x, dim=-1)  # Sparse softmax (many zeros)


class FeatureTransformer(nn.Module):
    """Feature transformation block with residual connection."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 shared_layers: nn.ModuleList, n_independent: int = 2,
                 virtual_batch_size: int = 128):
        super(FeatureTransformer, self).__init__()
        
        # Shared layers across all steps
        self.shared = nn.ModuleList(shared_layers) if shared_layers else None
        
        # Independent layers for this step
        self.independent = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else output_dim, output_dim),
                GhostBatchNorm(output_dim, virtual_batch_size=virtual_batch_size)
            )
            for i in range(n_independent)
        ])
        
        self.scale = torch.sqrt(torch.tensor(0.5))  # Residual scaling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with GLU activation and residual."""
        # Shared transformation
        if self.shared:
            for layer in self.shared:
                x = F.glu(layer(x), dim=-1)
        
        # Independent transformation
        for layer in self.independent:
            x = F.glu(layer(x) + x, dim=-1) * self.scale
        
        return x


class TabNetModel(nn.Module):
    """TabNet architecture for swing trading prediction.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes (1 for binary)
        n_d: Dimension of decision layers (default: 64)
        n_a: Dimension of attention layers (default: 64)
        n_steps: Number of sequential decision steps (default: 5)
        gamma: Relaxation parameter for attention sparsity (default: 1.3)
        n_independent: Number of independent GLU layers per step (default: 2)
        n_shared: Number of shared GLU layers across steps (default: 2)
        virtual_batch_size: Virtual batch size for ghost batch norm (default: 128)
        momentum: Momentum for batch normalization (default: 0.01)
    
    Example:
        >>> model = TabNetModel(input_dim=20, output_dim=1)
        >>> x = torch.randn(32, 20)  # [batch=32, features=20]
        >>> output, attention_masks = model(x)
        >>> output.shape  # torch.Size([32, 1])
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1,
                 n_d: int = 64, n_a: int = 64, n_steps: int = 5,
                 gamma: float = 1.3, n_independent: int = 2, n_shared: int = 2,
                 virtual_batch_size: int = 128, momentum: float = 0.01):
        super(TabNetModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.virtual_batch_size = virtual_batch_size
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim, momentum=momentum)
        
        # Shared feature transformer layers (used across all steps)
        if n_shared > 0:
            self.shared_feat_transform = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else n_d + n_a, 2 * (n_d + n_a))
                for i in range(n_shared)
            ])
        else:
            self.shared_feat_transform = None
        
        # Step-specific components
        self.initial_feat_transform = FeatureTransformer(
            input_dim, n_d + n_a, self.shared_feat_transform,
            n_independent, virtual_batch_size
        )
        
        self.feat_transformers = nn.ModuleList([
            FeatureTransformer(
                input_dim, n_d + n_a, self.shared_feat_transform,
                n_independent, virtual_batch_size
            )
            for _ in range(n_steps)
        ])
        
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(n_a, input_dim, virtual_batch_size)
            for _ in range(n_steps)
        ])
        
        # Final output layer
        self.final_projection = nn.Linear(n_d, output_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through TabNet.
        
        Args:
            x: Input features [batch, input_dim]
            
        Returns:
            output: Predictions [batch, output_dim]
            attention_masks: List of attention masks for interpretability
        """
        # Input normalization
        x = self.input_bn(x)
        
        # Initial prior (uniform attention)
        batch_size = x.size(0)
        prior = torch.ones(batch_size, self.input_dim).to(x.device)
        
        # Aggregate decision output
        aggregated_output = torch.zeros(batch_size, self.n_d).to(x.device)
        
        # Store attention masks for interpretability
        attention_masks = []
        
        # Initial feature transformation
        processed_feat = self.initial_feat_transform(x)
        
        # Sequential decision steps
        for step_i in range(self.n_steps):
            # Split into attention and decision components
            a, d = processed_feat[:, :self.n_a], processed_feat[:, self.n_a:]
            
            # Compute attention mask
            mask = self.attentive_transformers[step_i](prior, a)
            attention_masks.append(mask)
            
            # Update prior (sparsity enforcement)
            prior = prior * (self.gamma - mask)
            
            # Masked input
            masked_x = mask * x
            
            # Feature transformation for next step
            processed_feat = self.feat_transformers[step_i](masked_x)
            
            # Aggregate decision
            aggregated_output += d
        
        # Final prediction
        output = self.final_projection(aggregated_output)
        output = self.sigmoid(output)
        
        return output, attention_masks
    
    def predict_proba(self, X: np.ndarray, device: torch.device = None) -> np.ndarray:
        """Sklearn-compatible prediction interface.
        
        Args:
            X: Input features [n_samples, n_features]
            device: Torch device
            
        Returns:
            Predictions [n_samples, 2] with [prob_class_0, prob_class_1]
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            preds, _ = self.forward(X_tensor)
            preds = preds.cpu().numpy().flatten()
            
            prob_1 = preds
            prob_0 = 1 - preds
            return np.column_stack([prob_0, prob_1])
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, batch_size: int = 256,
            learning_rate: float = 0.02,
            device: torch.device = None,
            verbose: bool = True) -> 'TabNetModel':
        """Train TabNet model.
        
        Args:
            X: Training features [n_samples, n_features]
            y: Binary labels [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Adam learning rate
            device: Torch device
            verbose: Print progress
            
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
        
        # Data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs, _ = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"  [TabNet] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return self


# Sparsemax implementation (required for TabNet attention)
def sparsemax_function(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation (sparse softmax with many exact zeros)."""
    input_shape = input.shape
    input = input.view(-1, input_shape[dim])
    
    # Translate input
    input = input - torch.max(input, dim=1, keepdim=True)[0]
    
    # Sort input in descending order
    zs = torch.sort(input, dim=1, descending=True)[0]
    k_range = torch.arange(1, input.size(1) + 1, dtype=input.dtype, device=input.device).view(1, -1)
    
    # Determine sparsity threshold
    bound = 1 + k_range * zs
    cumsum = torch.cumsum(zs, dim=1)
    is_gt = (bound > cumsum).type(input.dtype)
    k = torch.sum(is_gt, dim=1, keepdim=True)
    
    # Compute threshold
    zs_sparse = is_gt * zs
    taus = (torch.sum(zs_sparse, dim=1, keepdim=True) - 1) / k
    
    # Apply sparsemax transformation
    output = torch.max(torch.zeros_like(input), input - taus)
    
    return output.view(input_shape)


# Register sparsemax as functional
F.sparsemax = sparsemax_function
