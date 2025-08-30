"""
Simplified Mamba-style state space model for modular arithmetic learning.

This module implements a simplified version of the Mamba architecture,
focusing on the core state space modeling concepts for comparison with transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class SimpleStateSpaceLayer(nn.Module):
    """
    Simplified state space layer inspired by Mamba.
    
    This implements a basic state space model with:
    - Linear state transition
    - Input-dependent gating
    - Selective filtering
    """
    
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))  # Log of diagonal A matrix
        self.B = nn.Linear(d_model, d_state, bias=False)  # Input matrix
        self.C = nn.Linear(d_model, d_state, bias=False)  # Output matrix
        self.D = nn.Parameter(torch.ones(d_model))  # Skip connection
        
        # Gating mechanism
        self.gate = nn.Linear(d_model, d_model)
        
        # Normalization and regularization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize state space parameters."""
        # Initialize A to be stable (negative real parts)
        nn.init.uniform_(self.A_log, -3, -1)
        
        # Initialize other parameters
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.C.weight)
        nn.init.ones_(self.D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through state space layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Convert A from log space for stability
        A = torch.exp(self.A_log)  # Shape: (d_model, d_state)
        
        # Get input-dependent B and C matrices
        B = self.B(x)  # Shape: (batch_size, seq_len, d_state)
        C = self.C(x)  # Shape: (batch_size, seq_len, d_state)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, d_model, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # Process sequence step by step (simplified discrete SSM)
        for t in range(seq_len):
            # Get current inputs
            x_t = x[:, t, :]  # (batch_size, d_model)
            B_t = B[:, t, :]  # (batch_size, d_state)  
            C_t = C[:, t, :]  # (batch_size, d_state)
            
            # State transition: h_t = A * h_{t-1} + B_t * x_t
            # Broadcasting: h is (batch, d_model, d_state), A is (d_model, d_state)
            h = h * A.unsqueeze(0) + B_t.unsqueeze(1) * x_t.unsqueeze(2)
            
            # Output: y_t = C_t * h_t + D * x_t
            # Sum over state dimension: (batch, d_model, d_state) * (batch, 1, d_state) -> (batch, d_model)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=2) + self.D * x_t
            
            outputs.append(y_t)
        
        # Stack outputs: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        output = torch.stack(outputs, dim=1)
        
        # Apply gating and residual connection
        gate_values = torch.sigmoid(self.gate(x))
        output = gate_values * output + (1 - gate_values) * x
        
        # Normalize and apply dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class ModularMamba(nn.Module):
    """
    Simplified Mamba-style model for modular arithmetic classification.
    
    Architecture:
    - 2-layer state space model
    - 64-dimensional embeddings and states
    - Classification head for mod p output
    - State extraction hooks for analysis
    """
    
    def __init__(self,
                 vocab_size: int = 17,
                 d_model: int = 64,
                 d_state: int = 16,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # State space layers
        self.layers = nn.ModuleList([
            SimpleStateSpaceLayer(d_model, d_state, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(d_model, vocab_size)
        
        # For activation extraction
        self.activation_hooks = {}
        self.activations = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 2) containing (a, b) pairs
            return_hidden: Whether to return hidden states
            
        Returns:
            Logits of shape (batch_size, vocab_size) for classification
        """
        batch_size = x.size(0)
        
        # Embed inputs: (batch_size, 2) -> (batch_size, 2, d_model)
        embedded = self.embedding(x)
        
        # Store embedding activations
        if 'embeddings' in self.activation_hooks:
            self.activations['embeddings'] = embedded.clone()
        
        # Apply state space layers
        hidden_states = embedded
        layer_outputs = []
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            layer_outputs.append(hidden_states)
            
            # Store layer activations if requested
            if f'layer_{i}' in self.activation_hooks:
                self.activations[f'layer_{i}'] = hidden_states.clone()
        
        # Store final hidden states
        if 'hidden_states' in self.activation_hooks:
            self.activations['hidden_states'] = hidden_states.clone()
        
        # Aggregate sequence: take mean over sequence dimension  
        # (batch_size, 2, d_model) -> (batch_size, d_model)
        aggregated = hidden_states.mean(dim=1)
        
        # Store aggregated activations
        if 'aggregated' in self.activation_hooks:
            self.activations['aggregated'] = aggregated.clone()
        
        # Classification: (batch_size, d_model) -> (batch_size, vocab_size)
        logits = self.classifier(aggregated)
        
        if return_hidden:
            return logits, {
                'embeddings': embedded,
                'layer_outputs': layer_outputs,
                'hidden_states': hidden_states,
                'aggregated': aggregated
            }
        
        return logits
    
    def register_activation_hook(self, layer_name: str):
        """Register hook to capture activations from a specific layer."""
        valid_layers = ['embeddings', 'hidden_states', 'aggregated'] + [f'layer_{i}' for i in range(self.num_layers)]
        if layer_name not in valid_layers:
            raise ValueError(f"Unknown layer: {layer_name}. Valid options: {valid_layers}")
        self.activation_hooks[layer_name] = True
    
    def remove_activation_hook(self, layer_name: str):
        """Remove activation hook."""
        if layer_name in self.activation_hooks:
            del self.activation_hooks[layer_name]
        if layer_name in self.activations:
            del self.activations[layer_name]
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations.copy()
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding representations for analysis.
        
        Args:
            x: Input tensor of shape (batch_size, 2)
            
        Returns:
            Embeddings of shape (batch_size, 2, d_model)
        """
        with torch.no_grad():
            embedded = self.embedding(x)
            return embedded
    
    def get_number_embeddings(self) -> torch.Tensor:
        """
        Get embeddings for all numbers 0 through vocab_size-1.
        
        Returns:
            Tensor of shape (vocab_size, d_model) with embeddings for each number
        """
        with torch.no_grad():
            numbers = torch.arange(self.vocab_size)
            return self.embedding(numbers)
    
    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for a batch of inputs.
        
        Args:
            x: Input tensor of shape (batch_size, 2)
            
        Returns:
            Predictions of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def compute_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute accuracy on a batch of data.
        
        Args:
            x: Input tensor of shape (batch_size, 2)
            y: Target tensor of shape (batch_size,)
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        with torch.no_grad():
            predictions = self.predict_batch(x)
            correct = (predictions == y).float()
            return correct.mean().item()
    
    def get_state_evolution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get evolution of internal states for analysis.
        
        Args:
            x: Input tensor of shape (batch_size, 2)
            
        Returns:
            Dictionary containing state evolution across layers
        """
        # Register hooks for all layers
        for i in range(self.num_layers):
            self.register_activation_hook(f'layer_{i}')
        
        with torch.no_grad():
            _ = self.forward(x)
            states = self.get_activations()
        
        # Clean up hooks
        for i in range(self.num_layers):
            self.remove_activation_hook(f'layer_{i}')
        
        return states


def create_mamba_model(vocab_size: int = 17, device: str = 'cpu') -> ModularMamba:
    """
    Create and initialize a ModularMamba model.
    
    Args:
        vocab_size: Size of vocabulary (typically p for mod p arithmetic)
        device: Device to place model on
        
    Returns:
        Initialized ModularMamba model
    """
    model = ModularMamba(vocab_size=vocab_size)
    model = model.to(device)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test the model with sample data
    model = create_mamba_model(vocab_size=17)
    
    # Create sample batch
    batch_size = 8
    x = torch.randint(0, 17, (batch_size, 2))  # Random (a, b) pairs
    y = (x[:, 0] + x[:, 1]) % 17  # Ground truth: (a + b) mod 17
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    
    # Test activation extraction
    model.register_activation_hook('embeddings')
    model.register_activation_hook('aggregated')
    
    logits = model(x)
    activations = model.get_activations()
    
    print(f"Embeddings shape: {activations['embeddings'].shape}")
    print(f"Aggregated shape: {activations['aggregated'].shape}")
    
    # Test state evolution
    state_evolution = model.get_state_evolution(x)
    print(f"State evolution keys: {list(state_evolution.keys())}")
    for key, tensor in state_evolution.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test number embeddings
    number_embeddings = model.get_number_embeddings()
    print(f"Number embeddings shape: {number_embeddings.shape}")
    
    # Test prediction and accuracy
    predictions = model.predict_batch(x)
    accuracy = model.compute_accuracy(x, y)
    
    print(f"Predictions: {predictions.tolist()}")
    print(f"Ground truth: {y.tolist()}")
    print(f"Random accuracy: {accuracy:.3f}")
    
    print("✅ Mamba model implementation complete!")