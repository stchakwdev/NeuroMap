"""
Minimal transformer model for modular arithmetic learning.

This module implements a 2-layer transformer with 4 attention heads,
specifically designed for learning f(a,b) = (a+b) mod p tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ModularTransformer(nn.Module):
    """
    Transformer model for modular arithmetic classification.
    
    Architecture:
    - 2-layer transformer encoder
    - 4 attention heads  
    - 64-dimensional embeddings
    - Classification head for mod p output
    - Activation hooks for analysis
    """
    
    def __init__(self, 
                 vocab_size: int = 17,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Standard 4x expansion
            dropout=dropout,
            batch_first=True  # Use batch_first for easier handling
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, vocab_size)
        
        # For activation extraction
        self.activation_hooks = {}
        self.activations = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
    
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
        embedded = self.pos_encoding(embedded)
        
        # Store embedding activations
        if 'embeddings' in self.activation_hooks:
            self.activations['embeddings'] = embedded.clone()
        
        # Apply transformer: (batch_size, 2, d_model) -> (batch_size, 2, d_model)  
        hidden_states = self.transformer(embedded)
        
        # Store transformer output activations
        if 'transformer_output' in self.activation_hooks:
            self.activations['transformer_output'] = hidden_states.clone()
        
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
                'hidden_states': hidden_states,
                'aggregated': aggregated
            }
        
        return logits
    
    def register_activation_hook(self, layer_name: str):
        """Register hook to capture activations from a specific layer."""
        if layer_name not in ['embeddings', 'transformer_output', 'aggregated']:
            raise ValueError(f"Unknown layer: {layer_name}")
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
            embedded = self.pos_encoding(embedded)
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


def create_model(vocab_size: int = 17, device: str = 'cpu') -> ModularTransformer:
    """
    Create and initialize a ModularTransformer model.
    
    Args:
        vocab_size: Size of vocabulary (typically p for mod p arithmetic)
        device: Device to place model on
        
    Returns:
        Initialized ModularTransformer model
    """
    model = ModularTransformer(vocab_size=vocab_size)
    model = model.to(device)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test the model with sample data
    model = create_model(vocab_size=17)
    
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
    
    # Test number embeddings
    number_embeddings = model.get_number_embeddings()
    print(f"Number embeddings shape: {number_embeddings.shape}")
    
    # Test prediction and accuracy
    predictions = model.predict_batch(x)
    accuracy = model.compute_accuracy(x, y)
    
    print(f"Predictions: {predictions.tolist()}")
    print(f"Ground truth: {y.tolist()}")
    print(f"Random accuracy: {accuracy:.3f}")
    
    print("✅ Transformer model implementation complete!")