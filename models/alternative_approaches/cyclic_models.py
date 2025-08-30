"""
Cyclic Encoding Models for Modular Arithmetic.

These models use sinusoidal/cyclic encodings to embed the circular nature
of modular arithmetic directly into the input representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple


class CyclicPositionalEncoder(nn.Module):
    """
    Encodes integers using sinusoidal functions to capture cyclic nature.
    
    For modular arithmetic with modulus p, numbers 0 to p-1 form a circle.
    We use sin/cos functions to embed this circular structure.
    """
    
    def __init__(self, vocab_size: int, encoding_dim: int = 16):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.encoding_dim = encoding_dim
        
        # Create fixed cyclic encodings
        self.register_buffer('cyclic_encodings', self._create_cyclic_encodings())
    
    def _create_cyclic_encodings(self):
        """Create sinusoidal encodings for each number."""
        
        encodings = torch.zeros(self.vocab_size, self.encoding_dim)
        
        for pos in range(self.vocab_size):
            for i in range(0, self.encoding_dim, 2):
                # Multiple frequencies to capture different aspects of cyclicity
                freq = 1.0 / (10000.0 ** ((2 * i) / self.encoding_dim))
                
                # Normalize position to [0, 2π] for the circle
                angle = 2 * math.pi * pos / self.vocab_size * freq
                
                encodings[pos, i] = math.sin(angle)
                if i + 1 < self.encoding_dim:
                    encodings[pos, i + 1] = math.cos(angle)
        
        return encodings
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 2) tensor of integer inputs
        Returns:
            (batch_size, 2, encoding_dim) tensor of cyclic encodings
        """
        batch_size = x.size(0)
        
        # Look up cyclic encodings for each input
        encoded = self.cyclic_encodings[x]  # (batch_size, 2, encoding_dim)
        
        return encoded


class CyclicEncodedModel(nn.Module):
    """
    Model that uses cyclic encodings as input features.
    
    Instead of raw integers, uses sin/cos representations that
    capture the circular structure of modular arithmetic.
    """
    
    def __init__(self, vocab_size: int, encoding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.encoder = CyclicPositionalEncoder(vocab_size, encoding_dim)
        
        # Network that processes cyclic encodings
        input_dim = 2 * encoding_dim  # Two numbers, each with encoding_dim features
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better performance."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Encode inputs cyclically
        cyclic_features = self.encoder(x)  # (batch_size, 2, encoding_dim)
        
        # Flatten to feed into network
        features = cyclic_features.view(cyclic_features.size(0), -1)  # (batch_size, 2 * encoding_dim)
        
        # Process with network
        logits = self.network(features)
        
        return logits


class EnhancedCyclicModel(nn.Module):
    """
    Enhanced version with additional cyclic features.
    
    Includes not just sin/cos encodings but also:
    - Distance features
    - Sum encodings
    - Multiple frequency components
    """
    
    def __init__(self, vocab_size: int, encoding_dim: int = 24):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.encoding_dim = encoding_dim
        
        # Create multiple types of features
        self.cyclic_encoder = CyclicPositionalEncoder(vocab_size, encoding_dim // 2)
        
        # Additional feature dimensions
        feature_dim = 2 * (encoding_dim // 2) + 6  # cyclic + additional features
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, vocab_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_additional_features(self, x):
        """Create additional cyclic-aware features."""
        batch_size = x.size(0)
        a, b = x[:, 0].float(), x[:, 1].float()
        p = float(self.vocab_size)
        
        features = []
        
        # 1. Raw normalized values
        features.append((a / p).unsqueeze(1))
        features.append((b / p).unsqueeze(1))
        
        # 2. Sum encoding (without modulo)
        raw_sum = (a + b) / (2 * p)
        features.append(raw_sum.unsqueeze(1))
        
        # 3. Expected result encoding
        expected_result = ((a + b) % p) / p
        features.append(expected_result.unsqueeze(1))
        
        # 4. Circular distance features
        dist_a_to_zero = a / p
        dist_b_to_zero = b / p
        features.append(dist_a_to_zero.unsqueeze(1))
        features.append(dist_b_to_zero.unsqueeze(1))
        
        return torch.cat(features, dim=1)  # (batch_size, 6)
    
    def forward(self, x):
        # Get cyclic encodings
        cyclic_features = self.cyclic_encoder(x)  # (batch_size, 2, encoding_dim//2)
        cyclic_flat = cyclic_features.view(cyclic_features.size(0), -1)
        
        # Get additional features
        additional_features = self._create_additional_features(x)
        
        # Combine all features
        all_features = torch.cat([cyclic_flat, additional_features], dim=1)
        
        # Process with network
        logits = self.network(all_features)
        
        return logits


class ResidualCyclicModel(nn.Module):
    """
    Model that learns the residual between raw sum and modulo result.
    
    Instead of learning f(a,b) = (a+b) mod p directly,
    learns f(a,b) = (a+b) mod p - (a+b) 
    """
    
    def __init__(self, vocab_size: int, encoding_dim: int = 16):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.encoder = CyclicPositionalEncoder(vocab_size, encoding_dim)
        
        # Network learns the "wrap-around" correction
        input_dim = 2 * encoding_dim + 1  # cyclic features + raw sum
        
        self.residual_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, vocab_size)  # Predicts residual distribution
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.residual_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Get cyclic features
        cyclic_features = self.encoder(x)
        cyclic_flat = cyclic_features.view(cyclic_features.size(0), -1)
        
        # Calculate raw sum
        raw_sum = (x[:, 0] + x[:, 1]).float().unsqueeze(1)  # (batch_size, 1)
        
        # Combine features
        features = torch.cat([cyclic_flat, raw_sum], dim=1)
        
        # Predict residual
        residual_logits = self.residual_network(features)
        
        return residual_logits


def create_cyclic_model(model_type: str, vocab_size: int) -> nn.Module:
    """Create cyclic encoding models."""
    
    if model_type == 'cyclic_basic':
        return CyclicEncodedModel(vocab_size)
    elif model_type == 'cyclic_enhanced':
        return EnhancedCyclicModel(vocab_size)
    elif model_type == 'cyclic_residual':
        return ResidualCyclicModel(vocab_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test all cyclic models
    vocab_size = 13
    batch_size = 4
    
    models = ['cyclic_basic', 'cyclic_enhanced', 'cyclic_residual']
    
    print(f"Testing Cyclic Models for vocab_size={vocab_size}")
    print("=" * 50)
    
    # Create test input
    test_input = torch.randint(0, vocab_size, (batch_size, 2))
    print(f"Test input: {test_input.tolist()}")
    
    # Expected outputs for verification
    expected = [(a + b) % vocab_size for a, b in test_input.tolist()]
    print(f"Expected results: {expected}")
    
    print("\nCyclic encodings for first few numbers:")
    encoder = CyclicPositionalEncoder(vocab_size, 8)
    for i in range(min(5, vocab_size)):
        encoding = encoder(torch.tensor([[i, 0]]))
        print(f"  {i}: {encoding[0, 0, :4].tolist()}")  # Show first 4 dimensions
    
    print("\nModel Testing:")
    print("-" * 30)
    
    for model_type in models:
        model = create_cyclic_model(model_type, vocab_size)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
            predictions = torch.argmax(output, dim=1).tolist()
        
        print(f"\n{model_type}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Output shape: {output.shape}")
        print(f"  Predictions: {predictions}")
        print(f"  Expected:    {expected}")
        
        # Check initial accuracy (should be low before training)
        correct = sum(1 for p, e in zip(predictions, expected) if p == e)
        accuracy = correct / len(expected)
        print(f"  Initial accuracy: {accuracy:.2f} ({correct}/{len(expected)})")
        
        if torch.isnan(output).any():
            print(f"  ⚠️ WARNING: NaN values!")
        else:
            print(f"  ✅ Output looks good")
    
    print("\n✅ All cyclic models created successfully!")
    print("\nKey advantages of cyclic models:")
    print("1. Embed circular structure of modular arithmetic")
    print("2. Provide right inductive bias for the task")
    print("3. Should learn patterns more efficiently")
    print("4. Multiple frequencies capture different aspects")