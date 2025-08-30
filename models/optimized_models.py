"""
Optimized model architectures for small datasets.

These models are specifically designed for the modular arithmetic task
with very limited data (49 examples).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class TinyTransformer(nn.Module):
    """Minimal transformer optimized for small datasets."""
    
    def __init__(self, 
                 vocab_size: int = 7,
                 d_model: int = 32,
                 nhead: int = 2,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (simple)
        self.pos_encoding = nn.Parameter(torch.randn(3, d_model) * 0.1)
        
        # Single transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, vocab_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better small dataset performance."""
        nn.init.normal_(self.embedding.weight, std=0.1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 2) - two numbers for addition
        batch_size = x.size(0)
        
        # Embed inputs
        embeddings = self.embedding(x)  # (batch_size, 2, d_model)
        
        # Add positional encoding
        embeddings = embeddings + self.pos_encoding[:2].unsqueeze(0)
        
        # Transform
        transformed = self.transformer(embeddings)  # (batch_size, 2, d_model)
        
        # Pool (simple mean)
        pooled = transformed.mean(dim=1)  # (batch_size, d_model)
        
        # Classify
        logits = self.classifier(pooled)  # (batch_size, vocab_size)
        
        return logits


class LinearModel(nn.Module):
    """Simple linear model as baseline."""
    
    def __init__(self, vocab_size: int = 7, hidden_dim: int = 32):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 2)
        embeddings = self.embedding(x)  # (batch_size, 2, hidden_dim)
        flattened = embeddings.view(embeddings.size(0), -1)  # (batch_size, hidden_dim * 2)
        logits = self.classifier(flattened)
        return logits


class TinyMamba(nn.Module):
    """Simplified Mamba-style model optimized for small datasets."""
    
    def __init__(self, vocab_size: int = 7, d_model: int = 32, expand_factor: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple state space layer
        self.input_proj = nn.Linear(d_model, self.d_inner)
        self.state_proj = nn.Linear(d_model, self.d_inner)
        self.output_proj = nn.Linear(self.d_inner, d_model)
        
        # Classification head
        self.classifier = nn.Linear(d_model, vocab_size)
        
        # State parameters
        self.A = nn.Parameter(torch.randn(self.d_inner) * 0.1)
        self.B = nn.Parameter(torch.randn(self.d_inner) * 0.1)
        self.C = nn.Parameter(torch.randn(self.d_inner) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        for layer in [self.input_proj, self.state_proj, self.output_proj, self.classifier]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 2)
        batch_size = x.size(0)
        
        # Embed
        embeddings = self.embedding(x)  # (batch_size, 2, d_model)
        
        # Process sequence
        h = torch.zeros(batch_size, self.d_inner, device=x.device)
        outputs = []
        
        for t in range(2):  # Two time steps
            x_t = embeddings[:, t, :]  # (batch_size, d_model)
            
            # State space update
            u_t = self.input_proj(x_t)  # (batch_size, d_inner)
            h = self.A * h + self.B * u_t  # (batch_size, d_inner)
            y_t = self.C * h  # (batch_size, d_inner)
            
            # Project back
            out_t = self.output_proj(y_t)  # (batch_size, d_model)
            outputs.append(out_t)
        
        # Pool outputs
        final_output = torch.stack(outputs, dim=1).mean(dim=1)  # (batch_size, d_model)
        
        # Classify
        logits = self.classifier(final_output)
        return logits


class MLP(nn.Module):
    """Simple MLP for modular arithmetic."""
    
    def __init__(self, vocab_size: int = 7):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Direct mapping from two integers to output
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, vocab_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 2) - integers
        x_float = x.float()  # Convert to float for linear layers
        logits = self.network(x_float)
        return logits


def create_optimized_model(model_type: str, vocab_size: int = 7, device: str = 'cpu'):
    """Create optimized models for small dataset training."""
    
    if model_type == 'tiny_transformer':
        model = TinyTransformer(vocab_size=vocab_size)
    elif model_type == 'linear':
        model = LinearModel(vocab_size=vocab_size)
    elif model_type == 'tiny_mamba':
        model = TinyMamba(vocab_size=vocab_size)
    elif model_type == 'mlp':
        model = MLP(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model


def get_model_info(model):
    """Get model parameter count and other info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'name': model.__class__.__name__
    }


if __name__ == "__main__":
    # Test all models
    vocab_size = 7
    batch_size = 4
    
    models = ['tiny_transformer', 'linear', 'tiny_mamba', 'mlp']
    
    print("Model Comparison for Small Dataset Training:")
    print("=" * 60)
    
    for model_type in models:
        model = create_optimized_model(model_type, vocab_size)
        info = get_model_info(model)
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (batch_size, 2))
        with torch.no_grad():
            output = model(x)
        
        print(f"{info['name']:<15} | {info['total_parameters']:<8} params | "
              f"{info['model_size_mb']:<6.2f} MB | Output: {output.shape}")
    
    print("\n✅ All models created successfully!")