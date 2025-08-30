"""
Scaled model architectures for larger vocabulary sizes.

These models adapt their capacity based on the vocabulary size,
maintaining the proven approach but scaling appropriately for complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


def get_scaled_dimensions(vocab_size: int):
    """Calculate scaled dimensions based on vocab size."""
    
    # Base scaling factors derived from our p=7 success
    base_vocab = 7
    scale_factor = vocab_size / base_vocab
    
    # Scale dimensions with square root to avoid over-parameterization
    dim_scale = math.sqrt(scale_factor)
    
    return {
        'embedding_dim': max(16, int(32 * dim_scale)),      # 32 for p=7, scales up
        'hidden_dim': max(32, int(64 * dim_scale)),         # Hidden layer sizes
        'mlp_hidden': max(64, int(128 * dim_scale)),        # MLP hidden sizes
        'transformer_ff': max(64, int(128 * dim_scale)),    # Transformer feedforward
    }


class ScaledLinearModel(nn.Module):
    """Linear model that scales with vocabulary size."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        dims = get_scaled_dimensions(vocab_size)
        
        embedding_dim = dims['embedding_dim']
        hidden_dim = dims['hidden_dim']
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Scale network depth based on complexity
        if vocab_size <= 7:
            # Simple network for small vocab
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocab_size)
            )
        else:
            # Slightly deeper for larger vocab
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, vocab_size)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        # Use smaller initialization for larger models
        std = 0.1 / math.sqrt(self.vocab_size / 7)
        nn.init.normal_(self.embedding.weight, std=std)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        embeddings = self.embedding(x)
        flattened = embeddings.view(embeddings.size(0), -1)
        logits = self.classifier(flattened)
        return logits


class ScaledTinyTransformer(nn.Module):
    """Transformer that scales with vocabulary size."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        dims = get_scaled_dimensions(vocab_size)
        
        d_model = dims['embedding_dim']
        ff_dim = dims['transformer_ff']
        
        # Scale attention heads appropriately
        if vocab_size <= 7:
            nhead = 2
            num_layers = 1
        elif vocab_size <= 17:
            nhead = 4
            num_layers = 1
            d_model = max(d_model, 64)  # Ensure divisible by nhead
        else:
            nhead = 4
            num_layers = 2
            d_model = max(d_model, 64)
        
        # Ensure d_model is divisible by nhead
        d_model = (d_model // nhead) * nhead
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(3, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        std = 0.1 / math.sqrt(self.vocab_size / 7)
        nn.init.normal_(self.embedding.weight, std=std)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        embeddings = self.embedding(x)
        embeddings = embeddings + self.pos_encoding[:2].unsqueeze(0)
        
        transformed = self.transformer(embeddings)
        pooled = transformed.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits


class ScaledMLP(nn.Module):
    """MLP that scales with vocabulary size and complexity."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        dims = get_scaled_dimensions(vocab_size)
        
        hidden_dim = dims['mlp_hidden']
        
        # Scale network depth based on vocabulary size
        if vocab_size <= 7:
            layers = [
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, vocab_size)
            ]
        elif vocab_size <= 17:
            layers = [
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, vocab_size)
            ]
        else:
            layers = [
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, vocab_size)
            ]
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x_float = x.float()
        logits = self.network(x_float)
        return logits


def create_scaled_model(model_type: str, vocab_size: int, device: str = 'cpu'):
    """Create scaled models for various vocabulary sizes."""
    
    if model_type == 'scaled_linear':
        model = ScaledLinearModel(vocab_size=vocab_size)
    elif model_type == 'scaled_transformer':
        model = ScaledTinyTransformer(vocab_size=vocab_size)
    elif model_type == 'scaled_mlp':
        model = ScaledMLP(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown scaled model type: {model_type}")
    
    model = model.to(device)
    return model


def get_scaling_info(vocab_size: int):
    """Get information about how dimensions scale with vocabulary size."""
    
    dims = get_scaled_dimensions(vocab_size)
    
    # Create models to get parameter counts
    models = {}
    param_counts = {}
    
    for model_type in ['scaled_linear', 'scaled_transformer', 'scaled_mlp']:
        model = create_scaled_model(model_type, vocab_size)
        models[model_type] = model
        param_counts[model_type] = sum(p.numel() for p in model.parameters())
    
    return {
        'vocab_size': vocab_size,
        'dimensions': dims,
        'parameter_counts': param_counts,
        'dataset_size': vocab_size * vocab_size,
        'param_to_data_ratios': {
            k: v / (vocab_size * vocab_size) 
            for k, v in param_counts.items()
        }
    }


def print_scaling_analysis():
    """Print analysis of how models scale across different vocabulary sizes."""
    
    print("Scaled Model Analysis")
    print("=" * 60)
    
    vocab_sizes = [7, 13, 17, 23]
    
    print(f"{'Vocab':<6} {'Examples':<9} {'Model':<17} {'Params':<8} {'Ratio':<8}")
    print("-" * 60)
    
    for vocab_size in vocab_sizes:
        info = get_scaling_info(vocab_size)
        dataset_size = info['dataset_size']
        
        for model_type, param_count in info['parameter_counts'].items():
            ratio = info['param_to_data_ratios'][model_type]
            
            print(f"{vocab_size:<6} {dataset_size:<9} {model_type:<17} "
                  f"{param_count:<8} {ratio:<8.2f}")
        print()


if __name__ == "__main__":
    print("Testing scaled model architectures...\n")
    
    # Test all vocab sizes
    vocab_sizes = [7, 13, 17, 23]
    model_types = ['scaled_linear', 'scaled_transformer', 'scaled_mlp']
    
    for vocab_size in vocab_sizes:
        print(f"Vocab size {vocab_size}:")
        dims = get_scaled_dimensions(vocab_size)
        print(f"  Dimensions: {dims}")
        
        for model_type in model_types:
            model = create_scaled_model(model_type, vocab_size)
            params = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            batch_size = 4
            x = torch.randint(0, vocab_size, (batch_size, 2))
            
            with torch.no_grad():
                output = model(x)
            
            print(f"  {model_type}: {params:,} params, output {output.shape}")
        print()
    
    print("\nScaling Analysis:")
    print_scaling_analysis()
    
    print("✅ All scaled models created successfully!")