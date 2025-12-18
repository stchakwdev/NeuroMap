"""
TransformerLens Integration for NeuroMap.

Provides HookedTransformer-compatible architecture for modular arithmetic,
enabling seamless integration with the TransformerLens ecosystem for
mechanistic interpretability research.

This module creates transformer models that:
1. Have standard hook points (hook_embed, hook_attn_out, hook_mlp_out, etc.)
2. Support run_with_cache() for activation caching
3. Are compatible with NeuroMap analysis pipeline
4. Follow TransformerLens naming conventions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import math
from pathlib import Path
import numpy as np

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    TRANSFORMERLENS_AVAILABLE = True
except ImportError:
    TRANSFORMERLENS_AVAILABLE = False


@dataclass
class NeuroMapTransformerConfig:
    """Configuration for NeuroMap Transformer."""
    n_layers: int = 2
    d_model: int = 64
    d_head: int = 16
    n_heads: int = 4
    d_mlp: int = 256
    d_vocab: int = 17  # Modulus p
    n_ctx: int = 2  # Context length (a, b)
    act_fn: str = "relu"
    normalization_type: str = "LN"  # "LN" or "LNPre" or None
    use_attn_scale: bool = True
    use_local_attn: bool = False
    device: str = "cpu"
    seed: int = 42


class HookPoint(nn.Module):
    """
    A helper class for hooking into model activations.

    Mimics TransformerLens HookPoint for compatibility.
    """

    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
        self._hooks: List[Tuple[Callable, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through, but allow hooks to capture/modify the tensor."""
        for hook_fn, modify in self._hooks:
            result = hook_fn(x)
            if modify and result is not None:
                x = result
        return x

    def add_hook(self, hook_fn: Callable, modify: bool = False):
        """Add a hook function."""
        self._hooks.append((hook_fn, modify))

    def clear_hooks(self):
        """Remove all hooks."""
        self._hooks.clear()


class Attention(nn.Module):
    """
    Multi-head attention with TransformerLens-style hook points.
    """

    def __init__(self, config: NeuroMapTransformerConfig):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_K = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_V = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_O = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        # Hook points
        self.hook_q = HookPoint("hook_q")
        self.hook_k = HookPoint("hook_k")
        self.hook_v = HookPoint("hook_v")
        self.hook_attn_scores = HookPoint("hook_attn_scores")
        self.hook_pattern = HookPoint("hook_pattern")  # Post-softmax attention
        self.hook_z = HookPoint("hook_z")  # Attention output before projection
        self.hook_result = HookPoint("hook_result")  # Final attention output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.W_Q(x).view(batch, seq_len, self.config.n_heads, self.config.d_head)
        k = self.W_K(x).view(batch, seq_len, self.config.n_heads, self.config.d_head)
        v = self.W_V(x).view(batch, seq_len, self.config.n_heads, self.config.d_head)

        # Apply hooks
        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        # Compute attention scores
        # q, k: (batch, seq, n_heads, d_head)
        q = q.transpose(1, 2)  # (batch, n_heads, seq, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.config.d_head) if self.config.use_attn_scale else 1.0
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = self.hook_attn_scores(attn_scores)

        # Softmax
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = self.hook_pattern(pattern)

        # Apply attention to values
        z = torch.matmul(pattern, v)  # (batch, n_heads, seq, d_head)
        z = self.hook_z(z)

        # Reshape and project
        z = z.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        result = self.W_O(z)
        result = self.hook_result(result)

        return result


class MLP(nn.Module):
    """
    MLP with TransformerLens-style hook points.
    """

    def __init__(self, config: NeuroMapTransformerConfig):
        super().__init__()
        self.config = config

        self.W_in = nn.Linear(config.d_model, config.d_mlp)
        self.W_out = nn.Linear(config.d_mlp, config.d_model)

        # Activation function
        if config.act_fn == "relu":
            self.act_fn = F.relu
        elif config.act_fn == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu

        # Hook points
        self.hook_pre = HookPoint("hook_pre")
        self.hook_post = HookPoint("hook_post")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.W_in(x)
        pre = self.hook_pre(pre)

        post = self.act_fn(pre)
        post = self.hook_post(post)

        out = self.W_out(post)
        return out


class TransformerBlock(nn.Module):
    """
    A single transformer block with attention and MLP.
    """

    def __init__(self, config: NeuroMapTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = Attention(config)
        self.mlp = MLP(config)

        if config.normalization_type == "LN":
            self.ln1 = nn.LayerNorm(config.d_model)
            self.ln2 = nn.LayerNorm(config.d_model)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # Hook points for residual stream
        self.hook_resid_pre = HookPoint("hook_resid_pre")
        self.hook_resid_mid = HookPoint("hook_resid_mid")
        self.hook_resid_post = HookPoint("hook_resid_post")
        self.hook_attn_out = HookPoint("hook_attn_out")
        self.hook_mlp_out = HookPoint("hook_mlp_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-residual hook
        x = self.hook_resid_pre(x)

        # Attention with residual
        attn_out = self.attn(self.ln1(x))
        attn_out = self.hook_attn_out(attn_out)
        x = x + attn_out

        # Mid-residual hook
        x = self.hook_resid_mid(x)

        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))
        mlp_out = self.hook_mlp_out(mlp_out)
        x = x + mlp_out

        # Post-residual hook
        x = self.hook_resid_post(x)

        return x


class NeuroMapHookedTransformer(nn.Module):
    """
    A HookedTransformer-compatible model for modular arithmetic.

    Provides TransformerLens-style interface while being optimized
    for the modular arithmetic task.

    Usage:
        model = NeuroMapHookedTransformer(config)
        output, cache = model.run_with_cache(input)
    """

    def __init__(self, config: NeuroMapTransformerConfig):
        super().__init__()
        self.config = config

        # Set device and seed
        self.device = config.device
        torch.manual_seed(config.seed)

        # Embedding
        self.embed = nn.Embedding(config.d_vocab, config.d_model)

        # Positional embedding (learnable)
        self.pos_embed = nn.Embedding(config.n_ctx, config.d_model)

        # Hook points for embeddings
        self.hook_embed = HookPoint("hook_embed")
        self.hook_pos_embed = HookPoint("hook_pos_embed")

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])

        # Final layer norm
        if config.normalization_type == "LN":
            self.ln_final = nn.LayerNorm(config.d_model)
        else:
            self.ln_final = nn.Identity()

        # Unembedding (output projection)
        self.unembed = nn.Linear(config.d_model, config.d_vocab)

        # Hook points
        self.hook_resid_final = HookPoint("hook_resid_final")

        # Cache for activations
        self._cache: Dict[str, torch.Tensor] = {}
        self._caching = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len) with token indices

        Returns:
            Logits of shape (batch, d_vocab)
        """
        batch, seq_len = x.shape

        # Token embeddings
        tok_embed = self.embed(x)
        tok_embed = self.hook_embed(tok_embed)

        # Position embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_embed = self.pos_embed(positions)
        pos_embed = self.hook_pos_embed(pos_embed)

        # Combine embeddings
        residual = tok_embed + pos_embed

        # Store in cache if enabled
        if self._caching:
            self._cache["hook_embed"] = tok_embed.detach().clone()
            self._cache["hook_pos_embed"] = pos_embed.detach().clone()

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            residual = block(residual)

            if self._caching:
                self._cache[f"blocks.{i}.hook_resid_post"] = residual.detach().clone()
                self._cache[f"blocks.{i}.hook_attn_out"] = block.hook_attn_out(
                    torch.zeros_like(residual)
                )

        # Final layer norm
        residual = self.ln_final(residual)
        residual = self.hook_resid_final(residual)

        if self._caching:
            self._cache["hook_resid_final"] = residual.detach().clone()

        # Aggregate: use last position or mean
        aggregated = residual.mean(dim=1)  # (batch, d_model)

        # Unembed to get logits
        logits = self.unembed(aggregated)

        return logits

    def run_with_cache(self,
                       x: torch.Tensor,
                       names_filter: Optional[List[str]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and return cached activations.

        Args:
            x: Input tensor
            names_filter: Optional list of hook names to cache (None = cache all)

        Returns:
            Tuple of (output logits, cache dictionary)
        """
        self._caching = True
        self._cache.clear()

        # Set up hooks for capturing activations
        hooks = []

        def make_cache_hook(name):
            def hook_fn(tensor):
                if names_filter is None or name in names_filter:
                    self._cache[name] = tensor.detach().clone()
                return tensor
            return hook_fn

        # Add hooks to all hook points
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.add_hook(make_cache_hook(name), modify=False)
                hooks.append(module)

        # Run forward pass
        with torch.no_grad():
            output = self(x)

        # Clear hooks
        for module in hooks:
            module.clear_hooks()

        self._caching = False

        return output, self._cache.copy()

    def get_number_embeddings(self) -> torch.Tensor:
        """Get the embedding vectors for all numbers 0 to d_vocab-1."""
        indices = torch.arange(self.config.d_vocab, device=self.device)
        return self.embed(indices)

    @classmethod
    def from_config(cls, config: NeuroMapTransformerConfig) -> "NeuroMapHookedTransformer":
        """Create model from config."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: Path) -> "NeuroMapHookedTransformer":
        """Load a pretrained model."""
        checkpoint = torch.load(path, map_location='cpu')

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Infer config from state dict
            config = NeuroMapTransformerConfig()

        model = cls(config)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        return model

    def to_transformerlens(self) -> Optional["HookedTransformer"]:
        """
        Convert to a TransformerLens HookedTransformer.

        Requires transformer_lens to be installed.

        Returns:
            HookedTransformer or None if transformer_lens not available
        """
        if not TRANSFORMERLENS_AVAILABLE:
            print("TransformerLens not installed. Install with: pip install transformer-lens")
            return None

        tl_config = HookedTransformerConfig(
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            d_head=self.config.d_head,
            n_heads=self.config.n_heads,
            d_mlp=self.config.d_mlp,
            d_vocab=self.config.d_vocab,
            n_ctx=self.config.n_ctx,
            act_fn=self.config.act_fn,
            normalization_type=self.config.normalization_type,
        )

        tl_model = HookedTransformer(tl_config)

        # Copy weights
        with torch.no_grad():
            tl_model.embed.W_E.copy_(self.embed.weight)
            tl_model.pos_embed.W_pos.copy_(self.pos_embed.weight)

            for i, block in enumerate(self.blocks):
                # Attention weights
                tl_model.blocks[i].attn.W_Q.copy_(
                    block.attn.W_Q.weight.view(
                        self.config.n_heads, self.config.d_head, self.config.d_model
                    ).transpose(-1, -2)
                )
                tl_model.blocks[i].attn.W_K.copy_(
                    block.attn.W_K.weight.view(
                        self.config.n_heads, self.config.d_head, self.config.d_model
                    ).transpose(-1, -2)
                )
                tl_model.blocks[i].attn.W_V.copy_(
                    block.attn.W_V.weight.view(
                        self.config.n_heads, self.config.d_head, self.config.d_model
                    ).transpose(-1, -2)
                )
                tl_model.blocks[i].attn.W_O.copy_(
                    block.attn.W_O.weight.view(
                        self.config.d_model, self.config.n_heads, self.config.d_head
                    )
                )

                # MLP weights
                tl_model.blocks[i].mlp.W_in.copy_(block.mlp.W_in.weight.T)
                tl_model.blocks[i].mlp.W_out.copy_(block.mlp.W_out.weight.T)

            # Unembed
            tl_model.unembed.W_U.copy_(self.unembed.weight.T)

        return tl_model


def create_hooked_model(
    vocab_size: int = 17,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    d_mlp: int = 256,
    device: str = 'cpu'
) -> NeuroMapHookedTransformer:
    """
    Create a NeuroMapHookedTransformer with the specified configuration.

    Args:
        vocab_size: Size of vocabulary (modulus p)
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_mlp: MLP hidden dimension
        device: Device to place model on

    Returns:
        Configured NeuroMapHookedTransformer
    """
    config = NeuroMapTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=vocab_size,
        device=device
    )

    model = NeuroMapHookedTransformer(config)
    model.to(device)

    return model


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    print("Testing NeuroMapHookedTransformer")
    print("=" * 50)

    # Create model
    config = NeuroMapTransformerConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_mlp=256,
        d_vocab=17,
    )
    model = NeuroMapHookedTransformer(config)
    print(f"Created model with config: {config}")

    # Test forward pass
    batch_size = 8
    inputs = torch.randint(0, 17, (batch_size, 2))
    outputs = model(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")

    # Test run_with_cache
    outputs, cache = model.run_with_cache(inputs)
    print(f"Cached activations: {list(cache.keys())[:5]}...")

    # Test number embeddings
    embeddings = model.get_number_embeddings()
    print(f"Number embeddings shape: {embeddings.shape}")

    # Test TransformerLens conversion
    if TRANSFORMERLENS_AVAILABLE:
        tl_model = model.to_transformerlens()
        print("Successfully converted to TransformerLens HookedTransformer")
    else:
        print("TransformerLens not available (optional)")

    print("\nNeuroMapHookedTransformer implementation complete!")
