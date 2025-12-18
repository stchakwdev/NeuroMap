# Hooked Transformer API

TransformerLens-compatible transformer architecture with hook points for interpretability.

## Classes

### NeuroMapHookedTransformer

Main transformer model with activation hooks.

```python
from models.hooked_transformer import NeuroMapHookedTransformer, NeuroMapTransformerConfig

config = NeuroMapTransformerConfig(
    vocab_size=17,
    n_layers=2,
    d_model=64,
    n_heads=4,
    d_head=16,
    d_mlp=256
)
model = NeuroMapHookedTransformer(config)
```

**Config Parameters:**
- `vocab_size` - Vocabulary size (typically p for mod-p arithmetic)
- `n_layers` - Number of transformer layers
- `d_model` - Model dimension
- `n_heads` - Number of attention heads
- `d_head` - Dimension per head
- `d_mlp` - MLP hidden dimension
- `act_fn` - Activation function ('relu', 'gelu')
- `device` - Computation device

#### Methods

##### forward

```python
output = model(input_ids)
```

Standard forward pass.

**Parameters:**
- `input_ids` - Input tensor [batch, seq_len]

**Returns:** Logits tensor [batch, vocab_size]

##### run_with_cache

```python
output, cache = model.run_with_cache(input_ids)
```

Forward pass with activation caching.

**Parameters:**
- `input_ids` - Input tensor [batch, seq_len]

**Returns:** Tuple of (logits, cache_dict)

Cache contains activations at all hook points:
- `hook_embed` - Embeddings [batch, seq, d_model]
- `blocks.{i}.hook_attn_out` - Attention output [batch, seq, d_model]
- `blocks.{i}.hook_mlp_out` - MLP output [batch, seq, d_model]
- `blocks.{i}.hook_resid_pre` - Residual before block [batch, seq, d_model]
- `blocks.{i}.hook_resid_post` - Residual after block [batch, seq, d_model]

##### get_number_embeddings

```python
embeddings = model.get_number_embeddings()
```

Get embeddings for all vocabulary tokens.

**Returns:** Tensor [vocab_size, d_model]

##### to_transformerlens

```python
tl_model = model.to_transformerlens()
```

Convert to TransformerLens HookedTransformer for ecosystem compatibility.

**Returns:** TransformerLens HookedTransformer instance

##### add_hook

```python
model.add_hook(hook_point_name, hook_fn)
```

Add a hook function to a specific hook point.

**Parameters:**
- `hook_point_name` - Name of hook point (e.g., 'blocks.0.hook_attn_out')
- `hook_fn` - Function taking (tensor, hook) and returning modified tensor

##### remove_hooks

```python
model.remove_hooks()
```

Remove all added hooks.

---

### HookPoint

Hook point for capturing and modifying activations.

```python
from models.hooked_transformer import HookPoint

hook = HookPoint()
```

Used internally by the model. Each hook point can:
- Store activations during forward pass
- Apply hook functions to modify activations
- Support multiple simultaneous hooks

---

## Factory Functions

### create_hooked_model

```python
from models.hooked_transformer import create_hooked_model

model = create_hooked_model(
    vocab_size=17,
    n_layers=2,
    d_model=64,
    n_heads=4,
    d_head=16,
    d_mlp=256,
    device='cpu'
)
```

Convenience function to create a model with sensible defaults.

**Parameters:**
- `vocab_size` - Required
- `n_layers` - Default: 2
- `d_model` - Default: 64
- `n_heads` - Default: 4
- `d_head` - Default: 16
- `d_mlp` - Default: 256
- `device` - Default: 'cpu'

**Returns:** NeuroMapHookedTransformer instance

---

## Hook Points

Standard hook points available in the model:

| Hook Name | Shape | Description |
|-----------|-------|-------------|
| `hook_embed` | [batch, seq, d_model] | After embedding + position |
| `blocks.{i}.hook_resid_pre` | [batch, seq, d_model] | Before attention + MLP |
| `blocks.{i}.hook_attn_out` | [batch, seq, d_model] | After attention |
| `blocks.{i}.hook_mlp_out` | [batch, seq, d_model] | After MLP |
| `blocks.{i}.hook_resid_post` | [batch, seq, d_model] | After residual add |
| `blocks.{i}.attn.hook_pattern` | [batch, heads, seq, seq] | Attention patterns |
| `blocks.{i}.attn.hook_v` | [batch, seq, heads, d_head] | Value vectors |

---

## Example Usage

### Basic Forward Pass

```python
from models.hooked_transformer import create_hooked_model

model = create_hooked_model(vocab_size=17)
inputs = torch.tensor([[1, 2]])  # a=1, b=2
output = model(inputs)
predicted = output.argmax(-1)
print(f"1 + 2 mod 17 = {predicted.item()}")
```

### Caching Activations

```python
model = create_hooked_model(vocab_size=17)
inputs = torch.tensor([[5, 7]])

output, cache = model.run_with_cache(inputs)

# Access cached activations
embeddings = cache['hook_embed']
attn_out_0 = cache['blocks.0.hook_attn_out']
mlp_out_1 = cache['blocks.1.hook_mlp_out']

print(f"Embedding shape: {embeddings.shape}")
print(f"L0 attention output shape: {attn_out_0.shape}")
```

### Adding Hooks

```python
def ablate_hook(tensor, hook):
    """Zero out activations."""
    return torch.zeros_like(tensor)

model = create_hooked_model(vocab_size=17)
model.add_hook('blocks.0.hook_attn_out', ablate_hook)

# Forward pass now applies the hook
output = model(inputs)  # L0 attention output is zeroed

model.remove_hooks()  # Clean up
```

### TransformerLens Integration

```python
# Convert to TransformerLens for ecosystem tools
model = create_hooked_model(vocab_size=17)
tl_model = model.to_transformerlens()

# Use TransformerLens functionality
from transformer_lens import ActivationCache
output, cache = tl_model.run_with_cache(inputs)
```

---

## Configuration Presets

```python
from models.model_configs import get_config

# Standard 2-layer for modular arithmetic
config = get_config('standard_2l')
model = NeuroMapHookedTransformer(config)

# Grokking configuration (longer training)
config = get_config('grokking')

# Tiny model for quick experiments
config = get_config('tiny')
```
