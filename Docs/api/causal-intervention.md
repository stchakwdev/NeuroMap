# Causal Intervention API

Module for activation patching and ablation studies.

## Classes

### ActivationPatcher

Main class for performing activation interventions.

```python
from analysis.causal_intervention import ActivationPatcher

patcher = ActivationPatcher(model, device='cpu')
```

**Parameters:**
- `model` - HookedTransformer model instance
- `device` - Device for computation ('cpu' or 'cuda')

**Attributes:**
- `model` - The wrapped model
- `device` - Computation device
- `hook_points` - Dictionary mapping layer names to hook point objects

#### Methods

##### mean_ablation

```python
result = patcher.mean_ablation(inputs, targets, layer_name, positions=None)
```

Replace activations with dataset-wide mean.

**Parameters:**
- `inputs` - Input tensor [batch, seq_len]
- `targets` - Target tensor [batch]
- `layer_name` - Name of layer to ablate
- `positions` - Optional positions to ablate (default: all)

**Returns:** `AblationResult`
- `accuracy_drop` - Decrease in accuracy
- `loss_increase` - Increase in loss
- `original_accuracy` - Accuracy before ablation
- `ablated_accuracy` - Accuracy after ablation

##### zero_ablation

```python
result = patcher.zero_ablation(inputs, targets, layer_name, positions=None)
```

Set activations to zero.

**Parameters:** Same as `mean_ablation`

**Returns:** `AblationResult`

##### compute_patching_effect

```python
result = patcher.compute_patching_effect(
    clean_input, corrupted_input, targets, patch_layers
)
```

Measure effect of patching from corrupted to clean.

**Parameters:**
- `clean_input` - Clean input tensor
- `corrupted_input` - Corrupted input tensor
- `targets` - Target tensor
- `patch_layers` - List of layer names to patch

**Returns:** `PatchingResult`
- `clean_accuracy` - Accuracy on clean inputs
- `corrupted_accuracy` - Accuracy on corrupted inputs
- `patched_accuracy` - Accuracy after patching
- `effect_size` - (patched - corrupted) / (clean - corrupted)

##### patch_activations

```python
patched_output = patcher.patch_activations(
    source_input, target_input, patch_layers, patch_positions=None
)
```

Patch activations from source run into target run.

**Parameters:**
- `source_input` - Source input for activations
- `target_input` - Target input to patch into
- `patch_layers` - Layers to patch
- `patch_positions` - Positions to patch (optional)

**Returns:** Model output with patched activations

---

### AttentionPatcher

Specialized patcher for attention heads.

```python
from analysis.causal_intervention import AttentionPatcher

patcher = AttentionPatcher(model, device='cpu')
```

#### Methods

##### ablate_head

```python
result = patcher.ablate_head(
    inputs, targets, layer_idx, head_idx, ablation_type='zero'
)
```

Ablate a single attention head.

**Parameters:**
- `inputs` - Input tensor
- `targets` - Target tensor
- `layer_idx` - Layer index
- `head_idx` - Head index
- `ablation_type` - 'zero' or 'mean'

**Returns:** `AblationResult`

##### ablate_heads

```python
result = patcher.ablate_heads(inputs, targets, heads, ablation_type='zero')
```

Ablate multiple heads simultaneously.

**Parameters:**
- `heads` - List of (layer_idx, head_idx) tuples

---

### CausalAnalyzer

High-level analysis utilities.

```python
from analysis.causal_intervention import CausalAnalyzer

analyzer = CausalAnalyzer(model, device='cpu')
```

#### Methods

##### analyze_layer_importance

```python
importance = analyzer.analyze_layer_importance(
    dataset, n_samples=100, ablation_type='mean'
)
```

Compute importance scores for all layers.

**Parameters:**
- `dataset` - ModularArithmeticDataset instance
- `n_samples` - Number of samples to use
- `ablation_type` - 'mean' or 'zero'

**Returns:** Dictionary mapping layer names to importance scores

##### analyze_head_importance

```python
importance = analyzer.analyze_head_importance(dataset, n_samples=100)
```

Compute importance for all attention heads.

**Returns:** Dictionary mapping (layer, head) tuples to importance

---

## Helper Functions

### create_corrupted_inputs

```python
from analysis.causal_intervention import create_corrupted_inputs

corrupted = create_corrupted_inputs(inputs, method='shuffle', p=17)
```

Create corrupted versions of inputs for patching experiments.

**Parameters:**
- `inputs` - Original input tensor [batch, seq_len]
- `method` - Corruption method:
  - `'shuffle'` - Shuffle operands within batch
  - `'random'` - Replace with random values
  - `'shift'` - Shift values by constant
- `p` - Modulus (for value range)

**Returns:** Corrupted input tensor

---

## Data Classes

### AblationResult

```python
@dataclass
class AblationResult:
    layer_name: str
    accuracy_drop: float
    loss_increase: float
    original_accuracy: float
    ablated_accuracy: float
    original_loss: float
    ablated_loss: float
```

### PatchingResult

```python
@dataclass
class PatchingResult:
    patch_layers: List[str]
    clean_accuracy: float
    corrupted_accuracy: float
    patched_accuracy: float
    effect_size: float
```

---

## Example Usage

```python
from Dataset.dataset import ModularArithmeticDataset
from models.hooked_transformer import create_hooked_model
from analysis.causal_intervention import (
    ActivationPatcher,
    CausalAnalyzer,
    create_corrupted_inputs
)

# Setup
dataset = ModularArithmeticDataset(p=17)
model = create_hooked_model(vocab_size=17)
patcher = ActivationPatcher(model)

# Get data
inputs = dataset.data['inputs'][:100]
targets = dataset.data['targets'][:100]

# Mean ablation
result = patcher.mean_ablation(inputs, targets, 'embed')
print(f"Embedding importance: {result.accuracy_drop:.2%}")

# Activation patching
corrupted = create_corrupted_inputs(inputs, method='shuffle', p=17)
result = patcher.compute_patching_effect(
    inputs, corrupted, targets, ['blocks.0.hook_attn_out']
)
print(f"Attention effect: {result.effect_size:.4f}")

# Full analysis
analyzer = CausalAnalyzer(model)
importance = analyzer.analyze_layer_importance(dataset)
for layer, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"{layer}: {score:.4f}")
```
