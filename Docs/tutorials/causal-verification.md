# Causal Verification Tutorial

This tutorial demonstrates how to verify that discovered neural network structures are causally important for computation, not just correlational artifacts.

## Background

When analyzing neural networks, we often find interesting patterns in activations or embeddings. But how do we know these patterns are actually used by the network? Causal verification answers this question through interventions.

### Key Concepts

- **Activation Patching**: Replace activations from one input with activations from another
- **Mean Ablation**: Replace activations with their dataset-wide mean
- **Zero Ablation**: Set activations to zero
- **Effect Size**: How much the intervention changes model behavior

## Setup

```python
import torch
from Dataset.dataset import ModularArithmeticDataset
from models.hooked_transformer import create_hooked_model
from analysis.causal_intervention import (
    ActivationPatcher,
    CausalAnalyzer,
    create_corrupted_inputs
)

# Create dataset and trained model
p = 17
dataset = ModularArithmeticDataset(p=p)
model = create_hooked_model(vocab_size=p, n_layers=2)
device = 'cpu'

# Assume model is trained to high accuracy
# model.load_state_dict(torch.load('model.pt'))
model.eval()
```

## Mean Ablation

Mean ablation tests how important a layer is by replacing its activations with the mean activation across all inputs.

```python
patcher = ActivationPatcher(model, device=device)

# Prepare inputs
inputs = dataset.data['inputs'][:100].to(device)
targets = dataset.data['targets'][:100].to(device)

# Test each layer
layers = ['embed', 'blocks.0.hook_attn_out', 'blocks.0.hook_mlp_out',
          'blocks.1.hook_attn_out', 'blocks.1.hook_mlp_out']

print("Layer Importance via Mean Ablation:")
print("-" * 50)
for layer in layers:
    result = patcher.mean_ablation(inputs, targets, layer)
    print(f"{layer:30s} accuracy_drop={result.accuracy_drop:6.2%} "
          f"loss_increase={result.loss_increase:.3f}")
```

Expected output for a trained model:
```
embed                          accuracy_drop=45.00% loss_increase=2.341
blocks.0.hook_attn_out         accuracy_drop=23.00% loss_increase=1.123
blocks.0.hook_mlp_out          accuracy_drop=18.00% loss_increase=0.892
blocks.1.hook_attn_out         accuracy_drop=12.00% loss_increase=0.567
blocks.1.hook_mlp_out          accuracy_drop=31.00% loss_increase=1.456
```

### Interpretation

- **High accuracy drop** = layer is crucial for computation
- **Embeddings** typically show high importance (encode input structure)
- **Final MLP** often important (produces output)
- **Low drop** = layer may be redundant or doing cleanup

## Activation Patching

Activation patching measures how much information flows through a layer by patching from corrupted to clean inputs.

```python
# Create corrupted inputs (shuffle operands)
corrupted = create_corrupted_inputs(inputs, method='shuffle', p=p)

# Patch at specific layer
result = patcher.compute_patching_effect(
    clean_input=inputs,
    corrupted_input=corrupted,
    targets=targets,
    patch_layers=['blocks.0.hook_attn_out']
)

print(f"Clean accuracy: {result.clean_accuracy:.2%}")
print(f"Corrupted accuracy: {result.corrupted_accuracy:.2%}")
print(f"Patched accuracy: {result.patched_accuracy:.2%}")
print(f"Effect size: {result.effect_size:.4f}")
```

### Effect Size Interpretation

Effect size measures how much of the performance difference is recovered by patching:

```
effect_size = (patched_accuracy - corrupted_accuracy) /
              (clean_accuracy - corrupted_accuracy)
```

- **Effect size near 1.0**: Layer carries essential information
- **Effect size near 0.0**: Layer doesn't carry distinguishing information
- **Negative effect size**: Patching hurts performance (rare)

## Zero Ablation

Zero ablation is more aggressive than mean ablation - it completely removes layer contribution.

```python
result = patcher.zero_ablation(inputs, targets, 'blocks.0.hook_attn_out')
print(f"Zero ablation effect: accuracy_drop={result.accuracy_drop:.2%}")
```

## Comprehensive Analysis

Use `CausalAnalyzer` for automated analysis across all layers:

```python
analyzer = CausalAnalyzer(model, device=device)

# Analyze all layers
importance = analyzer.analyze_layer_importance(
    dataset,
    n_samples=100,
    ablation_type='mean'
)

# Rank layers by importance
sorted_layers = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nLayer Ranking by Importance:")
for i, (layer, score) in enumerate(sorted_layers, 1):
    print(f"{i}. {layer}: {score:.4f}")
```

## Verifying Concept Structure

To verify that circular topology in embeddings is causally meaningful:

```python
from analysis.fourier_analysis import FourierAnalyzer

# Get embeddings
embeddings = model.get_number_embeddings()
analyzer = FourierAnalyzer(p)

# Check circular structure
circular = analyzer.detect_circular_structure(embeddings)
print(f"Is circular: {circular['is_circular']}")
print(f"Distance correlation: {circular['distance_correlation']:.3f}")

# Verify causally: patch topologically close vs distant numbers
def patch_by_topology(patcher, inputs, targets, distance_threshold):
    """Patch activations between topologically close/distant inputs."""
    # Implementation depends on specific hypothesis
    pass

# If circular structure is causal:
# - Patching close numbers should have small effect
# - Patching distant numbers should have large effect
```

## Saving Results

```python
import json

# Collect all results
results = {
    'layer_importance': {k: float(v) for k, v in importance.items()},
    'ablation_type': 'mean',
    'n_samples': 100,
    'model_accuracy': float((model(inputs).argmax(-1) == targets).float().mean())
}

with open('causal_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Best Practices

1. **Use sufficient samples**: At least 100 examples for stable estimates
2. **Compare ablation types**: Mean and zero ablation can give different insights
3. **Report confidence intervals**: Run multiple times with different subsets
4. **Consider position**: For sequence models, patch at specific positions
5. **Combine with visualization**: Plot importance across layers

## Common Pitfalls

- **Batch effects**: Ensure batch size matches training
- **Correlated layers**: Adjacent layers may share information
- **Indirect effects**: A layer may be important through downstream effects
- **Distribution shift**: Ablated activations may be out-of-distribution

## Next Steps

- [Path Patching Tutorial](path-patching.md) - Trace information flow through circuits
- [Faithfulness Evaluation](faithfulness.md) - Measure concept fidelity
- [Circuit Discovery](circuit-discovery.md) - Find important components automatically
