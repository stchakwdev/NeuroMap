# NeuroMap Documentation

A mechanistic interpretability framework for recovering algorithmic structure in neural networks.

## Overview

NeuroMap provides tools for:

1. **Causal Verification** - Verify that discovered structures are computationally meaningful through activation patching and ablation studies
2. **Fourier Analysis** - Detect Fourier-based computation in learned representations
3. **Circuit Discovery** - Automatically identify important computational circuits
4. **Faithfulness Evaluation** - Measure how well extracted concepts represent model computation

## Documentation Structure

### Getting Started
- [Installation and Quickstart](getting-started.md) - Set up NeuroMap and run your first analysis

### Tutorials
- [Causal Verification Tutorial](tutorials/causal-verification.md) - Step-by-step guide to verifying concept structures
- [Fourier Analysis Tutorial](tutorials/fourier-analysis.md) - Detecting algorithmic structure in embeddings
- [Circuit Discovery Tutorial](tutorials/circuit-discovery.md) - Finding important model components

### API Reference
- [Causal Intervention API](api/causal-intervention.md) - Activation patching and ablation
- [Path Patching API](api/path-patching.md) - Circuit-level causal tracing
- [Faithfulness API](api/faithfulness.md) - Concept faithfulness evaluation
- [Fourier Analysis API](api/fourier-analysis.md) - Fourier structure detection
- [Circuit Discovery API](api/circuit-discovery.md) - Automated circuit finding
- [Gated SAE API](api/gated-sae.md) - Sparse autoencoder for feature extraction
- [Hooked Transformer API](api/hooked-transformer.md) - TransformerLens-compatible model

### Research
- [Methodology](research/methodology.md) - Research approach and validation strategy
- [Results](research/results.md) - Key findings from modular arithmetic experiments

## Quick Example

```python
from Dataset.dataset import ModularArithmeticDataset
from models.hooked_transformer import create_hooked_model
from analysis.fourier_analysis import FourierAnalyzer
from analysis.causal_intervention import ActivationPatcher

# Create dataset and model
p = 17
dataset = ModularArithmeticDataset(p=p)
model = create_hooked_model(vocab_size=p)

# Analyze Fourier structure
analyzer = FourierAnalyzer(p)
embeddings = model.get_number_embeddings()
alignment = analyzer.measure_fourier_alignment(embeddings)
print(f"Fourier alignment: {alignment.alignment_score:.2%}")

# Verify causally
patcher = ActivationPatcher(model)
inputs = dataset.data['inputs'][:100]
targets = dataset.data['targets'][:100]
result = patcher.mean_ablation(inputs, targets, 'embed')
print(f"Accuracy drop: {result.accuracy_drop:.2%}")
```

## Citation

```bibtex
@software{neuromap2024,
  title={NeuroMap: Recovering the Fourier Algorithm in Neural Networks},
  author={Samuel T. Chakwera},
  year={2024},
  url={https://github.com/stchakwdev/NeuroMap}
}
```
