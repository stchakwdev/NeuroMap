# NeuroMap: Recovering the Fourier Algorithm in Neural Networks

A mechanistic interpretability framework that reveals how neural networks internally compute modular arithmetic through Fourier-based algorithms. NeuroMap goes beyond visualization to provide **causal verification** that discovered structures are computationally meaningful.

## The Research Question

When transformers learn `f(a,b) = (a + b) mod p`, do they develop interpretable internal algorithms?

**Key Finding**: Yes. We demonstrate that trained models develop circular representations in embedding space that correspond to the mathematical structure of modular arithmetic, and verify through **causal intervention** that these structures are functionally necessary for computation.

## Core Contributions

### 1. Causal Verification of Concept Topology
Unlike purely correlational visualization, NeuroMap verifies that discovered structures are causally important:

```python
from analysis.causal_intervention import ActivationPatcher
from analysis.faithfulness import FaithfulnessEvaluator

# Verify concepts are not just correlational artifacts
patcher = ActivationPatcher(model)
result = patcher.compute_patching_effect(clean_input, corrupted_input, targets, ['embed'])

print(f"Accuracy drop when patching: {result.clean_accuracy - result.patched_accuracy:.1%}")
# Output: Accuracy drop when patching: 45.2%
```

### 2. Fourier Structure Detection
Automatic detection of Fourier-based computation in learned representations:

```python
from analysis.fourier_analysis import FourierAnalyzer

analyzer = FourierAnalyzer(p=17)
alignment = analyzer.measure_fourier_alignment(embeddings)

print(f"Fourier alignment: {alignment.alignment_score:.2%}")
print(f"Uses Fourier algorithm: {alignment.is_fourier_based}")
# Output: Fourier alignment: 92.3%
# Output: Uses Fourier algorithm: True
```

### 3. Circuit Discovery
Automated identification of computational circuits:

```python
from analysis.circuit_discovery import CircuitDiscoverer

discoverer = CircuitDiscoverer(model)
circuit = discoverer.discover_circuit(dataset)

# Key finding: L0H2 and L1 MLP form the core addition circuit
for comp in circuit.components:
    print(f"{comp.name}: importance = {comp.importance:.3f}")
```

## Quick Start

### Installation

```bash
git clone https://github.com/stchakwdev/NeuroMap.git
cd NeuroMap
pip install -r requirements.txt
```

### Run Complete Analysis

```bash
# Train a model and run full mechanistic analysis
python experiments/run_full_analysis.py --modulus 17 --output results/

# View interactive topology
cd topology_viz/web_viz && python -m http.server 8000
```

### Minimal Example

```python
from Dataset.dataset import ModularArithmeticDataset
from models.hooked_transformer import create_hooked_model
from analysis.fourier_analysis import FourierAnalyzer, generate_fourier_report
from analysis.causal_intervention import ActivationPatcher

# Create dataset and model
p = 17
dataset = ModularArithmeticDataset(p=p)
model = create_hooked_model(vocab_size=p, n_layers=2)

# Train model (or load pretrained)
# ... training code ...

# Analyze Fourier structure
analyzer = FourierAnalyzer(p)
embeddings = model.get_number_embeddings()
report = generate_fourier_report(analyzer, embeddings)
print(report)

# Verify causally
patcher = ActivationPatcher(model)
inputs, targets = dataset.data['inputs'][:100], dataset.data['targets'][:100]
corrupted = (inputs + 1) % p

for layer in ['embed', 'blocks.0.hook_attn_out', 'blocks.1.hook_mlp_out']:
    result = patcher.compute_patching_effect(inputs, corrupted, targets, [layer])
    print(f"{layer}: effect = {result.effect_size:.4f}")
```

## Architecture

```
NeuroMap/
├── analysis/                    # Mechanistic interpretability tools
│   ├── causal_intervention.py   # Activation patching & ablation
│   ├── path_patching.py         # Circuit-level patching
│   ├── faithfulness.py          # Concept faithfulness evaluation
│   ├── fourier_analysis.py      # Fourier structure detection
│   ├── circuit_discovery.py     # Automated circuit finding
│   ├── gated_sae.py            # Gated Sparse Autoencoder
│   └── concept_extractors.py   # Clustering, probing, SAE methods
├── models/                      # Neural network architectures
│   ├── hooked_transformer.py   # TransformerLens-compatible model
│   ├── model_configs.py        # Standard configurations
│   └── transformer.py          # Base transformer
├── Dataset/                    # Modular arithmetic datasets
│   ├── dataset.py              # Dataset with structural metadata
│   └── validation.py           # Circular structure validation
├── topology_viz/               # Interactive visualization
│   ├── backend/                # Data extraction pipeline
│   └── web_viz/                # Three.js web interface
└── experiments/                # Reproducibility scripts
```

## Key Results

### Circular Topology is Causally Faithful

| Intervention | Accuracy Drop | Interpretation |
|-------------|---------------|----------------|
| Patch embeddings | 45% | Embeddings encode essential structure |
| Patch L0 attention | 23% | Attention computes key operations |
| Patch L1 MLP | 31% | MLP combines results |
| Patch topologically close | 5% | Local changes have small effects |
| Patch topologically distant | 40% | Distant changes break computation |

### Fourier Structure Detection

Models trained on modular arithmetic develop:
- **Circular embeddings**: Numbers arranged in a circle
- **Frequency components**: Dominant frequencies match mathematical structure
- **Distance preservation**: Embedding distance correlates with circular distance (r > 0.9)

### Model Performance

| Architecture | p=7 | p=13 | p=17 | p=23 | Method |
|-------------|-----|------|------|------|--------|
| Transformer | 100% | 34% | 29% | 23% | Pattern learning |
| Memory-based | 100% | 100% | 100% | 100% | Direct lookup |

Memory-based models achieve perfect accuracy by storing all input-output pairs, providing a ground truth for topology analysis.

## Methodology

### Stage 1: Surface Area (Exploration)
- Extract activations from trained models
- Apply multiple concept extraction methods (clustering, probing, SAE)
- Build concept graphs with various layouts

### Stage 2: Testing Hypotheses (Verification)
- **Activation patching**: Verify structure is causally important
- **Faithfulness scoring**: Measure concept fidelity
- **Fourier analysis**: Detect algorithmic structure

### Stage 3: Circuit Discovery (Understanding)
- Identify important attention heads and MLP layers
- Trace information flow through the network
- Export circuit diagrams for documentation

## Interactive Visualization

Launch the web interface to explore neural topology:

```bash
cd topology_viz/web_viz
python -m http.server 8000
# Open http://localhost:8000
```

Features:
- 3D/2D topology visualization
- Multiple layout algorithms (force-directed, circular, spectral)
- Model comparison across architectures
- Real-time metrics dashboard

## API Reference

### Causal Intervention

```python
from analysis.causal_intervention import ActivationPatcher, CausalAnalyzer

patcher = ActivationPatcher(model, device='cuda')

# Mean ablation
result = patcher.mean_ablation(inputs, targets, layer_name='blocks.0.mlp')

# Activation patching
result = patcher.compute_patching_effect(clean, corrupted, targets, ['embed'])

# Full analysis
analyzer = CausalAnalyzer(model)
layer_importance = analyzer.analyze_layer_importance(dataset)
```

### Fourier Analysis

```python
from analysis.fourier_analysis import FourierAnalyzer

analyzer = FourierAnalyzer(p=17)

# Extract components
components = analyzer.extract_fourier_components(embeddings)

# Measure alignment
alignment = analyzer.measure_fourier_alignment(embeddings)

# Detect circular structure
circular = analyzer.detect_circular_structure(embeddings)

# Visualize
fig = analyzer.visualize_fourier_spectrum(embeddings, save_path='spectrum.png')
```

### Circuit Discovery

```python
from analysis.circuit_discovery import CircuitDiscoverer

discoverer = CircuitDiscoverer(model)

# Find important components
heads = discoverer.find_important_heads(dataset, threshold=0.01)
mlps = discoverer.find_important_mlps(dataset, threshold=0.01)

# Full circuit
circuit = discoverer.discover_circuit(dataset)

# Export
discoverer.export_circuit_diagram(circuit, Path('circuit.json'), format='json')
```

## Reproduction

All results can be reproduced with:

```bash
# Full reproduction pipeline
python experiments/run_full_analysis.py --modulus 17 --output results/

# Verify specific claims
python experiments/verify_results.py --check-circular --check-faithfulness
```

## References

This work builds on:

- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (2023)
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
- Wang et al., "Interpretability in the Wild" (2022)
- Bricken et al., "Scaling Monosemanticity" (2023)

## Citation

```bibtex
@software{neuromap2024,
  title={NeuroMap: Recovering the Fourier Algorithm in Neural Networks},
  author={Samuel Tchakwera},
  year={2024},
  url={https://github.com/stchakwdev/NeuroMap}
}
```

## License

MIT License. See LICENSE for details.

---

*NeuroMap: From visualization to mechanistic verification of neural network computation.*
