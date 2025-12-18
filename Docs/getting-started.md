# Getting Started

This guide walks you through installing NeuroMap and running your first mechanistic interpretability analysis.

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Git

### Install from Source

```bash
git clone https://github.com/stchakwdev/NeuroMap.git
cd NeuroMap
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
from models.hooked_transformer import create_hooked_model
from analysis.fourier_analysis import FourierAnalyzer

# Create a test model
model = create_hooked_model(vocab_size=17)
print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

# Create analyzer
analyzer = FourierAnalyzer(p=17)
print("Installation verified successfully!")
```

## Quick Start

### 1. Create a Dataset

```python
from Dataset.dataset import ModularArithmeticDataset

# Create modular arithmetic dataset
p = 17  # Working with mod 17
dataset = ModularArithmeticDataset(p=p)

print(f"Dataset size: {dataset.data['num_examples']} examples")
print(f"Task: (a + b) mod {p}")
```

### 2. Train a Model

```python
import torch
from models.hooked_transformer import create_hooked_model

# Create model
model = create_hooked_model(vocab_size=p, n_layers=2, d_model=64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
inputs = dataset.data['inputs'].to(device)
targets = dataset.data['targets'].to(device)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        accuracy = (outputs.argmax(-1) == targets).float().mean()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, accuracy={accuracy.item():.2%}")
```

### 3. Analyze Fourier Structure

```python
from analysis.fourier_analysis import FourierAnalyzer, generate_fourier_report

# Extract embeddings
embeddings = model.get_number_embeddings()
print(f"Embedding shape: {embeddings.shape}")

# Analyze Fourier components
analyzer = FourierAnalyzer(p)
components = analyzer.extract_fourier_components(embeddings)
alignment = analyzer.measure_fourier_alignment(embeddings)

print(f"Dominant frequency: k={components.dominant_frequency}")
print(f"Fourier alignment: {alignment.alignment_score:.2%}")
print(f"Uses Fourier algorithm: {alignment.is_fourier_based}")

# Generate full report
report = generate_fourier_report(analyzer, embeddings)
print(report)
```

### 4. Verify Causally

```python
from analysis.causal_intervention import ActivationPatcher

# Create patcher
patcher = ActivationPatcher(model, device=device)

# Test layer importance via mean ablation
inputs = dataset.data['inputs'][:100].to(device)
targets = dataset.data['targets'][:100].to(device)

for layer_name in ['embed', 'blocks.0.hook_attn_out', 'blocks.1.hook_mlp_out']:
    result = patcher.mean_ablation(inputs, targets, layer_name)
    print(f"{layer_name}: accuracy_drop={result.accuracy_drop:.2%}")
```

### 5. Discover Circuits

```python
from analysis.circuit_discovery import CircuitDiscoverer

discoverer = CircuitDiscoverer(model, device=device)
circuit = discoverer.discover_circuit(dataset, n_samples=50)

print(f"Found {len(circuit.components)} important components:")
for comp in circuit.components:
    print(f"  {comp.name}: importance={comp.importance:.3f}")
```

## Run Complete Analysis

For a complete analysis pipeline, use the experiment script:

```bash
python experiments/run_full_analysis.py --modulus 17 --output results/
```

This will:
1. Create the dataset
2. Train or load a model
3. Extract embeddings and activations
4. Analyze Fourier structure
5. Run causal interventions
6. Discover circuits
7. Generate visualizations and reports

## Interactive Visualization

Launch the web interface to explore neural topology:

```bash
cd topology_viz/web_viz
python -m http.server 8000
# Open http://localhost:8000
```

## Next Steps

- Read the [Causal Verification Tutorial](tutorials/causal-verification.md) for in-depth intervention analysis
- Explore the [Fourier Analysis Tutorial](tutorials/fourier-analysis.md) for algorithmic structure detection
- Check the [API Reference](api/causal-intervention.md) for detailed documentation
