# Fourier Analysis Tutorial

This tutorial explains how to detect and analyze Fourier-based computation in neural networks trained on modular arithmetic.

## Background

### The Fourier Algorithm Hypothesis

When neural networks learn modular arithmetic `(a + b) mod p`, they may discover that the Fourier basis provides an efficient representation. The discrete Fourier transform on Z_p converts addition to element-wise multiplication of complex exponentials.

For modular addition:
```
(a + b) mod p
```

The Fourier approach encodes numbers as points on p-th roots of unity:
```
n -> exp(2*pi*i*k*n/p) for frequency k
```

Addition then becomes:
```
exp(2*pi*i*k*a/p) * exp(2*pi*i*k*b/p) = exp(2*pi*i*k*(a+b)/p)
```

### What to Look For

1. **Circular embeddings**: Numbers arranged in a circle in embedding space
2. **Dominant frequencies**: Specific Fourier components with high power
3. **Distance preservation**: Embedding distance correlates with circular distance

## Setup

```python
import torch
import numpy as np
from Dataset.dataset import ModularArithmeticDataset
from models.hooked_transformer import create_hooked_model
from analysis.fourier_analysis import (
    FourierAnalyzer,
    FourierBasis,
    generate_fourier_report
)

# Create dataset and model
p = 17
dataset = ModularArithmeticDataset(p=p)
model = create_hooked_model(vocab_size=p, n_layers=2)

# Assume model is trained
model.eval()
```

## Extracting Embeddings

```python
# Get number embeddings (exclude special tokens)
embeddings = model.get_number_embeddings()
print(f"Embedding shape: {embeddings.shape}")
# Expected: torch.Size([17, 64]) for p=17, d_model=64
```

## Creating the Fourier Basis

```python
# Create Fourier basis for Z_p
basis = FourierBasis(p)

# Inspect basis vectors
print(f"Basis frequencies: {list(range(p))}")
print(f"Basis shape: {basis.basis.shape}")

# Each row is a frequency, each column is a number
# Entry [k, n] = exp(2*pi*i*k*n/p)
```

## Extracting Fourier Components

```python
analyzer = FourierAnalyzer(p)

# Extract Fourier components from embeddings
components = analyzer.extract_fourier_components(embeddings)

print(f"Dominant frequency: k={components.dominant_frequency}")
print(f"Frequency powers: {components.frequency_powers[:5]}...")  # First 5
```

### Understanding Frequency Powers

```python
import matplotlib.pyplot as plt

# Plot frequency spectrum
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(p), components.frequency_powers)
ax.set_xlabel('Frequency k')
ax.set_ylabel('Power')
ax.set_title('Fourier Spectrum of Embeddings')
plt.savefig('fourier_spectrum.png')
```

Key observations:
- **DC component (k=0)**: Mean of embeddings, usually small
- **Low frequencies (k=1,2,...)**: Capture global structure
- **Symmetric frequencies**: k and p-k have same power (real embeddings)

## Measuring Fourier Alignment

```python
# Compute alignment score
alignment = analyzer.measure_fourier_alignment(embeddings)

print(f"Alignment score: {alignment.alignment_score:.2%}")
print(f"Key frequencies: {alignment.key_frequencies}")
print(f"Is Fourier-based: {alignment.is_fourier_based}")
```

### Interpretation

- **Alignment > 0.7**: Strong evidence of Fourier algorithm
- **Alignment 0.3-0.7**: Partial Fourier structure
- **Alignment < 0.3**: Different algorithm (e.g., lookup table)

## Detecting Circular Structure

The Fourier algorithm naturally produces circular embeddings:

```python
circular = analyzer.detect_circular_structure(embeddings)

print(f"Is circular: {circular['is_circular']}")
print(f"Distance correlation: {circular['distance_correlation']:.3f}")
print(f"Angular coherence: {circular['angular_coherence']:.3f}")
```

### Visualizing the Circle

```python
# Project to 2D using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings.detach().numpy())

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(p):
    ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100)
    ax.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]))

# Draw expected circle
theta = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(np.var(embeddings_2d[:, 0]) + np.var(embeddings_2d[:, 1]))
ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.3)

ax.set_aspect('equal')
ax.set_title('Number Embeddings (PCA projection)')
plt.savefig('embedding_circle.png')
```

## Generating Full Report

```python
# Generate comprehensive report
report = generate_fourier_report(analyzer, embeddings)
print(report)

# Save to file
with open('fourier_report.txt', 'w') as f:
    f.write(report)
```

## Comparing Trained vs Random

```python
# Random model for comparison
random_model = create_hooked_model(vocab_size=p)
random_embeddings = random_model.get_number_embeddings()

random_alignment = analyzer.measure_fourier_alignment(random_embeddings)
print(f"Random model alignment: {random_alignment.alignment_score:.2%}")
print(f"Trained model alignment: {alignment.alignment_score:.2%}")
```

Expected: Random models have low alignment (~0.1), trained models have high alignment (>0.7).

## Tracking During Training

```python
def analyze_training_dynamics(model, dataset, n_epochs=1000):
    """Track Fourier alignment during training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    analyzer = FourierAnalyzer(dataset.p)

    history = {'epoch': [], 'loss': [], 'accuracy': [], 'alignment': []}

    for epoch in range(n_epochs):
        # Training step
        optimizer.zero_grad()
        outputs = model(dataset.data['inputs'])
        loss = torch.nn.functional.cross_entropy(outputs, dataset.data['targets'])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                embeddings = model.get_number_embeddings()
                alignment = analyzer.measure_fourier_alignment(embeddings)
                accuracy = (outputs.argmax(-1) == dataset.data['targets']).float().mean()

                history['epoch'].append(epoch + 1)
                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy.item())
                history['alignment'].append(alignment.alignment_score)

                print(f"Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"accuracy={accuracy.item():.2%}, "
                      f"alignment={alignment.alignment_score:.2%}")

    return history
```

## Visualization Methods

```python
# Built-in visualization
fig = analyzer.visualize_fourier_spectrum(
    embeddings,
    save_path='spectrum.png'
)

fig = analyzer.visualize_embedding_circle(
    embeddings,
    save_path='circle.png'
)
```

## Best Practices

1. **Check multiple layers**: Fourier structure may emerge in different layers
2. **Use sufficient training**: Fourier algorithm often emerges during "grokking"
3. **Compare architectures**: Different models may use different algorithms
4. **Validate causally**: High alignment doesn't prove the algorithm is used

## Troubleshooting

### Low Alignment Despite Good Accuracy

- Model may use lookup table instead of algorithm
- Check if embeddings cluster rather than form circle
- Try longer training (grokking phenomenon)

### Noisy Frequency Spectrum

- May need more training
- Try different random seeds
- Check for optimization issues

## Next Steps

- [Causal Verification](causal-verification.md) - Verify Fourier structure is used
- [Circuit Discovery](circuit-discovery.md) - Find the Fourier circuit
- [API Reference](../api/fourier-analysis.md) - Detailed API documentation
