# Fourier Analysis API

Module for detecting Fourier-based computation in neural network representations.

## Classes

### FourierBasis

Discrete Fourier basis for modular arithmetic.

```python
from analysis.fourier_analysis import FourierBasis

basis = FourierBasis(p=17)
```

**Parameters:**
- `p` - Prime modulus

**Attributes:**
- `p` - The modulus
- `basis` - Complex tensor [p, p] where basis[k, n] = exp(2*pi*i*k*n/p)

#### Methods

##### project

```python
coefficients = basis.project(embeddings)
```

Project embeddings onto Fourier basis.

**Parameters:**
- `embeddings` - Tensor [p, d_model]

**Returns:** Complex tensor [p, d_model] of Fourier coefficients

##### reconstruct

```python
reconstructed = basis.reconstruct(coefficients)
```

Reconstruct embeddings from Fourier coefficients.

**Parameters:**
- `coefficients` - Complex tensor [p, d_model]

**Returns:** Tensor [p, d_model]

---

### FourierAnalyzer

Main class for Fourier structure analysis.

```python
from analysis.fourier_analysis import FourierAnalyzer

analyzer = FourierAnalyzer(p=17)
```

**Parameters:**
- `p` - Prime modulus

#### Methods

##### extract_fourier_components

```python
components = analyzer.extract_fourier_components(embeddings)
```

Extract Fourier frequency components from embeddings.

**Parameters:**
- `embeddings` - Tensor [p, d_model]

**Returns:** `FourierComponents`
- `coefficients` - Complex Fourier coefficients [p, d_model]
- `frequency_powers` - Power at each frequency [p]
- `dominant_frequency` - Index of highest-power frequency
- `total_power` - Sum of all frequency powers

##### measure_fourier_alignment

```python
alignment = analyzer.measure_fourier_alignment(embeddings)
```

Measure how well embeddings align with Fourier representation.

**Parameters:**
- `embeddings` - Tensor [p, d_model]

**Returns:** `FourierAlignment`
- `alignment_score` - Float in [0, 1], higher = more Fourier-like
- `key_frequencies` - List of important frequency indices
- `is_fourier_based` - Boolean, True if alignment > 0.7
- `reconstruction_error` - Error when using top frequencies

##### detect_circular_structure

```python
circular = analyzer.detect_circular_structure(embeddings)
```

Detect if embeddings form a circular arrangement.

**Parameters:**
- `embeddings` - Tensor [p, d_model]

**Returns:** Dictionary with:
- `is_circular` - Boolean
- `distance_correlation` - Correlation between embedding and circular distance
- `angular_coherence` - How consistently ordered the embeddings are
- `fitted_circle_params` - Parameters of fitted circle (if circular)

##### visualize_fourier_spectrum

```python
fig = analyzer.visualize_fourier_spectrum(embeddings, save_path=None)
```

Create bar plot of frequency powers.

**Parameters:**
- `embeddings` - Tensor [p, d_model]
- `save_path` - Optional path to save figure

**Returns:** Matplotlib figure

##### visualize_embedding_circle

```python
fig = analyzer.visualize_embedding_circle(embeddings, save_path=None)
```

Visualize embeddings projected to 2D with circular overlay.

**Parameters:**
- `embeddings` - Tensor [p, d_model]
- `save_path` - Optional path to save figure

**Returns:** Matplotlib figure

---

## Data Classes

### FourierComponents

```python
@dataclass
class FourierComponents:
    coefficients: torch.Tensor  # Complex [p, d_model]
    frequency_powers: torch.Tensor  # [p]
    dominant_frequency: int
    total_power: float
```

### FourierAlignment

```python
@dataclass
class FourierAlignment:
    alignment_score: float  # [0, 1]
    key_frequencies: List[int]
    is_fourier_based: bool
    reconstruction_error: float
```

---

## Helper Functions

### generate_fourier_report

```python
from analysis.fourier_analysis import generate_fourier_report

report = generate_fourier_report(analyzer, embeddings)
```

Generate comprehensive text report of Fourier analysis.

**Parameters:**
- `analyzer` - FourierAnalyzer instance
- `embeddings` - Tensor [p, d_model]

**Returns:** String report

---

## Mathematical Background

### Discrete Fourier Transform on Z_p

For a prime p, the DFT basis vectors are:

```
f_k[n] = exp(2*pi*i*k*n/p)
```

where k is the frequency (0 to p-1) and n is the number (0 to p-1).

### Why Fourier Works for Modular Arithmetic

Addition mod p corresponds to convolution in the spatial domain. In the Fourier domain, convolution becomes multiplication:

```
DFT(a + b mod p) = DFT(a) * DFT(b)
```

This allows the network to compute modular addition via:
1. Encode numbers using Fourier basis
2. Multiply Fourier coefficients
3. Decode back to output

### Alignment Score Interpretation

| Score | Interpretation |
|-------|----------------|
| 0.0 - 0.3 | No Fourier structure (lookup table or other algorithm) |
| 0.3 - 0.5 | Weak Fourier structure |
| 0.5 - 0.7 | Moderate Fourier structure |
| 0.7 - 1.0 | Strong Fourier structure (likely using Fourier algorithm) |

---

## Example Usage

```python
from models.hooked_transformer import create_hooked_model
from analysis.fourier_analysis import FourierAnalyzer, generate_fourier_report

# Create and train model
p = 17
model = create_hooked_model(vocab_size=p)
# ... training ...

# Extract embeddings
embeddings = model.get_number_embeddings()

# Analyze
analyzer = FourierAnalyzer(p)

# Get components
components = analyzer.extract_fourier_components(embeddings)
print(f"Dominant frequency: k={components.dominant_frequency}")
print(f"Top 3 frequency powers: {components.frequency_powers[:3]}")

# Check alignment
alignment = analyzer.measure_fourier_alignment(embeddings)
print(f"Alignment: {alignment.alignment_score:.2%}")
print(f"Is Fourier-based: {alignment.is_fourier_based}")

# Check circular structure
circular = analyzer.detect_circular_structure(embeddings)
print(f"Is circular: {circular['is_circular']}")
print(f"Distance correlation: {circular['distance_correlation']:.3f}")

# Visualize
analyzer.visualize_fourier_spectrum(embeddings, save_path='spectrum.png')
analyzer.visualize_embedding_circle(embeddings, save_path='circle.png')

# Full report
report = generate_fourier_report(analyzer, embeddings)
print(report)
```
