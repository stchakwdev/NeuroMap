# Modular Arithmetic Dataset for Neural Network Interpretability

This dataset implementation creates complete training data for learning modular arithmetic operations, specifically designed for neural network interpretability research focusing on concept topology visualization.

## Overview

The dataset generates all possible input pairs `(a, b)` and their corresponding outputs `(a + b) mod p` for a chosen prime `p`. This creates a mathematically structured learning task where neural networks should learn circular representations of numbers 0 through p-1.

## Quick Start

```python
from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator

# Create dataset
dataset = ModularArithmeticDataset(p=17, representation='embedding')

# Basic information
print(f"Created {dataset.data['num_examples']} examples")
print(f"Input shape: {dataset.data['inputs'].shape}")

# Save for later use
dataset.save('data/mod_17_dataset.pkl')
```

## Files Description

- **`dataset.py`**: Core dataset creation with complete modular arithmetic examples and structural metadata
- **`validation.py`**: Functions to test if learned embeddings form expected circular structure  
- **`config.py`**: Configuration constants and model parameters
- **`utils.py`**: Helper functions for visualization and analysis
- **`test_dataset.py`**: Comprehensive unit tests
- **`README.md`**: This documentation

## Dataset Structure

### Core Data
- **inputs**: `(p², 2)` tensor of all `(a, b)` pairs 
- **targets**: `(p²,)` tensor of `(a + b) mod p` results
- **raw_inputs**: List of input tuples for human readability
- **raw_targets**: List of target values for human readability

### Structural Metadata
The dataset includes rich metadata about expected mathematical structure:

- **Circular structure**: Adjacent pairs, diameter pairs, expected angular spacing
- **Algebraic properties**: Commutativity, identity element, inverse pairs, complete addition table
- **Distance matrices**: Expected circular and Euclidean distances
- **Validation sets**: Specific test cases for model validation

## Why This Dataset?

### Perfect Ground Truth Structure
- Numbers 0 through p-1 should form a perfect circle in learned embeddings
- Adjacent numbers (n, n+1 mod p) should be consistently close
- Distances should correlate with circular distances on the mathematical circle

### Validation Advantages  
- Small scale (289 examples for p=17) enables exhaustive analysis
- Every property is testable and verifiable
- Clear success criteria for concept extraction methods

### Interpretability Research
- Tests if visualization methods can detect genuine mathematical structure
- Enables validation of concept extraction techniques
- Provides baseline for more complex interpretability tasks

## Usage Examples

### Basic Dataset Creation
```python
# Create dataset for mod 17 arithmetic
dataset = ModularArithmeticDataset(p=17, representation='embedding')

# Show some examples
for i in range(5):
    a, b = dataset.data['raw_inputs'][i]  
    result = dataset.data['raw_targets'][i]
    print(f"{a} + {b} = {result} (mod 17)")
```

### Structure Validation
```python
from validation import CircularStructureValidator

# Assuming you have learned embeddings from a model
validator = CircularStructureValidator(p=17)
results = validator.validate_embeddings(learned_embeddings)

print(f"Circular structure score: {results['overall_assessment']['overall_score']:.2f}")
print(f"Assessment: {results['overall_assessment']['quality_assessment']}")
```

### Visualization
```python
from utils import visualize_addition_table, visualize_circular_structure

# Show the mathematical structure  
visualize_addition_table(p=17)

# Compare learned vs expected structure
visualize_circular_structure(learned_embeddings, p=17)
```

## Testing

Run comprehensive unit tests:
```bash
python test_dataset.py
```

Tests cover:
- Mathematical correctness of all examples
- Completeness (all pairs included exactly once)
- Different representation formats
- Metadata generation
- Serialization/deserialization
- Structure validation with known embeddings

## Expected Model Behavior

A successfully trained model should:
1. **Achieve >99% accuracy** on all examples (task is fully learnable)
2. **Learn circular embeddings** where numbers form a circle when projected to 2D
3. **Maintain adjacency relationships** where consecutive numbers are consistently close
4. **Respect algebraic structure** where commutativity and identity properties hold

## Dataset Variants

### Different Primes
- **p=13**: 169 examples (smaller, faster training)
- **p=17**: 289 examples (recommended default)  
- **p=23**: 529 examples (larger scale testing)

### Different Representations
- **embedding**: Integer indices for learned embeddings (recommended)
- **one_hot**: One-hot vectors for each number
- **integer**: Direct integer values

## Integration with Model Training

This dataset is designed to work with:
- Small transformer models (2-layer, 4-head, 64-dim)
- Mamba/SSM models (2-layer, 64-dim)
- Any sequence-to-sequence or classification architecture

The key is extracting intermediate representations (embeddings) for structure validation, not just achieving high accuracy.

## Research Applications

This dataset enables research in:
- **Concept extraction**: Testing methods for identifying learned concepts
- **Topology visualization**: Validating graph construction from neural representations  
- **Interpretability methods**: Benchmarking visualization techniques against known structure
- **Architecture comparison**: Comparing how different models learn mathematical structure

## Installation and Dependencies

### Required Dependencies
```bash
pip install torch scikit-learn matplotlib numpy
```

### Optional Dependencies (for enhanced visualization)
```bash
pip install seaborn plotly
```

## API Reference

### ModularArithmeticDataset

Main class for creating modular arithmetic datasets.

```python
dataset = ModularArithmeticDataset(p=17, representation='embedding')
```

**Parameters:**
- `p` (int): Prime modulus (recommended: 17)
- `representation` (str): Input encoding ('embedding', 'one_hot', 'integer')

**Methods:**
- `save(filepath)`: Save dataset to pickle file
- `load(filepath)`: Load dataset from pickle file (class method)
- `export_metadata_json(filepath)`: Export metadata to JSON

### CircularStructureValidator

Validates whether learned embeddings form circular structure.

```python
validator = CircularStructureValidator(p=17)
results = validator.validate_embeddings(embeddings, visualize=True)
```

**Parameters:**
- `p` (int): Prime modulus matching the dataset
- `embeddings` (torch.Tensor): Learned embeddings of shape (p, embedding_dim)
- `visualize` (bool): Whether to create visualization plots

**Returns:**
- Dictionary with validation metrics and assessment

### Utility Functions

```python
from utils import (
    print_dataset_summary,
    check_dataset_consistency,
    analyze_algebraic_properties,
    create_training_splits,
    visualize_addition_table,
    visualize_circular_structure
)
```

## File Structure

```
modular_arithmetic_dataset/
├── config.py              # Configuration constants
├── dataset.py             # Core dataset creation
├── validation.py          # Structure validation
├── utils.py               # Helper utilities
├── test_dataset.py        # Unit tests
├── README.md              # This documentation
├── data/                  # Generated datasets
│   ├── mod_17_dataset.pkl
│   └── mod_17_metadata.json
├── results/               # Analysis results
└── models/                # Trained models (future)
```

## Examples and Tutorials

### Example 1: Basic Dataset Creation and Analysis

```python
from dataset import ModularArithmeticDataset
from utils import print_dataset_summary, check_dataset_consistency

# Create dataset
dataset = ModularArithmeticDataset(p=17)

# Print summary
print_dataset_summary(dataset)

# Run consistency checks
checks = check_dataset_consistency(dataset)
print("All checks passed:", all(checks.values()))
```

### Example 2: Model Training Integration

```python
import torch
import torch.nn as nn
from dataset import ModularArithmeticDataset
from utils import create_training_splits

# Create dataset and splits
dataset = ModularArithmeticDataset(p=17)
splits = create_training_splits(dataset, train_ratio=0.8)

# Simple model example
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4), 
            num_layers=2
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x shape: (batch, 2) - pairs of numbers
        embedded = self.embedding(x)  # (batch, 2, d_model)
        transformed = self.transformer(embedded.transpose(0, 1))  # (2, batch, d_model)
        output = self.output(transformed.mean(0))  # (batch, vocab_size)
        return output

# Training loop (simplified)
model = SimpleTransformer(vocab_size=17)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Train on the dataset...
```

### Example 3: Structure Validation

```python
from validation import CircularStructureValidator
import torch

# Assume you have trained a model and extracted embeddings
# embeddings = model.embedding.weight.data  # Shape: (17, d_model)

# For demonstration, create test embeddings
p = 17
angles = torch.linspace(0, 2 * torch.pi, p + 1)[:-1]
test_embeddings = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

# Validate structure
validator = CircularStructureValidator(p)
results = validator.validate_embeddings(test_embeddings)

print(f"Overall score: {results['overall_assessment']['overall_score']:.2f}")
print(f"Quality: {results['overall_assessment']['quality_assessment']}")
print("Recommendations:")
for rec in results['overall_assessment']['recommendations']:
    print(f"  - {rec}")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Use smaller p values for testing (p=5 or p=7)
3. **Visualization not showing**: Install matplotlib and ensure display is available
4. **Test failures**: Run `python test_dataset.py` to identify specific issues

### Performance Tips

1. **Use p=17 for development**: Good balance of complexity and speed
2. **Cache datasets**: Save/load datasets to avoid regeneration
3. **Batch processing**: Use DataLoader for model training
4. **GPU acceleration**: Move tensors to GPU for faster computation

## Contributing

This implementation is designed to be modular and extensible. To contribute:

1. Run all tests: `python test_dataset.py`
2. Follow the existing code style and documentation patterns
3. Add tests for new functionality
4. Update documentation as needed

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{modular_arithmetic_dataset,
  title={Modular Arithmetic Dataset for Neural Network Interpretability},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/modular-arithmetic-dataset}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contact and Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation and examples

This dataset is part of ongoing research in neural network interpretability and concept topology visualization.

