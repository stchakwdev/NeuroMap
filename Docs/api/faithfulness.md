# Faithfulness API

Module for evaluating how faithfully extracted concepts represent model computation.

## Classes

### FaithfulnessEvaluator

Main class for faithfulness scoring.

```python
from analysis.faithfulness import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(model, device='cpu')
```

**Parameters:**
- `model` - HookedTransformer model instance
- `device` - Computation device

#### Methods

##### compute_faithfulness

```python
score = evaluator.compute_faithfulness(
    concept_extractor, dataset, layer_name, n_samples=100
)
```

Compute overall faithfulness score for extracted concepts.

**Parameters:**
- `concept_extractor` - Concept extraction method (SAE, clustering, etc.)
- `dataset` - ModularArithmeticDataset instance
- `layer_name` - Layer to analyze
- `n_samples` - Number of samples

**Returns:** `FaithfulnessScore`
- `faithfulness` - Overall score [0, 1]
- `completeness` - How much variance is captured
- `separability` - How well concepts separate
- `compactness` - How tight concept clusters are

##### concept_completeness

```python
completeness = evaluator.concept_completeness(concepts, activations)
```

Measure what fraction of activation variance is explained by concepts.

**Parameters:**
- `concepts` - Extracted concept representations [n_concepts, d]
- `activations` - Original activations [n_samples, d]

**Returns:** Float in [0, 1]

##### concept_separability

```python
separability = evaluator.concept_separability(concepts, labels)
```

Measure linear separability of concepts via SVM.

**Parameters:**
- `concepts` - Concept representations [n_samples, d]
- `labels` - Concept labels [n_samples]

**Returns:** Float in [0, 1] (classification accuracy)

---

### LinearRepresentationTester

Test if concepts have linear representations.

```python
from analysis.faithfulness import LinearRepresentationTester

tester = LinearRepresentationTester()
```

#### Methods

##### test_linearity

```python
result = tester.test_linearity(activations, labels)
```

Test if concept labels are linearly decodable from activations.

**Parameters:**
- `activations` - Activation tensor [n_samples, d]
- `labels` - Concept labels [n_samples]

**Returns:** Dictionary with:
- `accuracy` - Linear probe accuracy
- `is_linear` - Boolean (accuracy > 0.9)
- `probe_weights` - Trained probe weights

---

## Data Classes

### FaithfulnessScore

```python
@dataclass
class FaithfulnessScore:
    faithfulness: float  # Overall [0, 1]
    completeness: float  # Variance explained
    separability: float  # Linear separability
    compactness: float  # Cluster tightness
```

---

## Example Usage

```python
from analysis.faithfulness import FaithfulnessEvaluator
from analysis.concept_extractors import ClusteringExtractor

# Setup
evaluator = FaithfulnessEvaluator(model)
extractor = ClusteringExtractor(n_clusters=17)

# Evaluate faithfulness
score = evaluator.compute_faithfulness(
    extractor, dataset, 'blocks.0.hook_mlp_out'
)

print(f"Faithfulness: {score.faithfulness:.2%}")
print(f"Completeness: {score.completeness:.2%}")
print(f"Separability: {score.separability:.2%}")
```
