# Gated SAE API

Module for Gated Sparse Autoencoders following Anthropic's approach.

## Classes

### GatedSAE

Gated Sparse Autoencoder with gating mechanism to avoid shrinkage.

```python
from analysis.gated_sae import GatedSAE

sae = GatedSAE(
    d_input=64,
    d_hidden=256,
    sparsity_coefficient=1e-3
)
```

**Parameters:**
- `d_input` - Input dimension (model's d_model)
- `d_hidden` - Number of SAE features (typically 4x d_input)
- `sparsity_coefficient` - L1 penalty weight

#### Methods

##### forward

```python
reconstructed, feature_acts, aux_loss = sae(activations)
```

**Parameters:**
- `activations` - Input activations [batch, d_input]

**Returns:**
- `reconstructed` - Reconstructed activations [batch, d_input]
- `feature_acts` - Feature activations [batch, d_hidden]
- `aux_loss` - Auxiliary loss for training

##### encode

```python
feature_acts = sae.encode(activations)
```

Get feature activations without reconstruction.

##### decode

```python
reconstructed = sae.decode(feature_acts)
```

Reconstruct from feature activations.

---

### SAETrainer

Training utilities for SAE.

```python
from analysis.gated_sae import SAETrainer

trainer = SAETrainer(sae, learning_rate=1e-4)
```

#### Methods

##### train_step

```python
loss = trainer.train_step(activations)
```

Single training step.

##### train

```python
history = trainer.train(activation_dataset, n_epochs=100)
```

Full training loop.

---

### SAEFeatureAnalyzer

Analyze learned SAE features.

```python
from analysis.gated_sae import SAEFeatureAnalyzer

analyzer = SAEFeatureAnalyzer(sae, model)
```

#### Methods

##### get_top_activating_examples

```python
examples = analyzer.get_top_activating_examples(
    feature_idx, dataset, k=10
)
```

Find inputs that most activate a feature.

##### compute_dead_feature_ratio

```python
ratio = analyzer.compute_dead_feature_ratio(activation_dataset)
```

Compute fraction of features that never activate.

---

## Example Usage

```python
from analysis.gated_sae import GatedSAE, SAETrainer

# Create SAE
sae = GatedSAE(d_input=64, d_hidden=256)

# Collect activations from model
model.eval()
with torch.no_grad():
    _, cache = model.run_with_cache(dataset.data['inputs'])
    activations = cache['blocks.0.hook_mlp_out'][:, -1, :]  # Last position

# Train SAE
trainer = SAETrainer(sae)
history = trainer.train(activations, n_epochs=100)

# Analyze features
analyzer = SAEFeatureAnalyzer(sae, model)
for i in range(5):
    examples = analyzer.get_top_activating_examples(i, dataset, k=5)
    print(f"Feature {i}: top examples = {examples}")
```
