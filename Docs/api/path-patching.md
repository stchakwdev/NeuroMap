# Path Patching API

Module for tracing information flow through computational circuits.

## Classes

### PathPatcher

Main class for path-level causal interventions.

```python
from analysis.path_patching import PathPatcher

patcher = PathPatcher(model, device='cpu')
```

#### Methods

##### patch_path

```python
result = patcher.patch_path(
    sender_node, receiver_node, clean_input, corrupted_input, targets
)
```

Patch a specific path between two nodes.

**Parameters:**
- `sender_node` - Source node name (e.g., 'blocks.0.hook_attn_out')
- `receiver_node` - Destination node name
- `clean_input` - Clean input tensor
- `corrupted_input` - Corrupted input tensor
- `targets` - Target labels

**Returns:** `PathPatchingResult`

##### compute_path_importance

```python
importance = patcher.compute_path_importance(dataset, n_samples=100)
```

Compute importance of all paths in the model.

**Returns:** Dictionary mapping (sender, receiver) to importance score

---

### ComputationalGraph

Represents the model's computational graph.

```python
from analysis.path_patching import ComputationalGraph

graph = ComputationalGraph(model)
```

#### Methods

##### get_paths

```python
paths = graph.get_paths(source='embed', target='output')
```

Get all paths from source to target.

**Returns:** List of path tuples

---

## Example Usage

```python
from analysis.path_patching import PathPatcher

patcher = PathPatcher(model)

# Patch embed -> L0 attention path
result = patcher.patch_path(
    sender_node='hook_embed',
    receiver_node='blocks.0.hook_attn_out',
    clean_input=clean,
    corrupted_input=corrupted,
    targets=targets
)

print(f"Path importance: {result.importance:.4f}")
```
