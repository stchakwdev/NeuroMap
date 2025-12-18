"""
Causal Intervention Module for NeuroMap.

Implements activation patching, ablation studies, and causal analysis
following mechanistic interpretability best practices (Neel Nanda methodology).

This module shifts NeuroMap from visualization to mechanistic verification
by enabling causal claims about concept importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import pickle


@dataclass
class PatchingResult:
    """Result of an activation patching experiment."""
    clean_logits: torch.Tensor
    patched_logits: torch.Tensor
    clean_loss: float
    patched_loss: float
    clean_accuracy: float
    patched_accuracy: float
    effect_size: float  # Normalized effect: (clean_loss - patched_loss) / clean_loss
    layer_name: str
    patch_type: str


@dataclass
class AblationResult:
    """Result of an ablation study."""
    original_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    original_loss: float
    ablated_loss: float
    loss_increase: float
    ablation_type: str
    target: str  # What was ablated (layer name, head index, etc.)


class ActivationCache:
    """
    Cache for storing activations during forward passes.

    Provides TransformerLens-style activation caching with hook management.
    """

    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def store(self, name: str, activation: torch.Tensor):
        """Store an activation in the cache."""
        self.cache[name] = activation.detach().clone()

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve an activation from the cache."""
        return self.cache.get(name)

    def clear(self):
        """Clear the cache and remove all hooks."""
        self.cache.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.cache[name]

    def __contains__(self, name: str) -> bool:
        return name in self.cache

    def keys(self):
        return self.cache.keys()


class ActivationPatcher:
    """
    Perform causal interventions on model activations.

    Implements activation patching (resampling ablation) to verify the
    functional importance of identified concepts and circuit components.

    Key operations:
    1. Clean run: model(clean_input) -> clean_activations, clean_output
    2. Corrupted run: model(corrupted_input) -> corrupted_activations
    3. Patched run: Replace clean_activations[layer] with corrupted_activations[layer]
    4. Measure: effect = change in model behavior (logits, loss, accuracy)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Cache for storing activations
        self.clean_cache = ActivationCache()
        self.corrupted_cache = ActivationCache()

        # Track registered hook points
        self.hook_points = self._discover_hook_points()

    def _discover_hook_points(self) -> Dict[str, nn.Module]:
        """Discover available hook points in the model."""
        hook_points = {}

        for name, module in self.model.named_modules():
            # Skip the root module
            if name == '':
                continue

            # Include key layer types
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm,
                                   nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                hook_points[name] = module

            # Also include any module ending with common patterns
            if any(pattern in name.lower() for pattern in
                   ['embed', 'attn', 'mlp', 'norm', 'output', 'residual']):
                hook_points[name] = module

        return hook_points

    def _create_cache_hook(self, cache: ActivationCache, name: str) -> Callable:
        """Create a forward hook that stores activations in a cache."""
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            cache.store(name, output)
        return hook_fn

    def _create_patch_hook(self,
                          corrupted_cache: ActivationCache,
                          name: str,
                          patch_mask: Optional[torch.Tensor] = None) -> Callable:
        """Create a forward hook that patches activations."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                original_output = output[0]
                rest = output[1:]
            else:
                original_output = output
                rest = None

            # Get corrupted activation
            corrupted_act = corrupted_cache.get(name)
            if corrupted_act is None:
                return output

            # Ensure shapes match
            if original_output.shape != corrupted_act.shape:
                # Try to broadcast
                corrupted_act = corrupted_act.to(original_output.device)

            # Apply patch
            if patch_mask is not None:
                # Partial patching using mask
                patched = original_output.clone()
                patched[patch_mask] = corrupted_act[patch_mask]
            else:
                # Full patching
                patched = corrupted_act.to(original_output.device)

            if rest is not None:
                return (patched,) + rest
            return patched

        return hook_fn

    def run_with_cache(self,
                      inputs: torch.Tensor,
                      layer_names: Optional[List[str]] = None) -> Tuple[torch.Tensor, ActivationCache]:
        """
        Run model forward pass while caching activations.

        Args:
            inputs: Input tensor
            layer_names: Layers to cache (None = cache all discovered hook points)

        Returns:
            Tuple of (model output, activation cache)
        """
        cache = ActivationCache()

        if layer_names is None:
            layer_names = list(self.hook_points.keys())

        # Register caching hooks
        for name in layer_names:
            if name in self.hook_points:
                module = self.hook_points[name]
                hook = module.register_forward_hook(self._create_cache_hook(cache, name))
                cache.hooks.append(hook)

        # Run forward pass
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        # Remove hooks (but keep cached values)
        for hook in cache.hooks:
            hook.remove()
        cache.hooks.clear()

        return outputs, cache

    def patch_activations(self,
                         clean_input: torch.Tensor,
                         corrupted_input: torch.Tensor,
                         patch_layers: List[str],
                         patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run forward pass with patched activations.

        Args:
            clean_input: The input we want to measure performance on
            corrupted_input: The input used to generate corrupted activations
            patch_layers: List of layer names to patch
            patch_mask: Optional boolean mask for partial patching

        Returns:
            Model output with patched activations
        """
        clean_input = clean_input.to(self.device)
        corrupted_input = corrupted_input.to(self.device)

        # Get corrupted activations
        _, corrupted_cache = self.run_with_cache(corrupted_input, patch_layers)

        # Register patching hooks
        patch_hooks = []
        for layer_name in patch_layers:
            if layer_name in self.hook_points:
                module = self.hook_points[layer_name]
                hook = module.register_forward_hook(
                    self._create_patch_hook(corrupted_cache, layer_name, patch_mask)
                )
                patch_hooks.append(hook)

        # Run forward pass with patches
        with torch.no_grad():
            patched_output = self.model(clean_input)

        # Clean up
        for hook in patch_hooks:
            hook.remove()
        corrupted_cache.clear()

        return patched_output

    def compute_patching_effect(self,
                               clean_input: torch.Tensor,
                               corrupted_input: torch.Tensor,
                               target: torch.Tensor,
                               patch_layers: List[str]) -> PatchingResult:
        """
        Measure the effect of activation patching.

        Args:
            clean_input: Clean input tensor
            corrupted_input: Corrupted input tensor
            target: Target labels for accuracy/loss computation
            patch_layers: Layers to patch

        Returns:
            PatchingResult with detailed metrics
        """
        clean_input = clean_input.to(self.device)
        corrupted_input = corrupted_input.to(self.device)
        target = target.to(self.device)

        # Get clean outputs
        with torch.no_grad():
            clean_logits = self.model(clean_input)

        # Get patched outputs
        patched_logits = self.patch_activations(
            clean_input, corrupted_input, patch_layers
        )

        # Compute metrics
        clean_loss = F.cross_entropy(clean_logits, target).item()
        patched_loss = F.cross_entropy(patched_logits, target).item()

        clean_preds = clean_logits.argmax(dim=-1)
        patched_preds = patched_logits.argmax(dim=-1)

        clean_accuracy = (clean_preds == target).float().mean().item()
        patched_accuracy = (patched_preds == target).float().mean().item()

        # Effect size: how much does patching change the loss
        if clean_loss > 0:
            effect_size = (patched_loss - clean_loss) / clean_loss
        else:
            effect_size = patched_loss

        return PatchingResult(
            clean_logits=clean_logits.cpu(),
            patched_logits=patched_logits.cpu(),
            clean_loss=clean_loss,
            patched_loss=patched_loss,
            clean_accuracy=clean_accuracy,
            patched_accuracy=patched_accuracy,
            effect_size=effect_size,
            layer_name=','.join(patch_layers),
            patch_type='activation_patch'
        )

    def mean_ablation(self,
                     inputs: torch.Tensor,
                     targets: torch.Tensor,
                     layer_name: str,
                     concept_indices: Optional[List[int]] = None) -> AblationResult:
        """
        Replace activations with their mean (zero ablation baseline).

        Args:
            inputs: Input tensor
            targets: Target labels
            layer_name: Layer to ablate
            concept_indices: Specific dimensions to ablate (None = ablate all)

        Returns:
            AblationResult with metrics
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get original outputs and activations
        original_output, cache = self.run_with_cache(inputs, [layer_name])
        original_act = cache.get(layer_name)

        if original_act is None:
            raise ValueError(f"Could not extract activations from layer: {layer_name}")

        # Compute mean activation
        mean_act = original_act.mean(dim=0, keepdim=True).expand_as(original_act)

        # Create ablation hook
        def ablate_hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
                rest = output[1:]
            else:
                act = output
                rest = None

            ablated = act.clone()
            if concept_indices is not None:
                # Ablate specific dimensions
                ablated[..., concept_indices] = mean_act[..., concept_indices]
            else:
                # Ablate all
                ablated = mean_act

            if rest is not None:
                return (ablated,) + rest
            return ablated

        # Run with ablation
        if layer_name in self.hook_points:
            module = self.hook_points[layer_name]
            hook = module.register_forward_hook(ablate_hook)

            with torch.no_grad():
                ablated_output = self.model(inputs)

            hook.remove()
        else:
            raise ValueError(f"Layer not found: {layer_name}")

        # Compute metrics
        original_loss = F.cross_entropy(original_output, targets).item()
        ablated_loss = F.cross_entropy(ablated_output, targets).item()

        original_preds = original_output.argmax(dim=-1)
        ablated_preds = ablated_output.argmax(dim=-1)

        original_accuracy = (original_preds == targets).float().mean().item()
        ablated_accuracy = (ablated_preds == targets).float().mean().item()

        cache.clear()

        return AblationResult(
            original_accuracy=original_accuracy,
            ablated_accuracy=ablated_accuracy,
            accuracy_drop=original_accuracy - ablated_accuracy,
            original_loss=original_loss,
            ablated_loss=ablated_loss,
            loss_increase=ablated_loss - original_loss,
            ablation_type='mean_ablation',
            target=layer_name
        )

    def zero_ablation(self,
                     inputs: torch.Tensor,
                     targets: torch.Tensor,
                     layer_name: str,
                     dimensions: Optional[List[int]] = None) -> AblationResult:
        """
        Zero out activations (stronger ablation than mean).

        Args:
            inputs: Input tensor
            targets: Target labels
            layer_name: Layer to ablate
            dimensions: Specific dimensions to zero (None = zero all)

        Returns:
            AblationResult with metrics
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get original outputs
        with torch.no_grad():
            original_output = self.model(inputs)

        # Create zero ablation hook
        def zero_hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
                rest = output[1:]
            else:
                act = output
                rest = None

            ablated = act.clone()
            if dimensions is not None:
                ablated[..., dimensions] = 0.0
            else:
                ablated = torch.zeros_like(act)

            if rest is not None:
                return (ablated,) + rest
            return ablated

        # Run with ablation
        if layer_name in self.hook_points:
            module = self.hook_points[layer_name]
            hook = module.register_forward_hook(zero_hook)

            with torch.no_grad():
                ablated_output = self.model(inputs)

            hook.remove()
        else:
            raise ValueError(f"Layer not found: {layer_name}")

        # Compute metrics
        original_loss = F.cross_entropy(original_output, targets).item()
        ablated_loss = F.cross_entropy(ablated_output, targets).item()

        original_preds = original_output.argmax(dim=-1)
        ablated_preds = ablated_output.argmax(dim=-1)

        original_accuracy = (original_preds == targets).float().mean().item()
        ablated_accuracy = (ablated_preds == targets).float().mean().item()

        return AblationResult(
            original_accuracy=original_accuracy,
            ablated_accuracy=ablated_accuracy,
            accuracy_drop=original_accuracy - ablated_accuracy,
            original_loss=original_loss,
            ablated_loss=ablated_loss,
            loss_increase=ablated_loss - original_loss,
            ablation_type='zero_ablation',
            target=layer_name
        )


class AttentionPatcher:
    """
    Specialized patcher for attention head interventions.

    Provides fine-grained control over attention pattern manipulation,
    including head ablation, attention pattern patching, and QK/OV analysis.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Discover attention layers
        self.attention_layers = self._find_attention_layers()

    def _find_attention_layers(self) -> Dict[str, nn.Module]:
        """Find all attention layers in the model."""
        attn_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                attn_layers[name] = module
            elif 'attn' in name.lower() or 'attention' in name.lower():
                attn_layers[name] = module
        return attn_layers

    def ablate_head(self,
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   layer_idx: int,
                   head_idx: int) -> AblationResult:
        """
        Zero out a specific attention head.

        Args:
            inputs: Input tensor
            targets: Target labels
            layer_idx: Index of the attention layer
            head_idx: Index of the head to ablate

        Returns:
            AblationResult with metrics
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get original outputs
        with torch.no_grad():
            original_output = self.model(inputs)

        # Find the target attention layer
        attn_layer_name = f'layers.{layer_idx}.self_attn'  # Common naming

        # Try alternative naming patterns
        if attn_layer_name not in self.attention_layers:
            for name in self.attention_layers:
                if f'{layer_idx}' in name and 'attn' in name.lower():
                    attn_layer_name = name
                    break

        if attn_layer_name not in self.attention_layers:
            # Fall back to index-based selection
            layer_names = list(self.attention_layers.keys())
            if layer_idx < len(layer_names):
                attn_layer_name = layer_names[layer_idx]
            else:
                raise ValueError(f"Attention layer {layer_idx} not found")

        attn_module = self.attention_layers[attn_layer_name]

        # Create head ablation hook
        # This assumes standard PyTorch MultiheadAttention output format
        def ablate_head_hook(module, input, output):
            attn_output = output[0] if isinstance(output, tuple) else output

            # Get head dimension
            if hasattr(module, 'num_heads'):
                num_heads = module.num_heads
                head_dim = attn_output.shape[-1] // num_heads

                # Zero out the specific head's contribution
                ablated = attn_output.clone()
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                ablated[..., start_idx:end_idx] = 0.0

                if isinstance(output, tuple):
                    return (ablated,) + output[1:]
                return ablated

            return output

        # Apply ablation
        hook = attn_module.register_forward_hook(ablate_head_hook)

        with torch.no_grad():
            ablated_output = self.model(inputs)

        hook.remove()

        # Compute metrics
        original_loss = F.cross_entropy(original_output, targets).item()
        ablated_loss = F.cross_entropy(ablated_output, targets).item()

        original_accuracy = (original_output.argmax(-1) == targets).float().mean().item()
        ablated_accuracy = (ablated_output.argmax(-1) == targets).float().mean().item()

        return AblationResult(
            original_accuracy=original_accuracy,
            ablated_accuracy=ablated_accuracy,
            accuracy_drop=original_accuracy - ablated_accuracy,
            original_loss=original_loss,
            ablated_loss=ablated_loss,
            loss_increase=ablated_loss - original_loss,
            ablation_type='head_ablation',
            target=f'layer_{layer_idx}_head_{head_idx}'
        )

    def get_attention_patterns(self,
                              inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns from all attention layers.

        Args:
            inputs: Input tensor

        Returns:
            Dictionary mapping layer names to attention weight tensors
        """
        inputs = inputs.to(self.device)
        patterns = {}
        hooks = []

        # Create hooks to capture attention weights
        def make_pattern_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    # Second element is typically attention weights
                    patterns[name] = output[1].detach().cpu()
            return hook_fn

        for name, module in self.attention_layers.items():
            hook = module.register_forward_hook(make_pattern_hook(name))
            hooks.append(hook)

        # Run forward pass
        with torch.no_grad():
            _ = self.model(inputs)

        # Clean up
        for hook in hooks:
            hook.remove()

        return patterns


class CausalAnalyzer:
    """
    High-level causal analysis combining patching and ablation studies.

    Provides methods for systematic causal verification of concepts.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.patcher = ActivationPatcher(model, device)
        self.attn_patcher = AttentionPatcher(model, device)

    def analyze_layer_importance(self,
                                dataset,
                                layers: Optional[List[str]] = None) -> Dict[str, AblationResult]:
        """
        Measure causal importance of each layer.

        Args:
            dataset: Dataset with inputs and targets
            layers: Layers to analyze (None = all discovered layers)

        Returns:
            Dictionary mapping layer names to ablation results
        """
        if layers is None:
            layers = list(self.patcher.hook_points.keys())

        inputs = dataset.data['inputs'].to(self.device)
        targets = dataset.data['targets'].to(self.device)

        results = {}
        for layer in layers:
            try:
                result = self.patcher.mean_ablation(inputs, targets, layer)
                results[layer] = result
            except Exception as e:
                print(f"Warning: Could not ablate layer {layer}: {e}")

        return results

    def analyze_concept_importance(self,
                                  dataset,
                                  concept_indices: Dict[str, List[int]],
                                  layer_name: str) -> Dict[str, AblationResult]:
        """
        Measure causal importance of extracted concepts.

        Args:
            dataset: Dataset with inputs and targets
            concept_indices: Mapping from concept names to dimension indices
            layer_name: Layer where concepts were extracted

        Returns:
            Dictionary mapping concept names to ablation results
        """
        inputs = dataset.data['inputs'].to(self.device)
        targets = dataset.data['targets'].to(self.device)

        results = {}
        for concept_name, indices in concept_indices.items():
            result = self.patcher.mean_ablation(
                inputs, targets, layer_name, concept_indices=indices
            )
            results[concept_name] = result

        return results

    def run_patching_scan(self,
                         clean_inputs: torch.Tensor,
                         corrupted_inputs: torch.Tensor,
                         targets: torch.Tensor,
                         layers: Optional[List[str]] = None) -> Dict[str, PatchingResult]:
        """
        Scan all layers to find which are most important for computation.

        Args:
            clean_inputs: Clean input examples
            corrupted_inputs: Corrupted input examples
            targets: Target labels
            layers: Layers to scan (None = all)

        Returns:
            Dictionary mapping layer names to patching results
        """
        if layers is None:
            layers = list(self.patcher.hook_points.keys())

        results = {}
        for layer in layers:
            try:
                result = self.patcher.compute_patching_effect(
                    clean_inputs, corrupted_inputs, targets, [layer]
                )
                results[layer] = result
            except Exception as e:
                print(f"Warning: Could not patch layer {layer}: {e}")

        return results

    def generate_report(self,
                       layer_results: Dict[str, AblationResult],
                       concept_results: Optional[Dict[str, AblationResult]] = None) -> str:
        """
        Generate a human-readable causal analysis report.

        Args:
            layer_results: Results from layer importance analysis
            concept_results: Optional results from concept importance analysis

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CAUSAL ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Layer importance
        lines.append("LAYER IMPORTANCE (sorted by accuracy drop)")
        lines.append("-" * 40)

        sorted_layers = sorted(
            layer_results.items(),
            key=lambda x: x[1].accuracy_drop,
            reverse=True
        )

        for layer_name, result in sorted_layers:
            lines.append(f"  {layer_name}:")
            lines.append(f"    Accuracy drop: {result.accuracy_drop:.2%}")
            lines.append(f"    Loss increase: {result.loss_increase:.4f}")

        if concept_results:
            lines.append("")
            lines.append("CONCEPT IMPORTANCE (sorted by accuracy drop)")
            lines.append("-" * 40)

            sorted_concepts = sorted(
                concept_results.items(),
                key=lambda x: x[1].accuracy_drop,
                reverse=True
            )

            for concept_name, result in sorted_concepts:
                lines.append(f"  {concept_name}:")
                lines.append(f"    Accuracy drop: {result.accuracy_drop:.2%}")
                lines.append(f"    Loss increase: {result.loss_increase:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def create_corrupted_inputs(inputs: torch.Tensor,
                           method: str = 'shuffle',
                           p: int = 17) -> torch.Tensor:
    """
    Create corrupted inputs for activation patching experiments.

    Args:
        inputs: Original input tensor of shape (batch, 2)
        method: Corruption method ('shuffle', 'random', 'adjacent')
        p: Modulus for modular arithmetic

    Returns:
        Corrupted input tensor
    """
    batch_size = inputs.shape[0]

    if method == 'shuffle':
        # Shuffle the batch order
        perm = torch.randperm(batch_size)
        return inputs[perm]

    elif method == 'random':
        # Generate completely random inputs
        return torch.randint(0, p, inputs.shape, dtype=inputs.dtype)

    elif method == 'adjacent':
        # Shift each number by 1 (circular)
        return (inputs + 1) % p

    else:
        raise ValueError(f"Unknown corruption method: {method}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))

    from Dataset.dataset import ModularArithmeticDataset
    from models.transformer import create_model

    print("Testing Causal Intervention Module")
    print("=" * 50)

    # Create test data and model
    p = 5
    dataset = ModularArithmeticDataset(p=p)
    model = create_model(vocab_size=p)

    # Initialize patcher
    patcher = ActivationPatcher(model)
    print(f"Discovered hook points: {list(patcher.hook_points.keys())[:5]}...")

    # Test activation caching
    inputs = dataset.data['inputs'][:8]
    outputs, cache = patcher.run_with_cache(inputs)
    print(f"Cached activations: {list(cache.keys())[:3]}...")

    # Test mean ablation
    targets = dataset.data['targets'][:8]

    for layer_name in list(patcher.hook_points.keys())[:2]:
        try:
            result = patcher.mean_ablation(inputs, targets, layer_name)
            print(f"\nMean ablation on {layer_name}:")
            print(f"  Accuracy drop: {result.accuracy_drop:.2%}")
            print(f"  Loss increase: {result.loss_increase:.4f}")
        except Exception as e:
            print(f"  Could not ablate {layer_name}: {e}")

    # Test activation patching
    corrupted_inputs = create_corrupted_inputs(inputs, method='shuffle', p=p)

    for layer_name in list(patcher.hook_points.keys())[:2]:
        try:
            result = patcher.compute_patching_effect(
                inputs, corrupted_inputs, targets, [layer_name]
            )
            print(f"\nActivation patching on {layer_name}:")
            print(f"  Effect size: {result.effect_size:.4f}")
            print(f"  Clean accuracy: {result.clean_accuracy:.2%}")
            print(f"  Patched accuracy: {result.patched_accuracy:.2%}")
        except Exception as e:
            print(f"  Could not patch {layer_name}: {e}")

    print("\nCausal intervention module implementation complete!")
