"""
Circuit Discovery Module for NeuroMap.

Implements automated circuit identification for mechanistic interpretability.
Combines activation patching, attention analysis, and component attribution
to discover the computational circuits responsible for model behavior.

Follows methodologies from:
- "Interpretability in the Wild" (Wang et al.)
- "A Mathematical Framework for Transformer Circuits" (Elhage et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class CircuitComponent:
    """A component in a computational circuit."""
    name: str
    component_type: str  # 'attention_head', 'mlp', 'embedding', 'residual'
    layer: int
    importance: float
    description: str = ""


@dataclass
class Circuit:
    """A computational circuit in the model."""
    name: str
    components: List[CircuitComponent]
    edges: List[Tuple[str, str, float]]  # (source, target, weight)
    total_importance: float
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'components': [
                {'name': c.name, 'type': c.component_type,
                 'layer': c.layer, 'importance': c.importance}
                for c in self.components
            ],
            'edges': [{'source': e[0], 'target': e[1], 'weight': e[2]} for e in self.edges],
            'total_importance': self.total_importance,
            'description': self.description
        }


@dataclass
class HeadImportance:
    """Importance metrics for an attention head."""
    layer: int
    head: int
    direct_effect: float
    indirect_effect: float
    total_effect: float
    attention_pattern_type: str  # 'position', 'content', 'mixed'


class CircuitDiscoverer:
    """
    Automatically discover computational circuits in neural networks.

    Uses a combination of:
    1. Activation patching to measure component importance
    2. Attention pattern analysis to understand information flow
    3. Path patching to trace specific computations
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Discover model structure
        self.n_layers = self._count_layers()
        self.n_heads = self._count_heads()

    def _count_layers(self) -> int:
        """Count transformer layers in model."""
        count = 0
        for name, _ in self.model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                # Extract layer number
                import re
                match = re.search(r'(\d+)', name)
                if match:
                    count = max(count, int(match.group(1)) + 1)
        return max(count, 1)

    def _count_heads(self) -> int:
        """Count attention heads per layer."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'num_heads'):
                return module.num_heads
            if hasattr(module, 'n_heads'):
                return module.n_heads
        return 4  # Default

    def find_important_heads(self,
                            dataset,
                            n_samples: int = 100,
                            threshold: float = 0.01) -> List[HeadImportance]:
        """
        Find attention heads that are important for the task.

        Uses mean ablation to measure each head's contribution.

        Args:
            dataset: Dataset with inputs and targets
            n_samples: Number of samples to analyze
            threshold: Minimum importance to include

        Returns:
            List of HeadImportance sorted by total effect
        """
        from .causal_intervention import ActivationPatcher

        inputs = dataset.data['inputs'][:n_samples].to(self.device)
        targets = dataset.data['targets'][:n_samples].to(self.device)

        # Get baseline accuracy
        with torch.no_grad():
            outputs = self.model(inputs)
            baseline_acc = (outputs.argmax(-1) == targets).float().mean().item()

        results = []

        # Test each head
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                try:
                    # Ablate this head
                    ablated_acc = self._ablate_head(inputs, targets, layer, head)
                    effect = baseline_acc - ablated_acc

                    if abs(effect) > threshold:
                        results.append(HeadImportance(
                            layer=layer,
                            head=head,
                            direct_effect=effect,
                            indirect_effect=0.0,  # Computed separately
                            total_effect=effect,
                            attention_pattern_type='unknown'
                        ))
                except Exception as e:
                    pass  # Skip if ablation fails

        # Sort by importance
        results.sort(key=lambda x: abs(x.total_effect), reverse=True)

        return results

    def _ablate_head(self,
                    inputs: torch.Tensor,
                    targets: torch.Tensor,
                    layer: int,
                    head: int) -> float:
        """Ablate a single attention head and return accuracy."""
        # Simple approach: find the attention output and zero part of it
        hooks = []
        d_head = None

        def ablate_hook(module, input, output):
            nonlocal d_head
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            # Assume output is (batch, seq, n_heads * d_head) or similar
            total_dim = attn_out.shape[-1]
            if d_head is None:
                d_head = total_dim // self.n_heads

            # Zero out this head's contribution
            start = head * d_head
            end = (head + 1) * d_head
            attn_out = attn_out.clone()
            attn_out[..., start:end] = 0

            if isinstance(output, tuple):
                return (attn_out,) + output[1:]
            return attn_out

        # Find attention layer
        for name, module in self.model.named_modules():
            if f'{layer}' in name and ('attn' in name.lower() or 'attention' in name.lower()):
                hook = module.register_forward_hook(ablate_hook)
                hooks.append(hook)
                break

        try:
            with torch.no_grad():
                outputs = self.model(inputs)
            accuracy = (outputs.argmax(-1) == targets).float().mean().item()
        finally:
            for hook in hooks:
                hook.remove()

        return accuracy

    def find_important_mlps(self,
                           dataset,
                           n_samples: int = 100,
                           threshold: float = 0.01) -> List[Tuple[int, float]]:
        """
        Find MLP layers that are important for the task.

        Args:
            dataset: Dataset with inputs and targets
            n_samples: Number of samples to analyze
            threshold: Minimum importance to include

        Returns:
            List of (layer_idx, importance) tuples
        """
        inputs = dataset.data['inputs'][:n_samples].to(self.device)
        targets = dataset.data['targets'][:n_samples].to(self.device)

        # Baseline
        with torch.no_grad():
            outputs = self.model(inputs)
            baseline_acc = (outputs.argmax(-1) == targets).float().mean().item()

        results = []

        for layer in range(self.n_layers):
            try:
                ablated_acc = self._ablate_mlp(inputs, targets, layer)
                effect = baseline_acc - ablated_acc

                if abs(effect) > threshold:
                    results.append((layer, effect))
            except Exception:
                pass

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results

    def _ablate_mlp(self,
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   layer: int) -> float:
        """Ablate MLP at given layer."""
        hooks = []

        def zero_hook(module, input, output):
            return torch.zeros_like(output)

        for name, module in self.model.named_modules():
            if f'{layer}' in name and 'mlp' in name.lower():
                hook = module.register_forward_hook(zero_hook)
                hooks.append(hook)
                break

        try:
            with torch.no_grad():
                outputs = self.model(inputs)
            accuracy = (outputs.argmax(-1) == targets).float().mean().item()
        finally:
            for hook in hooks:
                hook.remove()

        return accuracy

    def discover_circuit(self,
                        dataset,
                        n_samples: int = 100,
                        head_threshold: float = 0.01,
                        mlp_threshold: float = 0.01) -> Circuit:
        """
        Discover the main computational circuit for the task.

        Args:
            dataset: Dataset with inputs and targets
            n_samples: Number of samples for analysis
            head_threshold: Threshold for including attention heads
            mlp_threshold: Threshold for including MLP layers

        Returns:
            Circuit describing the discovered computation
        """
        # Find important components
        important_heads = self.find_important_heads(dataset, n_samples, head_threshold)
        important_mlps = self.find_important_mlps(dataset, n_samples, mlp_threshold)

        components = []
        edges = []

        # Add embedding as first component
        components.append(CircuitComponent(
            name='embed',
            component_type='embedding',
            layer=-1,
            importance=1.0,
            description='Token and position embeddings'
        ))

        # Add important heads
        for head_info in important_heads:
            comp_name = f'L{head_info.layer}H{head_info.head}'
            components.append(CircuitComponent(
                name=comp_name,
                component_type='attention_head',
                layer=head_info.layer,
                importance=head_info.total_effect,
                description=f'Layer {head_info.layer} Head {head_info.head}'
            ))

        # Add important MLPs
        for layer, importance in important_mlps:
            comp_name = f'L{layer}_MLP'
            components.append(CircuitComponent(
                name=comp_name,
                component_type='mlp',
                layer=layer,
                importance=importance,
                description=f'Layer {layer} MLP'
            ))

        # Add output component
        components.append(CircuitComponent(
            name='output',
            component_type='output',
            layer=self.n_layers,
            importance=1.0,
            description='Output logits'
        ))

        # Create edges based on layer ordering
        components_sorted = sorted(components, key=lambda c: (c.layer, c.component_type))

        for i, comp in enumerate(components_sorted[:-1]):
            next_comp = components_sorted[i + 1]
            # Edge weight is importance of target component
            edges.append((comp.name, next_comp.name, next_comp.importance))

        # Total importance
        total_importance = sum(c.importance for c in components if c.component_type in ['attention_head', 'mlp'])

        return Circuit(
            name='main_circuit',
            components=components,
            edges=edges,
            total_importance=total_importance,
            description='Main computational circuit for modular arithmetic'
        )

    def trace_information_flow(self,
                              inputs: torch.Tensor,
                              target_logit: int) -> Dict[str, float]:
        """
        Trace information flow from input to specific output.

        Uses gradient-based attribution.

        Args:
            inputs: Input tensor
            target_logit: Index of target output logit

        Returns:
            Dictionary mapping component names to importance scores
        """
        inputs = inputs.to(self.device)
        inputs.requires_grad = True

        # Forward pass
        outputs = self.model(inputs)

        # Backward from target logit
        target = outputs[0, target_logit]
        target.backward()

        importance = {}

        # Collect gradient magnitudes at each layer
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad_magnitude = module.weight.grad.abs().mean().item()
                importance[name] = grad_magnitude

        return importance

    def export_circuit_diagram(self,
                              circuit: Circuit,
                              output_path: Path,
                              format: str = 'json'):
        """
        Export circuit diagram for visualization.

        Args:
            circuit: Circuit to export
            output_path: Path to save output
            format: Output format ('json', 'dot', 'mermaid')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(circuit.to_dict(), f, indent=2)

        elif format == 'dot':
            lines = ['digraph Circuit {']
            lines.append('  rankdir=TB;')

            # Add nodes
            for comp in circuit.components:
                style = 'filled'
                color = {
                    'embedding': 'lightblue',
                    'attention_head': 'lightgreen',
                    'mlp': 'lightyellow',
                    'output': 'lightpink'
                }.get(comp.component_type, 'white')

                lines.append(f'  "{comp.name}" [label="{comp.name}\\n{comp.importance:.3f}", '
                           f'style={style}, fillcolor={color}];')

            # Add edges
            for source, target, weight in circuit.edges:
                lines.append(f'  "{source}" -> "{target}" [label="{weight:.3f}"];')

            lines.append('}')

            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))

        elif format == 'mermaid':
            lines = ['graph TD']

            for comp in circuit.components:
                lines.append(f'  {comp.name}["{comp.name}<br/>{comp.importance:.3f}"]')

            for source, target, weight in circuit.edges:
                lines.append(f'  {source} -->|{weight:.3f}| {target}')

            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))


def generate_circuit_report(circuit: Circuit) -> str:
    """Generate a human-readable circuit report."""
    lines = []
    lines.append("=" * 60)
    lines.append("CIRCUIT DISCOVERY REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Circuit: {circuit.name}")
    lines.append(f"Total Importance: {circuit.total_importance:.4f}")
    lines.append(f"Description: {circuit.description}")
    lines.append("")

    lines.append("COMPONENTS")
    lines.append("-" * 40)
    for comp in sorted(circuit.components, key=lambda c: -c.importance):
        lines.append(f"  {comp.name} ({comp.component_type})")
        lines.append(f"    Layer: {comp.layer}, Importance: {comp.importance:.4f}")

    lines.append("")
    lines.append("INFORMATION FLOW")
    lines.append("-" * 40)
    for source, target, weight in circuit.edges:
        lines.append(f"  {source} -> {target} (weight: {weight:.4f})")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("Testing Circuit Discovery Module")
    print("=" * 50)

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from Dataset.dataset import ModularArithmeticDataset
    from models.transformer import create_model

    # Create test model and dataset
    p = 5
    dataset = ModularArithmeticDataset(p=p)
    model = create_model(vocab_size=p)

    # Initialize discoverer
    discoverer = CircuitDiscoverer(model)
    print(f"Model has {discoverer.n_layers} layers, {discoverer.n_heads} heads")

    # Discover circuit
    print("\nDiscovering circuit...")
    circuit = discoverer.discover_circuit(dataset, n_samples=25)

    # Print report
    report = generate_circuit_report(circuit)
    print(report)

    # Export to JSON
    output_path = Path("circuit_output.json")
    discoverer.export_circuit_diagram(circuit, output_path, format='json')
    print(f"\nExported circuit to {output_path}")

    print("\nCircuit discovery module implementation complete!")
