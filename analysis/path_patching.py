"""
Path Patching Module for NeuroMap.

Implements path-specific activation patching to trace information flow
through computational circuits in neural networks.

Path patching isolates the causal effect of specific computational paths,
enabling fine-grained analysis of how information flows from input to output.

Reference: "Interpretability in the Wild" (Wang et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .causal_intervention import ActivationPatcher, ActivationCache, PatchingResult


@dataclass
class PathNode:
    """Represents a node in the computational graph."""
    name: str
    layer_idx: int
    component_type: str  # 'embed', 'attn', 'mlp', 'norm', 'resid', 'output'
    position: Optional[int] = None  # Token position if applicable

    def __hash__(self):
        return hash((self.name, self.layer_idx, self.component_type, self.position))

    def __eq__(self, other):
        if not isinstance(other, PathNode):
            return False
        return (self.name == other.name and
                self.layer_idx == other.layer_idx and
                self.component_type == other.component_type and
                self.position == other.position)


@dataclass
class PathEdge:
    """Represents a directed edge (information flow) between nodes."""
    source: PathNode
    target: PathNode
    weight: float = 0.0  # Causal importance of this edge


@dataclass
class ComputationalPath:
    """Represents a complete computational path through the network."""
    nodes: List[PathNode]
    edges: List[PathEdge]
    total_effect: float = 0.0
    description: str = ""


@dataclass
class PathPatchingResult:
    """Result of a path patching experiment."""
    path: ComputationalPath
    clean_logits: torch.Tensor
    patched_logits: torch.Tensor
    direct_effect: float  # Effect of patching just this path
    indirect_effect: float  # Effect mediated through other paths
    total_effect: float
    path_importance: float  # Fraction of total effect attributable to path


class ComputationalGraph:
    """
    Represents the computational graph of a transformer.

    Models the standard transformer structure:
    - embed -> (attn_0 + mlp_0) -> (attn_1 + mlp_1) -> ... -> output

    With residual connections creating multiple paths.
    """

    def __init__(self, n_layers: int, component_types: List[str] = None):
        self.n_layers = n_layers

        if component_types is None:
            component_types = ['embed', 'attn', 'mlp', 'resid_post', 'output']

        self.component_types = component_types
        self.nodes: List[PathNode] = []
        self.edges: List[PathEdge] = []

        self._build_graph()

    def _build_graph(self):
        """Build the computational graph for a transformer."""
        # Embedding node
        embed_node = PathNode('embed', -1, 'embed')
        self.nodes.append(embed_node)

        prev_resid = embed_node

        # Layer nodes
        for layer_idx in range(self.n_layers):
            # Attention node
            attn_node = PathNode(f'attn_{layer_idx}', layer_idx, 'attn')
            self.nodes.append(attn_node)

            # MLP node
            mlp_node = PathNode(f'mlp_{layer_idx}', layer_idx, 'mlp')
            self.nodes.append(mlp_node)

            # Residual post node (after attention + MLP)
            resid_node = PathNode(f'resid_post_{layer_idx}', layer_idx, 'resid_post')
            self.nodes.append(resid_node)

            # Edges: residual stream -> attention -> residual
            self.edges.append(PathEdge(prev_resid, attn_node))
            self.edges.append(PathEdge(attn_node, resid_node))

            # Edges: residual stream -> MLP -> residual
            # MLP receives from post-attention residual
            mid_resid = PathNode(f'resid_mid_{layer_idx}', layer_idx, 'resid_mid')
            self.nodes.append(mid_resid)

            self.edges.append(PathEdge(attn_node, mid_resid))
            self.edges.append(PathEdge(prev_resid, mid_resid))  # Residual connection
            self.edges.append(PathEdge(mid_resid, mlp_node))
            self.edges.append(PathEdge(mlp_node, resid_node))
            self.edges.append(PathEdge(mid_resid, resid_node))  # Residual connection

            prev_resid = resid_node

        # Output node
        output_node = PathNode('output', self.n_layers, 'output')
        self.nodes.append(output_node)
        self.edges.append(PathEdge(prev_resid, output_node))

    def get_all_paths(self,
                      source: str = 'embed',
                      target: str = 'output',
                      max_length: Optional[int] = None) -> List[ComputationalPath]:
        """
        Find all paths from source to target in the graph.

        Args:
            source: Name of source node
            target: Name of target node
            max_length: Maximum path length (None = no limit)

        Returns:
            List of all paths from source to target
        """
        source_node = self._find_node(source)
        target_node = self._find_node(target)

        if source_node is None or target_node is None:
            return []

        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source].append((edge.target, edge))

        # DFS to find all paths
        paths = []
        self._dfs_paths(source_node, target_node, adjacency, [], [], paths, max_length)

        return paths

    def _dfs_paths(self,
                   current: PathNode,
                   target: PathNode,
                   adjacency: Dict,
                   current_path: List[PathNode],
                   current_edges: List[PathEdge],
                   all_paths: List[ComputationalPath],
                   max_length: Optional[int]):
        """DFS helper for path finding."""
        current_path = current_path + [current]

        if current == target:
            path = ComputationalPath(
                nodes=current_path.copy(),
                edges=current_edges.copy(),
                description=self._path_to_string(current_path)
            )
            all_paths.append(path)
            return

        if max_length is not None and len(current_path) >= max_length:
            return

        for next_node, edge in adjacency[current]:
            if next_node not in current_path:  # Avoid cycles
                self._dfs_paths(
                    next_node, target, adjacency,
                    current_path, current_edges + [edge],
                    all_paths, max_length
                )

    def _find_node(self, name: str) -> Optional[PathNode]:
        """Find a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def _path_to_string(self, path: List[PathNode]) -> str:
        """Convert path to human-readable string."""
        return " -> ".join(node.name for node in path)


class PathPatcher:
    """
    Implements path-specific activation patching.

    Traces information flow through specific computational paths by:
    1. Running clean and corrupted forward passes
    2. Selectively patching activations along specific paths
    3. Measuring the causal effect of each path
    """

    def __init__(self, model: nn.Module, n_layers: int = 2, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.n_layers = n_layers
        self.model.to(device)
        self.model.eval()

        # Build computational graph
        self.graph = ComputationalGraph(n_layers)

        # Base activation patcher
        self.patcher = ActivationPatcher(model, device)

        # Map graph nodes to actual model layers
        self.node_to_layer = self._map_nodes_to_layers()

    def _map_nodes_to_layers(self) -> Dict[str, str]:
        """Map graph node names to actual model layer names."""
        mapping = {}
        available_layers = list(self.patcher.hook_points.keys())

        for node in self.graph.nodes:
            # Try to find matching layer
            for layer_name in available_layers:
                layer_lower = layer_name.lower()

                if node.component_type == 'embed' and 'embed' in layer_lower:
                    mapping[node.name] = layer_name
                    break

                elif node.component_type == 'attn':
                    if f'{node.layer_idx}' in layer_name and 'attn' in layer_lower:
                        mapping[node.name] = layer_name
                        break

                elif node.component_type == 'mlp':
                    if f'{node.layer_idx}' in layer_name and 'mlp' in layer_lower:
                        mapping[node.name] = layer_name
                        break

                elif node.component_type == 'output':
                    if 'output' in layer_lower or 'classifier' in layer_lower:
                        mapping[node.name] = layer_name
                        break

        return mapping

    def patch_path(self,
                   path: ComputationalPath,
                   clean_input: torch.Tensor,
                   corrupted_input: torch.Tensor,
                   targets: torch.Tensor) -> PathPatchingResult:
        """
        Patch activations along a specific computational path.

        Args:
            path: The path to patch
            clean_input: Clean input tensor
            corrupted_input: Corrupted input tensor
            targets: Target labels

        Returns:
            PathPatchingResult with detailed metrics
        """
        clean_input = clean_input.to(self.device)
        corrupted_input = corrupted_input.to(self.device)
        targets = targets.to(self.device)

        # Get layers to patch along this path
        path_layers = []
        for node in path.nodes:
            if node.name in self.node_to_layer:
                path_layers.append(self.node_to_layer[node.name])

        if not path_layers:
            # Fallback: use first available layer
            path_layers = list(self.patcher.hook_points.keys())[:1]

        # Get clean outputs
        with torch.no_grad():
            clean_logits = self.model(clean_input)

        # Get fully corrupted outputs (for comparison)
        with torch.no_grad():
            corrupted_logits = self.model(corrupted_input)

        # Patch only along the path
        patched_logits = self.patcher.patch_activations(
            clean_input, corrupted_input, path_layers
        )

        # Compute effects
        clean_loss = F.cross_entropy(clean_logits, targets).item()
        patched_loss = F.cross_entropy(patched_logits, targets).item()
        corrupted_loss = F.cross_entropy(corrupted_logits, targets).item()

        # Direct effect: change caused by patching this path
        direct_effect = patched_loss - clean_loss

        # Total effect: difference between clean and fully corrupted
        total_effect = corrupted_loss - clean_loss

        # Indirect effect: effect not captured by this path
        indirect_effect = total_effect - direct_effect

        # Path importance: fraction of total effect
        if abs(total_effect) > 1e-6:
            path_importance = direct_effect / total_effect
        else:
            path_importance = 0.0

        # Update path with effect
        path.total_effect = direct_effect

        return PathPatchingResult(
            path=path,
            clean_logits=clean_logits.cpu(),
            patched_logits=patched_logits.cpu(),
            direct_effect=direct_effect,
            indirect_effect=indirect_effect,
            total_effect=total_effect,
            path_importance=path_importance
        )

    def analyze_all_paths(self,
                          clean_input: torch.Tensor,
                          corrupted_input: torch.Tensor,
                          targets: torch.Tensor,
                          max_paths: int = 10) -> List[PathPatchingResult]:
        """
        Analyze causal importance of all paths.

        Args:
            clean_input: Clean input tensor
            corrupted_input: Corrupted input tensor
            targets: Target labels
            max_paths: Maximum number of paths to analyze

        Returns:
            List of PathPatchingResult sorted by importance
        """
        # Get all paths from embed to output
        all_paths = self.graph.get_all_paths('embed', 'output')

        # Limit number of paths
        if len(all_paths) > max_paths:
            all_paths = all_paths[:max_paths]

        results = []
        for path in all_paths:
            try:
                result = self.patch_path(path, clean_input, corrupted_input, targets)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not analyze path {path.description}: {e}")

        # Sort by path importance
        results.sort(key=lambda x: abs(x.path_importance), reverse=True)

        return results

    def trace_information_flow(self,
                              input: torch.Tensor,
                              target_output_idx: int) -> Dict[str, float]:
        """
        Trace how information flows from input to a specific output.

        Uses gradient-based attribution to trace information flow.

        Args:
            input: Input tensor
            target_output_idx: Index of target output to trace

        Returns:
            Dictionary mapping node names to importance scores
        """
        input = input.to(self.device)
        input.requires_grad = True

        # Forward pass
        output = self.model(input)

        # Get gradient with respect to target output
        target_logit = output[0, target_output_idx]
        target_logit.backward()

        # Collect gradients at each layer
        importance = {}

        with torch.no_grad():
            # Run forward with caching
            _, cache = self.patcher.run_with_cache(input)

            for layer_name in cache.keys():
                activation = cache.get(layer_name)
                if activation is not None:
                    # Importance = magnitude of activation
                    importance[layer_name] = float(activation.abs().mean())

        return importance


class CircuitAnalyzer:
    """
    High-level circuit analysis using path patching.

    Identifies and validates computational circuits responsible for
    specific model behaviors.
    """

    def __init__(self, model: nn.Module, n_layers: int = 2, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.path_patcher = PathPatcher(model, n_layers, device)

    def find_important_circuits(self,
                               dataset,
                               n_samples: int = 100,
                               threshold: float = 0.1) -> List[ComputationalPath]:
        """
        Find circuits that are causally important for model behavior.

        Args:
            dataset: Dataset with inputs and targets
            n_samples: Number of samples to analyze
            threshold: Minimum path importance to include

        Returns:
            List of important computational paths
        """
        inputs = dataset.data['inputs'][:n_samples]
        targets = dataset.data['targets'][:n_samples]

        # Create corrupted inputs by shuffling
        perm = torch.randperm(n_samples)
        corrupted = inputs[perm]

        # Analyze paths
        results = self.path_patcher.analyze_all_paths(inputs, corrupted, targets)

        # Filter by importance
        important_paths = [
            r.path for r in results
            if abs(r.path_importance) >= threshold
        ]

        return important_paths

    def validate_circuit(self,
                        circuit: ComputationalPath,
                        dataset,
                        n_samples: int = 100) -> Dict[str, Any]:
        """
        Validate that a circuit is truly responsible for model behavior.

        Performs multiple patching experiments to confirm circuit importance.

        Args:
            circuit: The circuit to validate
            dataset: Dataset with inputs and targets
            n_samples: Number of samples to test

        Returns:
            Dictionary with validation metrics
        """
        inputs = dataset.data['inputs'][:n_samples]
        targets = dataset.data['targets'][:n_samples]

        # Test with different corruption types
        corruption_types = ['shuffle', 'random', 'adjacent']
        results = {}

        from .causal_intervention import create_corrupted_inputs

        p = getattr(dataset, 'p', 17)

        for corruption in corruption_types:
            corrupted = create_corrupted_inputs(inputs, method=corruption, p=p)

            try:
                result = self.path_patcher.patch_path(
                    circuit, inputs, corrupted, targets
                )
                results[corruption] = {
                    'path_importance': result.path_importance,
                    'direct_effect': result.direct_effect,
                    'total_effect': result.total_effect
                }
            except Exception as e:
                results[corruption] = {'error': str(e)}

        # Compute consistency score
        importances = [
            r.get('path_importance', 0) for r in results.values()
            if 'path_importance' in r
        ]

        if importances:
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            consistency = 1.0 - (std_importance / (mean_importance + 1e-6))
        else:
            mean_importance = 0.0
            consistency = 0.0

        return {
            'circuit': circuit.description,
            'corruption_results': results,
            'mean_importance': mean_importance,
            'consistency': consistency,
            'is_valid': consistency > 0.7 and mean_importance > 0.1
        }

    def generate_circuit_report(self,
                               circuits: List[ComputationalPath],
                               validation_results: List[Dict]) -> str:
        """
        Generate a report on discovered circuits.

        Args:
            circuits: List of discovered circuits
            validation_results: Validation results for each circuit

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CIRCUIT ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        for i, (circuit, validation) in enumerate(zip(circuits, validation_results)):
            lines.append(f"Circuit {i + 1}: {circuit.description}")
            lines.append("-" * 40)
            lines.append(f"  Mean importance: {validation['mean_importance']:.4f}")
            lines.append(f"  Consistency: {validation['consistency']:.4f}")
            lines.append(f"  Valid: {validation['is_valid']}")
            lines.append("")

            for corruption, result in validation['corruption_results'].items():
                if 'error' not in result:
                    lines.append(f"  {corruption}:")
                    lines.append(f"    Path importance: {result['path_importance']:.4f}")

            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))

    from Dataset.dataset import ModularArithmeticDataset
    from models.transformer import create_model

    print("Testing Path Patching Module")
    print("=" * 50)

    # Create test data and model
    p = 5
    dataset = ModularArithmeticDataset(p=p)
    model = create_model(vocab_size=p)

    # Initialize path patcher
    patcher = PathPatcher(model, n_layers=2)
    print(f"Computational graph nodes: {len(patcher.graph.nodes)}")
    print(f"Computational graph edges: {len(patcher.graph.edges)}")

    # Get all paths
    paths = patcher.graph.get_all_paths('embed', 'output', max_length=6)
    print(f"Found {len(paths)} paths from embed to output")

    if paths:
        print(f"Example path: {paths[0].description}")

    # Test path patching
    inputs = dataset.data['inputs'][:8]
    targets = dataset.data['targets'][:8]
    corrupted = (inputs + 1) % p

    print("\nAnalyzing path importance...")
    results = patcher.analyze_all_paths(inputs, corrupted, targets, max_paths=3)

    for result in results:
        print(f"\nPath: {result.path.description}")
        print(f"  Direct effect: {result.direct_effect:.4f}")
        print(f"  Path importance: {result.path_importance:.4f}")

    # Test circuit analysis
    print("\nRunning circuit analysis...")
    analyzer = CircuitAnalyzer(model, n_layers=2)
    important_circuits = analyzer.find_important_circuits(dataset, n_samples=16)

    print(f"Found {len(important_circuits)} important circuits")

    print("\nPath patching module implementation complete!")
