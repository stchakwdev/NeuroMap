"""
Advanced layout algorithms for concept graph visualization.

This module implements sophisticated layout algorithms designed to preserve
the topology and structure of concept graphs for interpretable visualization.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.manifold import MDS, SpectralEmbedding, TSNE
from sklearn.decomposition import PCA


class CircularLayout:
    """
    Circular layout algorithm optimized for modular arithmetic structures.
    
    This layout attempts to arrange concepts in a circular pattern that
    respects the ground truth mathematical structure.
    """
    
    def __init__(self, preserve_distances: bool = True, radius: float = 1.0):
        self.preserve_distances = preserve_distances
        self.radius = radius
    
    def compute_layout(self, 
                      graph: nx.Graph,
                      true_labels: Optional[torch.Tensor] = None,
                      distance_matrix: Optional[torch.Tensor] = None) -> Dict[int, Tuple[float, float]]:
        """
        Compute circular layout for graph.
        
        Args:
            graph: NetworkX graph
            true_labels: Ground truth labels for ordering
            distance_matrix: Distance matrix for optimization
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        n_nodes = len(graph.nodes())
        
        if true_labels is not None:
            # Use ground truth ordering
            positions = self._ground_truth_circular_layout(graph, true_labels)
        elif self.preserve_distances and distance_matrix is not None:
            # Optimize circular layout to preserve distances
            positions = self._optimized_circular_layout(graph, distance_matrix)
        else:
            # Simple circular layout
            positions = self._simple_circular_layout(graph)
        
        return positions
    
    def _ground_truth_circular_layout(self, 
                                    graph: nx.Graph,
                                    true_labels: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """Create circular layout based on ground truth labels."""
        positions = {}
        
        # Sort nodes by their true labels
        node_label_pairs = [(node, true_labels[node].item()) for node in graph.nodes()]
        sorted_nodes = sorted(node_label_pairs, key=lambda x: x[1])
        
        # Place nodes in circular order
        n_nodes = len(sorted_nodes)
        for i, (node, _) in enumerate(sorted_nodes):
            angle = 2 * np.pi * i / n_nodes
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            positions[node] = (x, y)
        
        return positions
    
    def _optimized_circular_layout(self, 
                                 graph: nx.Graph,
                                 distance_matrix: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """Optimize circular layout to preserve distances."""
        n_nodes = len(graph.nodes())
        node_list = list(graph.nodes())
        
        # Initial circular positions
        initial_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        
        def objective(angles):
            # Convert angles to positions
            positions = []
            for angle in angles:
                x = self.radius * np.cos(angle)
                y = self.radius * np.sin(angle)
                positions.append([x, y])
            positions = np.array(positions)
            
            # Compute layout distances
            layout_distances = squareform(pdist(positions))
            
            # Compare with target distances
            target_distances = distance_matrix.numpy()
            
            # Stress function (minimize difference between distances)
            stress = 0
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    node_i, node_j = node_list[i], node_list[j]
                    target_dist = target_distances[node_i, node_j]
                    layout_dist = layout_distances[i, j]
                    stress += (target_dist - layout_dist) ** 2
            
            return stress
        
        # Optimize angles
        result = minimize(objective, initial_angles, method='L-BFGS-B')
        optimized_angles = result.x
        
        # Convert to positions
        positions = {}
        for i, node in enumerate(node_list):
            angle = optimized_angles[i]
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            positions[node] = (x, y)
        
        return positions
    
    def _simple_circular_layout(self, graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Simple circular layout without optimization."""
        return nx.circular_layout(graph, scale=self.radius)


class ForceDirectedLayout:
    """
    Enhanced force-directed layout with customizable forces.
    
    This layout uses physical simulation with springs and repulsion
    forces to create visually appealing and interpretable layouts.
    """
    
    def __init__(self, 
                 k: float = 1.0,
                 iterations: int = 1000,
                 temperature: float = 1.0,
                 cooling_factor: float = 0.95):
        self.k = k  # Optimal distance between nodes
        self.iterations = iterations
        self.temperature = temperature
        self.cooling_factor = cooling_factor
    
    def compute_layout(self, 
                      graph: nx.Graph,
                      distance_matrix: Optional[torch.Tensor] = None,
                      fixed_positions: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[int, Tuple[float, float]]:
        """
        Compute force-directed layout.
        
        Args:
            graph: NetworkX graph
            distance_matrix: Optional distance matrix for custom forces
            fixed_positions: Optional fixed positions for some nodes
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if distance_matrix is not None:
            positions = self._custom_force_layout(graph, distance_matrix, fixed_positions)
        else:
            positions = self._standard_force_layout(graph, fixed_positions)
        
        return positions
    
    def _standard_force_layout(self, 
                             graph: nx.Graph,
                             fixed_positions: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[int, Tuple[float, float]]:
        """Standard Fruchterman-Reingold force-directed layout."""
        positions = nx.spring_layout(
            graph,
            k=self.k,
            iterations=self.iterations,
            pos=fixed_positions
        )
        return positions
    
    def _custom_force_layout(self, 
                           graph: nx.Graph,
                           distance_matrix: torch.Tensor,
                           fixed_positions: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[int, Tuple[float, float]]:
        """Custom force-directed layout using distance matrix."""
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Initialize positions
        if fixed_positions is not None:
            positions = np.array([[fixed_positions.get(node, (np.random.random(), np.random.random()))[0],
                                 fixed_positions.get(node, (np.random.random(), np.random.random()))[1]] 
                                for node in nodes])
        else:
            positions = np.random.random((n_nodes, 2)) * 2 - 1  # Range [-1, 1]
        
        # Get distance matrix
        target_distances = distance_matrix.numpy()
        
        temperature = self.temperature
        
        for iteration in range(self.iterations):
            # Calculate forces
            forces = np.zeros_like(positions)
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        # Current distance
                        diff = positions[i] - positions[j]
                        current_dist = np.linalg.norm(diff)
                        
                        if current_dist > 0:
                            # Target distance
                            target_dist = target_distances[nodes[i], nodes[j]] * self.k
                            
                            # Force magnitude (spring force)
                            force_mag = (target_dist - current_dist) / current_dist
                            
                            # Apply force
                            forces[i] += force_mag * diff * 0.1
            
            # Update positions with temperature
            displacement = forces * temperature
            
            # Limit maximum displacement
            displacement_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
            max_displacement = temperature
            displacement = np.where(displacement_norm > max_displacement,
                                  displacement * max_displacement / displacement_norm,
                                  displacement)
            
            positions += displacement
            
            # Cool down
            temperature *= self.cooling_factor
        
        # Convert to dictionary
        layout = {}
        for i, node in enumerate(nodes):
            layout[node] = tuple(positions[i])
        
        return layout


class SpectralLayout:
    """
    Spectral layout using graph Laplacian eigenvectors.
    
    This layout uses the spectral properties of the graph to create
    layouts that preserve global graph structure.
    """
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
    
    def compute_layout(self, graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Compute spectral layout."""
        try:
            # Use NetworkX spectral layout if graph is connected
            if nx.is_connected(graph):
                positions = nx.spectral_layout(graph, dim=self.n_components)
                return positions
            else:
                # Handle disconnected graphs
                return self._disconnected_spectral_layout(graph)
        except:
            # Fallback to manual implementation
            return self._manual_spectral_layout(graph)
    
    def _disconnected_spectral_layout(self, graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Handle spectral layout for disconnected graphs."""
        positions = {}
        components = list(nx.connected_components(graph))
        
        # Layout each component separately
        for i, component in enumerate(components):
            subgraph = graph.subgraph(component)
            
            if len(component) > 2:
                try:
                    sub_positions = nx.spectral_layout(subgraph, dim=self.n_components)
                except:
                    sub_positions = nx.spring_layout(subgraph)
            else:
                # Simple layout for small components
                sub_positions = nx.circular_layout(subgraph)
            
            # Offset each component to avoid overlap
            offset_x = (i % 3) * 3.0
            offset_y = (i // 3) * 3.0
            
            for node, (x, y) in sub_positions.items():
                positions[node] = (x + offset_x, y + offset_y)
        
        return positions
    
    def _manual_spectral_layout(self, graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Manual spectral embedding implementation."""
        try:
            # Get adjacency matrix
            adjacency = nx.adjacency_matrix(graph).toarray()
            
            # Spectral embedding
            embedding = SpectralEmbedding(
                n_components=self.n_components,
                random_state=self.random_state
            )
            positions_array = embedding.fit_transform(adjacency)
            
            # Convert to dictionary
            positions = {}
            nodes = list(graph.nodes())
            for i, node in enumerate(nodes):
                if self.n_components == 2:
                    positions[node] = tuple(positions_array[i])
                else:
                    positions[node] = tuple(positions_array[i][:2])  # Take first 2 dimensions
            
            return positions
        except:
            # Final fallback
            return nx.spring_layout(graph)


class TopologyPreservingLayout:
    """
    Advanced layout that preserves topological properties of the concept space.
    
    This layout combines multiple techniques to create visualizations that
    maintain both local neighborhoods and global structure.
    """
    
    def __init__(self, method: str = 'hybrid', perplexity: float = 30.0):
        self.method = method
        self.perplexity = perplexity
    
    def compute_layout(self, 
                      graph: nx.Graph,
                      distance_matrix: torch.Tensor,
                      concept_representations: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """
        Compute topology-preserving layout.
        
        Args:
            graph: NetworkX graph
            distance_matrix: Distance matrix between concepts
            concept_representations: High-dimensional concept representations
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if self.method == 'mds':
            return self._mds_layout(distance_matrix)
        elif self.method == 'tsne':
            return self._tsne_layout(concept_representations)
        elif self.method == 'hybrid':
            return self._hybrid_layout(graph, distance_matrix, concept_representations)
        elif self.method == 'umap':
            return self._umap_layout(concept_representations)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mds_layout(self, distance_matrix: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """Multi-dimensional scaling layout."""
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions_array = mds.fit_transform(distance_matrix.numpy())
        
        positions = {}
        for i, (x, y) in enumerate(positions_array):
            positions[i] = (x, y)
        
        return positions
    
    def _tsne_layout(self, representations: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """t-SNE layout for preserving local neighborhoods."""
        tsne = TSNE(n_components=2, perplexity=min(self.perplexity, len(representations)-1), random_state=42)
        positions_array = tsne.fit_transform(representations.numpy())
        
        positions = {}
        for i, (x, y) in enumerate(positions_array):
            positions[i] = (x, y)
        
        return positions
    
    def _hybrid_layout(self, 
                      graph: nx.Graph,
                      distance_matrix: torch.Tensor,
                      representations: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """Hybrid layout combining multiple techniques."""
        # Start with MDS for global structure
        mds_positions = self._mds_layout(distance_matrix)
        
        # Refine with force-directed layout
        force_layout = ForceDirectedLayout(iterations=500)
        final_positions = force_layout.compute_layout(graph, distance_matrix, mds_positions)
        
        return final_positions
    
    def _umap_layout(self, representations: torch.Tensor) -> Dict[int, Tuple[float, float]]:
        """UMAP layout (if available)."""
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            positions_array = reducer.fit_transform(representations.numpy())
            
            positions = {}
            for i, (x, y) in enumerate(positions_array):
                positions[i] = (x, y)
            
            return positions
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            return self._tsne_layout(representations)


def compare_layout_quality(layouts: Dict[str, Dict[int, Tuple[float, float]]],
                          distance_matrix: torch.Tensor,
                          graph: nx.Graph) -> Dict[str, Dict[str, float]]:
    """
    Compare quality of different layouts.
    
    Args:
        layouts: Dictionary mapping layout names to position dictionaries
        distance_matrix: Target distance matrix
        graph: NetworkX graph
        
    Returns:
        Dictionary of quality metrics for each layout
    """
    quality_metrics = {}
    
    for layout_name, positions in layouts.items():
        metrics = {}
        
        # Extract position arrays
        nodes = sorted(positions.keys())
        pos_array = np.array([positions[node] for node in nodes])
        
        # Compute layout distances
        layout_distances = squareform(pdist(pos_array))
        
        # Distance preservation (Stress)
        target_distances = distance_matrix.numpy()
        stress = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                target_dist = target_distances[node_i, node_j]
                layout_dist = layout_distances[i, j]
                stress += (target_dist - layout_dist) ** 2
        
        metrics['stress'] = stress / (len(nodes) * (len(nodes) - 1) / 2)
        
        # Distance correlation
        target_flat = []
        layout_flat = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                target_flat.append(target_distances[nodes[i], nodes[j]])
                layout_flat.append(layout_distances[i, j])
        
        correlation = np.corrcoef(target_flat, layout_flat)[0, 1]
        metrics['distance_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Edge preservation (how well connected nodes are close in layout)
        edge_distances = []
        non_edge_distances = []
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                layout_dist = layout_distances[i, j]
                
                if graph.has_edge(node_i, node_j):
                    edge_distances.append(layout_dist)
                else:
                    non_edge_distances.append(layout_dist)
        
        if edge_distances and non_edge_distances:
            edge_ratio = np.mean(edge_distances) / np.mean(non_edge_distances)
            metrics['edge_preservation'] = 1 - edge_ratio  # Lower is better
        else:
            metrics['edge_preservation'] = 0.0
        
        quality_metrics[layout_name] = metrics
    
    return quality_metrics


# Example usage and testing
if __name__ == "__main__":
    # Create test graph
    n_nodes = 8
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    
    # Create graph
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add circular edges
    for i in range(n_nodes):
        G.add_edge(i, (i + 1) % n_nodes)
    
    # Create distance matrix
    positions_3d = np.array([[np.cos(a), np.sin(a), 0] for a in angles])
    distance_matrix = torch.tensor(squareform(pdist(positions_3d)))
    concept_representations = torch.tensor(positions_3d, dtype=torch.float32)
    
    print("Testing layout algorithms...")
    
    # Test different layouts
    layouts = {}
    
    # Circular layout
    circular_layout = CircularLayout()
    layouts['circular'] = circular_layout.compute_layout(G, distance_matrix=distance_matrix)
    print(f"Circular layout: {len(layouts['circular'])} positions")
    
    # Force-directed layout
    force_layout = ForceDirectedLayout(iterations=100)
    layouts['force'] = force_layout.compute_layout(G, distance_matrix)
    print(f"Force-directed layout: {len(layouts['force'])} positions")
    
    # Spectral layout
    spectral_layout = SpectralLayout()
    layouts['spectral'] = spectral_layout.compute_layout(G)
    print(f"Spectral layout: {len(layouts['spectral'])} positions")
    
    # Topology-preserving layout
    topo_layout = TopologyPreservingLayout(method='mds')
    layouts['topology'] = topo_layout.compute_layout(G, distance_matrix, concept_representations)
    print(f"Topology-preserving layout: {len(layouts['topology'])} positions")
    
    # Compare quality
    quality = compare_layout_quality(layouts, distance_matrix, G)
    print(f"\nLayout quality comparison:")
    for layout_name, metrics in quality.items():
        print(f"  {layout_name}:")
        print(f"    Stress: {metrics['stress']:.3f}")
        print(f"    Distance correlation: {metrics['distance_correlation']:.3f}")
        print(f"    Edge preservation: {metrics['edge_preservation']:.3f}")
    
    print("✅ Layout algorithms implementation complete!")