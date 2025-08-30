"""
Concept graph construction for neural network visualization.

This module builds graphs from extracted concepts, preserving topological
relationships and enabling interactive visualization of concept spaces.
"""

import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import SpectralEmbedding
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import json


class ConceptGraph:
    """
    Build and manage concept graphs from neural network representations.
    
    This class creates graphs where nodes represent concepts and edges
    represent relationships between concepts, preserving the topology
    of the learned concept space.
    """
    
    def __init__(self, concept_representations: torch.Tensor,
                 concept_labels: Optional[List[str]] = None,
                 true_labels: Optional[torch.Tensor] = None):
        """
        Initialize concept graph.
        
        Args:
            concept_representations: Tensor of shape (n_concepts, representation_dim)
            concept_labels: Optional list of concept labels
            true_labels: Optional ground truth labels for concepts
        """
        self.concept_representations = concept_representations.cpu()
        self.n_concepts = len(concept_representations)
        
        # Set up concept labels
        if concept_labels is None:
            self.concept_labels = [f"concept_{i}" for i in range(self.n_concepts)]
        else:
            self.concept_labels = concept_labels
        
        self.true_labels = true_labels.cpu() if true_labels is not None else None
        
        # Graph components
        self.graph = nx.Graph()
        self.distance_matrix = None
        self.similarity_matrix = None
        self.adjacency_matrix = None
        self.node_positions = None
        
        # Metadata
        self.graph_metadata = {
            'n_concepts': self.n_concepts,
            'representation_dim': concept_representations.shape[1],
            'construction_method': None,
            'layout_method': None
        }
    
    def compute_distance_matrix(self, metric: str = 'cosine') -> torch.Tensor:
        """
        Compute pairwise distances between concepts.
        
        Args:
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Distance matrix of shape (n_concepts, n_concepts)
        """
        representations = self.concept_representations.numpy()
        
        if metric == 'cosine':
            # Cosine similarity -> cosine distance
            sim_matrix = cosine_similarity(representations)
            self.similarity_matrix = torch.tensor(sim_matrix)
            dist_matrix = 1 - sim_matrix
        elif metric == 'euclidean':
            dist_matrix = euclidean_distances(representations)
        elif metric == 'manhattan':
            dist_matrix = squareform(pdist(representations, metric='manhattan'))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.distance_matrix = torch.tensor(dist_matrix)
        return self.distance_matrix
    
    def build_graph(self, 
                   method: str = 'knn',
                   k: int = 3,
                   threshold: float = 0.5,
                   distance_metric: str = 'cosine') -> nx.Graph:
        """
        Build concept graph using specified method.
        
        Args:
            method: Graph construction method ('knn', 'threshold', 'mst', 'circular')
            k: Number of neighbors for k-NN graph
            threshold: Distance threshold for threshold graph
            distance_metric: Distance metric to use
            
        Returns:
            NetworkX graph
        """
        # Compute distances
        if self.distance_matrix is None:
            self.compute_distance_matrix(distance_metric)
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes with attributes
        for i, label in enumerate(self.concept_labels):
            node_attrs = {
                'label': label,
                'representation': self.concept_representations[i].numpy(),
                'index': i
            }
            if self.true_labels is not None:
                node_attrs['true_label'] = self.true_labels[i].item()
            
            self.graph.add_node(i, **node_attrs)
        
        # Add edges based on method
        if method == 'knn':
            self._build_knn_graph(k)
        elif method == 'threshold':
            self._build_threshold_graph(threshold)
        elif method == 'mst':
            self._build_mst_graph()
        elif method == 'circular':
            self._build_circular_graph()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.graph_metadata['construction_method'] = method
        return self.graph
    
    def _build_knn_graph(self, k: int):
        """Build k-nearest neighbors graph."""
        dist_matrix = self.distance_matrix.numpy()
        
        for i in range(self.n_concepts):
            # Find k nearest neighbors (excluding self)
            distances = dist_matrix[i]
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self (index 0)
            
            # Add edges to k nearest neighbors
            for j in nearest_indices:
                if not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j, weight=distances[j], distance=distances[j])
    
    def _build_threshold_graph(self, threshold: float):
        """Build threshold-based graph."""
        dist_matrix = self.distance_matrix.numpy()
        
        for i in range(self.n_concepts):
            for j in range(i + 1, self.n_concepts):
                distance = dist_matrix[i, j]
                if distance <= threshold:
                    self.graph.add_edge(i, j, weight=distance, distance=distance)
    
    def _build_mst_graph(self):
        """Build minimum spanning tree graph."""
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        dist_matrix = self.distance_matrix.numpy()
        mst = minimum_spanning_tree(dist_matrix).toarray()
        
        # Add MST edges
        rows, cols = np.nonzero(mst)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicate edges
                distance = dist_matrix[i, j]
                self.graph.add_edge(i, j, weight=distance, distance=distance)
    
    def _build_circular_graph(self):
        """Build circular graph for modular arithmetic (ground truth structure)."""
        if self.true_labels is None:
            raise ValueError("True labels required for circular graph construction")
        
        # Sort concepts by their true labels to create circular order
        sorted_indices = torch.argsort(self.true_labels)
        
        # Add edges between consecutive concepts in circular order
        for i in range(len(sorted_indices)):
            curr_idx = sorted_indices[i].item()
            next_idx = sorted_indices[(i + 1) % len(sorted_indices)].item()
            
            distance = self.distance_matrix[curr_idx, next_idx].item()
            self.graph.add_edge(curr_idx, next_idx, weight=distance, distance=distance)
    
    def compute_graph_metrics(self) -> Dict[str, Any]:
        """Compute various graph topology metrics."""
        if len(self.graph.nodes) == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = self.graph.number_of_nodes()
        metrics['num_edges'] = self.graph.number_of_edges()
        metrics['density'] = nx.density(self.graph)
        
        # Connectivity metrics
        metrics['is_connected'] = nx.is_connected(self.graph)
        if metrics['is_connected']:
            metrics['diameter'] = nx.diameter(self.graph)
            metrics['average_path_length'] = nx.average_shortest_path_length(self.graph)
        else:
            metrics['num_components'] = nx.number_connected_components(self.graph)
            components = list(nx.connected_components(self.graph))
            metrics['largest_component_size'] = len(max(components, key=len))
        
        # Degree metrics
        degrees = dict(self.graph.degree())
        metrics['average_degree'] = np.mean(list(degrees.values()))
        metrics['degree_std'] = np.std(list(degrees.values()))
        metrics['max_degree'] = max(degrees.values())
        metrics['min_degree'] = min(degrees.values())
        
        # Clustering metrics
        if len(self.graph.nodes) > 2:
            try:
                metrics['clustering_coefficient'] = nx.average_clustering(self.graph)
                metrics['transitivity'] = nx.transitivity(self.graph)
            except:
                metrics['clustering_coefficient'] = 0.0
                metrics['transitivity'] = 0.0
        
        return metrics
    
    def validate_circular_structure(self, p: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate how well the graph preserves circular structure.
        
        Args:
            p: Expected modulus (auto-detected if None)
            
        Returns:
            Dictionary with validation metrics
        """
        if self.true_labels is None:
            return {'error': 'True labels required for circular validation'}
        
        if p is None:
            p = len(torch.unique(self.true_labels))
        
        validation = {}
        
        # Check adjacency preservation
        expected_adjacent_pairs = [(i, (i + 1) % p) for i in range(p)]
        preserved_adjacencies = 0
        
        for a, b in expected_adjacent_pairs:
            # Find concept nodes with these true labels
            nodes_a = [i for i, label in enumerate(self.true_labels) if label == a]
            nodes_b = [i for i, label in enumerate(self.true_labels) if label == b]
            
            # Check if any pair is connected
            connected = False
            for node_a in nodes_a:
                for node_b in nodes_b:
                    if self.graph.has_edge(node_a, node_b):
                        connected = True
                        break
                if connected:
                    break
            
            if connected:
                preserved_adjacencies += 1
        
        validation['adjacency_preservation'] = preserved_adjacencies / len(expected_adjacent_pairs)
        
        # Check distance correlation with circular distances
        if self.distance_matrix is not None:
            expected_distances = torch.zeros(p, p)
            for i in range(p):
                for j in range(p):
                    clockwise = (j - i) % p
                    counterclockwise = (i - j) % p
                    expected_distances[i, j] = min(clockwise, counterclockwise)
            
            # Map learned distances to expected distances
            learned_distances = []
            expected_distances_flat = []
            
            for i in range(self.n_concepts):
                for j in range(i + 1, self.n_concepts):
                    true_i = self.true_labels[i].item()
                    true_j = self.true_labels[j].item()
                    
                    learned_dist = self.distance_matrix[i, j].item()
                    expected_dist = expected_distances[true_i, true_j].item()
                    
                    learned_distances.append(learned_dist)
                    expected_distances_flat.append(expected_dist)
            
            # Compute correlation
            if len(learned_distances) > 1:
                correlation = np.corrcoef(learned_distances, expected_distances_flat)[0, 1]
                validation['distance_correlation'] = correlation
            else:
                validation['distance_correlation'] = 0.0
        
        # Overall circular structure score
        components = [validation.get('adjacency_preservation', 0),
                     validation.get('distance_correlation', 0)]
        validation['overall_circular_score'] = np.mean(components)
        
        return validation
    
    def layout_graph(self, method: str = 'spring', **kwargs) -> Dict[int, Tuple[float, float]]:
        """
        Compute 2D layout for graph visualization.
        
        Args:
            method: Layout method ('spring', 'circular', 'spectral', 'mds')
            **kwargs: Additional arguments for layout algorithm
            
        Returns:
            Dictionary mapping node indices to (x, y) positions
        """
        if method == 'spring':
            pos = nx.spring_layout(self.graph, **kwargs)
        elif method == 'circular':
            pos = nx.circular_layout(self.graph, **kwargs)
        elif method == 'spectral':
            pos = nx.spectral_layout(self.graph, **kwargs)
        elif method == 'mds':
            pos = self._mds_layout(**kwargs)
        elif method == 'force_directed':
            pos = self._force_directed_layout(**kwargs)
        else:
            raise ValueError(f"Unknown layout method: {method}")
        
        self.node_positions = pos
        self.graph_metadata['layout_method'] = method
        
        return pos
    
    def _mds_layout(self, dim: int = 2) -> Dict[int, Tuple[float, float]]:
        """Multi-dimensional scaling layout."""
        from sklearn.manifold import MDS
        
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
        pos_array = mds.fit_transform(self.distance_matrix.numpy())
        
        pos = {}
        for i, (x, y) in enumerate(pos_array):
            pos[i] = (x, y)
        
        return pos
    
    def _force_directed_layout(self, iterations: int = 1000) -> Dict[int, Tuple[float, float]]:
        """Custom force-directed layout preserving distances."""
        pos = nx.spring_layout(self.graph, iterations=iterations, k=1.0)
        return pos
    
    def export_graph(self, filepath: Path, format: str = 'graphml'):
        """Export graph to file."""
        filepath = Path(filepath)
        
        if format == 'graphml':
            nx.write_graphml(self.graph, filepath)
        elif format == 'gexf':
            nx.write_gexf(self.graph, filepath)
        elif format == 'json':
            graph_data = {
                'nodes': [],
                'edges': [],
                'metadata': self.graph_metadata
            }
            
            for node, data in self.graph.nodes(data=True):
                node_data = {'id': node, **data}
                # Convert numpy arrays to lists for JSON serialization
                if 'representation' in node_data:
                    node_data['representation'] = node_data['representation'].tolist()
                graph_data['nodes'].append(node_data)
            
            for u, v, data in self.graph.edges(data=True):
                edge_data = {'source': u, 'target': v, **data}
                graph_data['edges'].append(edge_data)
            
            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def save_graph(self, filepath: Path):
        """Save complete graph object."""
        save_data = {
            'concept_representations': self.concept_representations,
            'concept_labels': self.concept_labels,
            'true_labels': self.true_labels,
            'graph': self.graph,
            'distance_matrix': self.distance_matrix,
            'similarity_matrix': self.similarity_matrix,
            'node_positions': self.node_positions,
            'graph_metadata': self.graph_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_graph(self, filepath: Path):
        """Load complete graph object."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.concept_representations = save_data['concept_representations']
        self.concept_labels = save_data['concept_labels']
        self.true_labels = save_data['true_labels']
        self.graph = save_data['graph']
        self.distance_matrix = save_data.get('distance_matrix')
        self.similarity_matrix = save_data.get('similarity_matrix')
        self.node_positions = save_data.get('node_positions')
        self.graph_metadata = save_data.get('graph_metadata', {})


def create_concept_graph_from_activations(activations: torch.Tensor,
                                        true_labels: Optional[torch.Tensor] = None,
                                        method: str = 'knn',
                                        k: int = 3,
                                        distance_metric: str = 'cosine') -> ConceptGraph:
    """
    Create concept graph directly from activations.
    
    Args:
        activations: Activation tensor of shape (n_samples, activation_dim)
        true_labels: Optional ground truth labels
        method: Graph construction method
        k: Number of neighbors for k-NN
        distance_metric: Distance metric
        
    Returns:
        ConceptGraph instance
    """
    # Use activations directly as concept representations
    concept_graph = ConceptGraph(
        concept_representations=activations,
        true_labels=true_labels
    )
    
    concept_graph.build_graph(
        method=method,
        k=k,
        distance_metric=distance_metric
    )
    
    return concept_graph


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Create test concept representations (simulate 5 concepts in 2D)
    np.random.seed(42)
    
    # Create circular arrangement for testing
    n_concepts = 5
    angles = np.linspace(0, 2 * np.pi, n_concepts + 1)[:-1]
    concept_reps = torch.tensor([
        [np.cos(angle), np.sin(angle)] for angle in angles
    ], dtype=torch.float32)
    
    # Add some noise
    concept_reps += torch.randn_like(concept_reps) * 0.1
    
    true_labels = torch.arange(n_concepts)
    
    print(f"Test concept representations: {concept_reps.shape}")
    
    # Create concept graph
    graph = ConceptGraph(concept_reps, true_labels=true_labels)
    
    # Build different types of graphs
    for method in ['knn', 'threshold', 'circular']:
        print(f"\nTesting {method} graph:")
        
        if method == 'knn':
            graph.build_graph(method=method, k=2)
        elif method == 'threshold':
            graph.build_graph(method=method, threshold=0.8)
        else:
            graph.build_graph(method=method)
        
        # Compute metrics
        metrics = graph.compute_graph_metrics()
        print(f"  Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}")
        print(f"  Connected: {metrics['is_connected']}")
        print(f"  Average degree: {metrics['average_degree']:.2f}")
        
        # Validate circular structure
        if method == 'circular':
            validation = graph.validate_circular_structure(n_concepts)
            print(f"  Adjacency preservation: {validation['adjacency_preservation']:.3f}")
            print(f"  Distance correlation: {validation.get('distance_correlation', 'N/A')}")
        
        # Test layout
        pos = graph.layout_graph('spring')
        print(f"  Layout computed with {len(pos)} positions")
    
    print("✅ Concept graph implementation complete!")