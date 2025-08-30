"""
Graph validation metrics for concept topology analysis.

This module provides comprehensive metrics for evaluating how well
concept graphs preserve the underlying mathematical structure.
"""

import torch
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


class GraphValidator:
    """
    Comprehensive graph validation for concept topology analysis.
    
    This class evaluates how well concept graphs preserve mathematical
    structure, local neighborhoods, and global topology.
    """
    
    def __init__(self, expected_structure: str = 'circular'):
        self.expected_structure = expected_structure
        self.validation_results = {}
    
    def validate_graph(self,
                      graph: nx.Graph,
                      distance_matrix: torch.Tensor,
                      positions: Dict[int, Tuple[float, float]],
                      true_labels: Optional[torch.Tensor] = None,
                      p: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive graph validation.
        
        Args:
            graph: NetworkX graph to validate
            distance_matrix: Distance matrix between concepts
            positions: Node positions for layout validation
            true_labels: Ground truth labels
            p: Expected modulus (auto-detected if None)
            
        Returns:
            Dictionary with comprehensive validation metrics
        """
        results = {}
        
        # Basic graph metrics
        results['basic_metrics'] = self._compute_basic_metrics(graph)
        
        # Topology preservation
        results['topology_preservation'] = self._validate_topology_preservation(
            graph, distance_matrix, positions
        )
        
        # Structure-specific validation
        if self.expected_structure == 'circular' and true_labels is not None:
            if p is None:
                p = len(torch.unique(true_labels))
            results['circular_structure'] = self._validate_circular_structure(
                graph, distance_matrix, positions, true_labels, p
            )
        
        # Layout quality
        results['layout_quality'] = self._validate_layout_quality(
            positions, distance_matrix
        )
        
        # Neighborhood preservation
        results['neighborhood_preservation'] = self._validate_neighborhood_preservation(
            graph, distance_matrix
        )
        
        # Overall assessment
        results['overall_assessment'] = self._compute_overall_assessment(results)
        
        self.validation_results = results
        return results
    
    def _compute_basic_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Compute basic graph topology metrics."""
        metrics = {}
        
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        metrics['n_nodes'] = n_nodes
        metrics['n_edges'] = n_edges
        metrics['density'] = nx.density(graph)
        metrics['is_connected'] = nx.is_connected(graph)
        
        if n_nodes > 0:
            degrees = dict(graph.degree())
            metrics['avg_degree'] = np.mean(list(degrees.values()))
            metrics['degree_std'] = np.std(list(degrees.values()))
            metrics['min_degree'] = min(degrees.values())
            metrics['max_degree'] = max(degrees.values())
        
        # Connectivity metrics
        if metrics['is_connected'] and n_nodes > 1:
            try:
                metrics['diameter'] = nx.diameter(graph)
                metrics['avg_path_length'] = nx.average_shortest_path_length(graph)
                metrics['radius'] = nx.radius(graph)
            except:
                metrics['diameter'] = None
                metrics['avg_path_length'] = None
                metrics['radius'] = None
        else:
            metrics['n_components'] = nx.number_connected_components(graph)
            if n_nodes > 0:
                components = list(nx.connected_components(graph))
                metrics['largest_component_size'] = len(max(components, key=len))
        
        # Clustering metrics
        if n_nodes > 2:
            try:
                metrics['avg_clustering'] = nx.average_clustering(graph)
                metrics['transitivity'] = nx.transitivity(graph)
            except:
                metrics['avg_clustering'] = 0.0
                metrics['transitivity'] = 0.0
        
        return metrics
    
    def _validate_topology_preservation(self,
                                      graph: nx.Graph,
                                      distance_matrix: torch.Tensor,
                                      positions: Dict[int, Tuple[float, float]]) -> Dict[str, Any]:
        """Validate how well graph preserves topological relationships."""
        metrics = {}
        
        # Graph distance vs concept distance correlation
        graph_distances = self._compute_graph_distances(graph)
        concept_distances = distance_matrix.numpy()
        
        # Flatten distance matrices for correlation
        n_nodes = len(graph.nodes())
        graph_dist_flat = []
        concept_dist_flat = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if i in graph_distances and j in graph_distances[i]:
                    graph_dist_flat.append(graph_distances[i][j])
                    concept_dist_flat.append(concept_distances[i, j])
        
        if len(graph_dist_flat) > 1:
            try:
                pearson_corr, _ = pearsonr(graph_dist_flat, concept_dist_flat)
                spearman_corr, _ = spearmanr(graph_dist_flat, concept_dist_flat)
                metrics['graph_concept_pearson'] = pearson_corr
                metrics['graph_concept_spearman'] = spearman_corr
            except:
                metrics['graph_concept_pearson'] = 0.0
                metrics['graph_concept_spearman'] = 0.0
        else:
            metrics['graph_concept_pearson'] = 0.0
            metrics['graph_concept_spearman'] = 0.0
        
        # Edge weight vs concept distance correlation
        edge_weights = []
        edge_concept_distances = []
        
        for u, v, data in graph.edges(data=True):
            weight = data.get('distance', data.get('weight', 1.0))
            edge_weights.append(weight)
            edge_concept_distances.append(concept_distances[u, v])
        
        if len(edge_weights) > 1:
            try:
                edge_corr, _ = pearsonr(edge_weights, edge_concept_distances)
                metrics['edge_weight_correlation'] = edge_corr
            except:
                metrics['edge_weight_correlation'] = 0.0
        else:
            metrics['edge_weight_correlation'] = 0.0
        
        return metrics
    
    def _compute_graph_distances(self, graph: nx.Graph) -> Dict[int, Dict[int, float]]:
        """Compute shortest path distances in graph."""
        try:
            if nx.is_connected(graph):
                return dict(nx.all_pairs_shortest_path_length(graph))
            else:
                # Handle disconnected graphs
                distances = {}
                for component in nx.connected_components(graph):
                    subgraph = graph.subgraph(component)
                    component_distances = dict(nx.all_pairs_shortest_path_length(subgraph))
                    distances.update(component_distances)
                return distances
        except:
            return {}
    
    def _validate_circular_structure(self,
                                   graph: nx.Graph,
                                   distance_matrix: torch.Tensor,
                                   positions: Dict[int, Tuple[float, float]],
                                   true_labels: torch.Tensor,
                                   p: int) -> Dict[str, Any]:
        """Validate circular structure preservation."""
        metrics = {}
        
        # Adjacency preservation
        expected_adjacencies = [(i, (i + 1) % p) for i in range(p)]
        preserved_adjacencies = 0
        total_adjacencies = 0
        
        for a, b in expected_adjacencies:
            # Find nodes with these true labels
            nodes_a = [i for i, label in enumerate(true_labels) if label.item() == a]
            nodes_b = [i for i, label in enumerate(true_labels) if label.item() == b]
            
            total_adjacencies += len(nodes_a) * len(nodes_b)
            
            for node_a in nodes_a:
                for node_b in nodes_b:
                    if graph.has_edge(node_a, node_b):
                        preserved_adjacencies += 1
        
        metrics['adjacency_preservation'] = preserved_adjacencies / max(total_adjacencies, 1)
        
        # Circular distance correlation
        expected_circular_distances = torch.zeros(p, p)
        for i in range(p):
            for j in range(p):
                clockwise = (j - i) % p
                counterclockwise = (i - j) % p
                expected_circular_distances[i, j] = min(clockwise, counterclockwise)
        
        # Map learned distances to expected distances
        learned_distances = []
        expected_distances_flat = []
        
        n_concepts = len(true_labels)
        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                true_i = true_labels[i].item()
                true_j = true_labels[j].item()
                
                learned_dist = distance_matrix[i, j].item()
                expected_dist = expected_circular_distances[true_i, true_j].item()
                
                learned_distances.append(learned_dist)
                expected_distances_flat.append(expected_dist)
        
        if len(learned_distances) > 1:
            try:
                circular_corr, _ = pearsonr(learned_distances, expected_distances_flat)
                metrics['circular_distance_correlation'] = circular_corr
            except:
                metrics['circular_distance_correlation'] = 0.0
        else:
            metrics['circular_distance_correlation'] = 0.0
        
        # Angular consistency in layout
        if len(positions) == p:
            layout_angles = []
            expected_angles = []
            
            for i in range(p):
                # Find node with true label i
                nodes_with_label = [idx for idx, label in enumerate(true_labels) if label.item() == i]
                if nodes_with_label:
                    node = nodes_with_label[0]
                    if node in positions:
                        x, y = positions[node]
                        angle = np.arctan2(y, x)
                        if angle < 0:
                            angle += 2 * np.pi
                        layout_angles.append(angle)
                        expected_angles.append(2 * np.pi * i / p)
            
            if len(layout_angles) == p:
                # Sort both by expected angles to align
                paired_angles = list(zip(expected_angles, layout_angles))
                paired_angles.sort(key=lambda x: x[0])
                expected_sorted, layout_sorted = zip(*paired_angles)
                
                # Compute angular correlation
                try:
                    angular_corr, _ = pearsonr(expected_sorted, layout_sorted)
                    metrics['angular_correlation'] = angular_corr
                except:
                    metrics['angular_correlation'] = 0.0
            else:
                metrics['angular_correlation'] = 0.0
        else:
            metrics['angular_correlation'] = 0.0
        
        # Overall circular structure score
        components = [
            metrics['adjacency_preservation'],
            max(0, metrics['circular_distance_correlation']),
            max(0, metrics.get('angular_correlation', 0))
        ]
        metrics['overall_circular_score'] = np.mean(components)
        
        return metrics
    
    def _validate_layout_quality(self,
                               positions: Dict[int, Tuple[float, float]],
                               distance_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate layout quality metrics."""
        metrics = {}
        
        if not positions:
            return {'stress': float('inf'), 'distance_correlation': 0.0}
        
        # Extract position arrays
        nodes = sorted(positions.keys())
        pos_array = np.array([positions[node] for node in nodes])
        
        # Compute layout distances
        from scipy.spatial.distance import pdist, squareform
        layout_distances = squareform(pdist(pos_array))
        
        # Stress (normalized mean squared error)
        concept_distances = distance_matrix.numpy()
        
        stress = 0.0
        count = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                if node_i < len(concept_distances) and node_j < len(concept_distances):
                    concept_dist = concept_distances[node_i, node_j]
                    layout_dist = layout_distances[i, j]
                    stress += (concept_dist - layout_dist) ** 2
                    count += 1
        
        metrics['stress'] = stress / max(count, 1)
        
        # Distance correlation
        concept_flat = []
        layout_flat = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                if node_i < len(concept_distances) and node_j < len(concept_distances):
                    concept_flat.append(concept_distances[node_i, node_j])
                    layout_flat.append(layout_distances[i, j])
        
        if len(concept_flat) > 1:
            try:
                layout_corr, _ = pearsonr(concept_flat, layout_flat)
                metrics['distance_correlation'] = layout_corr
            except:
                metrics['distance_correlation'] = 0.0
        else:
            metrics['distance_correlation'] = 0.0
        
        # Layout spread (how well distributed are the points)
        if len(pos_array) > 1:
            distances_from_center = np.linalg.norm(pos_array - np.mean(pos_array, axis=0), axis=1)
            metrics['layout_spread_std'] = np.std(distances_from_center)
            metrics['layout_spread_mean'] = np.mean(distances_from_center)
        else:
            metrics['layout_spread_std'] = 0.0
            metrics['layout_spread_mean'] = 0.0
        
        return metrics
    
    def _validate_neighborhood_preservation(self,
                                          graph: nx.Graph,
                                          distance_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate how well local neighborhoods are preserved."""
        metrics = {}
        
        concept_distances = distance_matrix.numpy()
        n_nodes = len(graph.nodes())
        
        if n_nodes < 3:
            return {'neighborhood_preservation': 0.0}
        
        # For each node, check if its graph neighbors are close in concept space
        preservation_scores = []
        
        for node in graph.nodes():
            if node >= len(concept_distances):
                continue
                
            graph_neighbors = set(graph.neighbors(node))
            
            if len(graph_neighbors) == 0:
                continue
            
            # Get distances to all other nodes in concept space
            node_distances = [(i, concept_distances[node, i]) for i in range(n_nodes) if i != node]
            node_distances.sort(key=lambda x: x[1])  # Sort by distance
            
            # Get k nearest neighbors in concept space (k = number of graph neighbors)
            k = min(len(graph_neighbors), len(node_distances))
            concept_neighbors = set([node_distances[i][0] for i in range(k)])
            
            # Compute overlap
            overlap = len(graph_neighbors.intersection(concept_neighbors))
            preservation_score = overlap / k if k > 0 else 0
            preservation_scores.append(preservation_score)
        
        metrics['neighborhood_preservation'] = np.mean(preservation_scores) if preservation_scores else 0.0
        metrics['neighborhood_preservation_std'] = np.std(preservation_scores) if preservation_scores else 0.0
        
        return metrics
    
    def _compute_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall assessment score and quality rating."""
        assessment = {}
        
        # Collect key metrics
        scores = []
        weights = []
        
        # Topology preservation (weight: 0.3)
        topo = results.get('topology_preservation', {})
        if 'graph_concept_pearson' in topo:
            scores.append(max(0, topo['graph_concept_pearson']))
            weights.append(0.3)
        
        # Layout quality (weight: 0.25)
        layout = results.get('layout_quality', {})
        if 'distance_correlation' in layout:
            scores.append(max(0, layout['distance_correlation']))
            weights.append(0.25)
        
        # Neighborhood preservation (weight: 0.25)
        neighborhood = results.get('neighborhood_preservation', {})
        if 'neighborhood_preservation' in neighborhood:
            scores.append(neighborhood['neighborhood_preservation'])
            weights.append(0.25)
        
        # Structure-specific metrics (weight: 0.2)
        if 'circular_structure' in results:
            circular = results['circular_structure']
            if 'overall_circular_score' in circular:
                scores.append(circular['overall_circular_score'])
                weights.append(0.2)
        
        # Compute weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            assessment['overall_score'] = weighted_score
        else:
            assessment['overall_score'] = 0.0
        
        # Quality rating
        score = assessment['overall_score']
        if score >= 0.8:
            assessment['quality_rating'] = 'Excellent'
        elif score >= 0.6:
            assessment['quality_rating'] = 'Good'
        elif score >= 0.4:
            assessment['quality_rating'] = 'Fair'
        elif score >= 0.2:
            assessment['quality_rating'] = 'Poor'
        else:
            assessment['quality_rating'] = 'Very Poor'
        
        # Recommendations
        assessment['recommendations'] = self._generate_recommendations(results)
        
        return assessment
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check topology preservation
        topo = results.get('topology_preservation', {})
        if topo.get('graph_concept_pearson', 0) < 0.5:
            recommendations.append("Consider using a different graph construction method to better preserve concept relationships")
        
        # Check layout quality
        layout = results.get('layout_quality', {})
        if layout.get('stress', float('inf')) > 1.0:
            recommendations.append("Layout has high stress - try a different layout algorithm or adjust parameters")
        
        if layout.get('distance_correlation', 0) < 0.5:
            recommendations.append("Layout poorly preserves distances - consider MDS or force-directed layout")
        
        # Check neighborhood preservation
        neighborhood = results.get('neighborhood_preservation', {})
        if neighborhood.get('neighborhood_preservation', 0) < 0.5:
            recommendations.append("Local neighborhoods not well preserved - adjust graph construction parameters")
        
        # Check circular structure (if applicable)
        if 'circular_structure' in results:
            circular = results['circular_structure']
            if circular.get('adjacency_preservation', 0) < 0.7:
                recommendations.append("Circular adjacencies not well preserved - model may not have learned proper circular structure")
            
            if circular.get('circular_distance_correlation', 0) < 0.5:
                recommendations.append("Distance relationships don't match circular structure - check model training")
        
        # Basic connectivity
        basic = results.get('basic_metrics', {})
        if not basic.get('is_connected', True) and basic.get('n_nodes', 0) > 1:
            recommendations.append("Graph is disconnected - consider increasing connection threshold or using different method")
        
        if len(recommendations) == 0:
            recommendations.append("Graph structure looks good - proceed with analysis")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Create test data
    n_nodes = 6
    
    # Create circular graph
    G = nx.cycle_graph(n_nodes)
    
    # Create distance matrix (circular distances)
    distance_matrix = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            diff = abs(i - j)
            circular_dist = min(diff, n_nodes - diff)
            distance_matrix[i, j] = circular_dist
    
    # Create positions (circular layout)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n_nodes)}
    
    # Create true labels
    true_labels = torch.arange(n_nodes)
    
    print("Testing graph validation...")
    
    # Test validator
    validator = GraphValidator('circular')
    results = validator.validate_graph(
        G, distance_matrix, positions, true_labels, n_nodes
    )
    
    print(f"\nValidation Results:")
    print(f"Overall score: {results['overall_assessment']['overall_score']:.3f}")
    print(f"Quality rating: {results['overall_assessment']['quality_rating']}")
    
    # Print detailed metrics
    for category, metrics in results.items():
        if category != 'overall_assessment':
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.3f}")
                elif isinstance(value, bool):
                    print(f"  {metric}: {value}")
    
    print("\nRecommendations:")
    for rec in results['overall_assessment']['recommendations']:
        print(f"  - {rec}")
    
    print("✅ Graph metrics implementation complete!")