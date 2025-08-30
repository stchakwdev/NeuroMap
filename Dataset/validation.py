"""
Validation functions for testing circular structure in learned embeddings.

These functions test whether neural networks learn the expected circular
topology for modular arithmetic representations.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Any
from Dataset.config import CIRCULAR_STRUCTURE_THRESHOLD, ADJACENCY_DISTANCE_THRESHOLD

class CircularStructureValidator:
    """Validates whether learned embeddings form expected circular structure."""
    
    def __init__(self, p: int):
        self.p = p
        
    def validate_embeddings(self, embeddings: torch.Tensor, 
                          visualize: bool = True) -> Dict[str, Any]:
        """
        Complete validation of embedding circular structure.
        
        Args:
            embeddings: (p, embedding_dim) tensor of learned number representations
            visualize: Whether to create plots
            
        Returns:
            Dictionary with validation results and metrics
        """
        
        results = {}
        
        # Convert to numpy for analysis
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # 1. Test circular ordering
        results.update(self._test_circular_ordering(embeddings_np))
        
        # 2. Test distance consistency
        results.update(self._test_distance_consistency(embeddings, embeddings_np))
        
        # 3. Test adjacency relationships
        results.update(self._test_adjacency_structure(embeddings))
        
        # 4. Create visualizations if requested
        if visualize:
            results['visualizations'] = self._create_visualizations(embeddings_np)
        
        # 5. Overall assessment
        results['overall_assessment'] = self._assess_circular_structure(results)
        
        return results
    
    def _test_circular_ordering(self, embeddings_np: np.ndarray) -> Dict[str, Any]:
        """Test if embeddings can be arranged in circular order."""
        
        # Project to 2D if needed
        if embeddings_np.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_np)
        else:
            embeddings_2d = embeddings_np
        
        # Calculate angles from center
        center = np.mean(embeddings_2d, axis=0)
        centered = embeddings_2d - center
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        
        # Sort by angle to get circular order
        sorted_indices = np.argsort(angles)
        
        # Check if sorted order matches expected circular order
        # For perfect circular structure, sorted indices should be a rotation of [0,1,2,...,p-1]
        expected_orders = []
        for start in range(self.p):
            expected_order = [(start + i) % self.p for i in range(self.p)]
            expected_orders.append(expected_order)
        
        # Check if any rotation matches
        is_circular = False
        best_match_score = 0
        for expected_order in expected_orders:
            matches = sum(1 for i, j in zip(sorted_indices, expected_order) if i == j)
            match_score = matches / self.p
            if match_score > best_match_score:
                best_match_score = match_score
            if match_score > 0.8:  # Allow some tolerance
                is_circular = True
                break
        
        return {
            'circular_ordering': {
                'is_circular_order': is_circular,
                'best_match_score': best_match_score,
                'sorted_indices': sorted_indices.tolist(),
                'angles': angles.tolist(),
                'center': center.tolist()
            }
        }
    
    def _test_distance_consistency(self, embeddings: torch.Tensor, 
                                 embeddings_np: np.ndarray) -> Dict[str, Any]:
        """Test if distances between embeddings match expected circular distances."""
        
        # Calculate pairwise distances in embedding space
        embedding_distances = torch.cdist(embeddings, embeddings).cpu().numpy()
        
        # Expected circular distances
        expected_distances = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                clockwise = (j - i) % self.p
                counterclockwise = (i - j) % self.p
                expected_distances[i, j] = min(clockwise, counterclockwise)
        
        # Correlation between embedding distances and expected distances
        embedding_dist_flat = embedding_distances[np.triu_indices(self.p, k=1)]
        expected_dist_flat = expected_distances[np.triu_indices(self.p, k=1)]
        
        correlation = np.corrcoef(embedding_dist_flat, expected_dist_flat)[0, 1]
        
        # Test radius consistency (all points should be roughly same distance from center)
        if embeddings_np.shape[1] >= 2:
            center = np.mean(embeddings_np, axis=0)
            distances_from_center = np.linalg.norm(embeddings_np - center, axis=1)
            radius_variance = np.var(distances_from_center)
            mean_radius = np.mean(distances_from_center)
        else:
            radius_variance = float('nan')
            mean_radius = float('nan')
        
        return {
            'distance_consistency': {
                'distance_correlation': correlation,
                'radius_variance': radius_variance,
                'mean_radius': mean_radius,
                'embedding_distances': embedding_distances.tolist(),
                'expected_distances': expected_distances.tolist()
            }
        }
    
    def _test_adjacency_structure(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Test if adjacent numbers (i, i+1 mod p) are close in embedding space."""
        
        adjacent_distances = []
        non_adjacent_distances = []
        
        # Calculate distances between adjacent pairs
        for i in range(self.p):
            next_i = (i + 1) % self.p
            dist = torch.norm(embeddings[i] - embeddings[next_i]).item()
            adjacent_distances.append(dist)
        
        # Calculate distances between non-adjacent pairs for comparison
        for i in range(self.p):
            for j in range(self.p):
                if abs(i - j) % self.p not in [0, 1, self.p - 1]:  # Not self or adjacent
                    dist = torch.norm(embeddings[i] - embeddings[j]).item()
                    non_adjacent_distances.append(dist)
        
        # Statistics
        mean_adjacent_dist = np.mean(adjacent_distances)
        mean_non_adjacent_dist = np.mean(non_adjacent_distances)
        adjacent_variance = np.var(adjacent_distances)
        
        # Test if adjacent distances are consistently smaller
        adjacency_ratio = mean_adjacent_dist / mean_non_adjacent_dist if mean_non_adjacent_dist > 0 else float('inf')
        
        return {
            'adjacency_structure': {
                'mean_adjacent_distance': mean_adjacent_dist,
                'mean_non_adjacent_distance': mean_non_adjacent_dist,
                'adjacent_distance_variance': adjacent_variance,
                'adjacency_ratio': adjacency_ratio,  # Should be < 1 for good structure
                'adjacent_distances': adjacent_distances,
                'passes_adjacency_test': adjacency_ratio < 0.8  # Heuristic threshold
            }
        }
    
    def _create_visualizations(self, embeddings_np: np.ndarray) -> Dict[str, Any]:
        """Create visualization plots for the embeddings."""
        
        visualizations = {}
        
        # 2D PCA projection
        if embeddings_np.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_np)
        else:
            embeddings_2d = embeddings_np
        
        # Create circular reference
        angles = np.linspace(0, 2 * np.pi, self.p, endpoint=False)
        circle_x = np.cos(angles)
        circle_y = np.sin(angles)
        
        visualizations['pca_projection'] = {
            'embeddings_2d': embeddings_2d.tolist(),
            'reference_circle': {
                'x': circle_x.tolist(),
                'y': circle_y.tolist()
            },
            'labels': list(range(self.p))
        }
        
        return visualizations
    
    def _assess_circular_structure(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide overall assessment of circular structure quality."""
        
        # Collect key metrics
        is_circular = results['circular_ordering']['is_circular_order']
        distance_corr = results['distance_consistency']['distance_correlation']
        passes_adjacency = results['adjacency_structure']['passes_adjacency_test']
        adjacency_ratio = results['adjacency_structure']['adjacency_ratio']
        
        # Overall score (0-1)
        score = 0.0
        if is_circular:
            score += 0.4
        if distance_corr > 0.5:
            score += 0.3
        if passes_adjacency:
            score += 0.3
        
        # Qualitative assessment
        if score >= 0.8:
            quality = "Excellent - Clear circular structure"
        elif score >= 0.6:
            quality = "Good - Recognizable circular structure"
        elif score >= 0.4:
            quality = "Fair - Some circular properties"
        else:
            quality = "Poor - No clear circular structure"
        
        return {
            'overall_score': score,
            'quality_assessment': quality,
            'key_metrics': {
                'circular_ordering': is_circular,
                'distance_correlation': distance_corr,
                'adjacency_test_passed': passes_adjacency,
                'adjacency_ratio': adjacency_ratio
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        if not results['circular_ordering']['is_circular_order']:
            recommendations.append("Model did not learn circular ordering - try longer training or different architecture")
        
        if results['distance_consistency']['distance_correlation'] < 0.5:
            recommendations.append("Distance relationships don't match expected pattern - check if model converged")
        
        if not results['adjacency_structure']['passes_adjacency_test']:
            recommendations.append("Adjacent numbers not consistently close - model may need more capacity")
        
        if results['distance_consistency']['radius_variance'] > 1.0:
            recommendations.append("High radius variance - embeddings not forming consistent circle")
        
        if len(recommendations) == 0:
            recommendations.append("Structure looks good - proceed with concept extraction and visualization")
        
        return recommendations

def create_validation_test_suite(p: int = 17) -> Dict[str, Any]:
    """Create comprehensive test suite for model validation."""
    
    # Generate test cases for different scenarios
    test_suite = {
        'perfect_circle_test': _generate_perfect_circle_embeddings(p),
        'noisy_circle_test': _generate_noisy_circle_embeddings(p),
        'random_embeddings_test': _generate_random_embeddings(p),
        'validator': CircularStructureValidator(p)
    }
    
    return test_suite

def _generate_perfect_circle_embeddings(p: int) -> torch.Tensor:
    """Generate perfect circular embeddings for testing validator."""
    angles = torch.linspace(0, 2 * np.pi, p, dtype=torch.float32)
    x = torch.cos(angles)
    y = torch.sin(angles)
    return torch.stack([x, y], dim=1)

def _generate_noisy_circle_embeddings(p: int, noise_level: float = 0.1) -> torch.Tensor:
    """Generate noisy circular embeddings for testing validator."""
    perfect = _generate_perfect_circle_embeddings(p)
    noise = torch.randn_like(perfect) * noise_level
    return perfect + noise

def _generate_random_embeddings(p: int, dim: int = 64) -> torch.Tensor:
    """Generate random embeddings for testing validator."""
    return torch.randn(p, dim)

if __name__ == "__main__":
    # Test the validator with known embeddings
    p = 17
    validator = CircularStructureValidator(p)
    
    # Test with perfect circle
    perfect_embeddings = _generate_perfect_circle_embeddings(p)
    results = validator.validate_embeddings(perfect_embeddings, visualize=True)
    
    print("Perfect circle validation:")
    print(f"  Score: {results['overall_assessment']['overall_score']:.2f}")
    print(f"  Quality: {results['overall_assessment']['quality_assessment']}")
    
    # Test with random embeddings
    random_embeddings = _generate_random_embeddings(p)
    results = validator.validate_embeddings(random_embeddings, visualize=False)
    
    print("Random embeddings validation:")
    print(f"  Score: {results['overall_assessment']['overall_score']:.2f}")
    print(f"  Quality: {results['overall_assessment']['quality_assessment']}")

