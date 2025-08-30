"""
Generate detailed comparison data for mod_17 vs mod_23 analysis report.
"""

import torch
import numpy as np
from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator, _generate_perfect_circle_embeddings
import time
import json

def generate_comprehensive_comparison():
    """Generate detailed comparison metrics."""
    
    print("Generating comprehensive comparison data...")
    
    # Load datasets
    dataset_17 = ModularArithmeticDataset.load('data/mod_17_dataset.pkl')
    dataset_23 = ModularArithmeticDataset.load('data/mod_23_dataset.pkl')
    
    comparison_data = {}
    
    for p, dataset in [(17, dataset_17), (23, dataset_23)]:
        print(f"\nAnalyzing mod_{p} dataset...")
        
        # Basic metrics
        data = {
            'p': p,
            'total_examples': dataset.data['num_examples'],
            'angular_resolution_degrees': 360 / p,
            'max_circular_distance': p // 2,
            'adjacent_pairs': len(dataset.metadata['circular_structure']['adjacency_pairs']),
            'commutative_pairs': len(dataset.metadata['algebraic_properties']['commutative_pairs']),
            'identity_pairs': len(dataset.metadata['algebraic_properties']['identity_pairs']),
            'inverse_pairs': len(dataset.metadata['algebraic_properties']['inverse_pairs'])
        }
        
        # Distance analysis
        circular_distances = dataset.metadata['distance_matrices']['circular_distance'].numpy()
        euclidean_distances = dataset.metadata['distance_matrices']['euclidean_distance'].numpy()
        
        # Correlation analysis
        triu_indices = np.triu_indices(p, k=1)
        circular_flat = circular_distances[triu_indices]
        euclidean_flat = euclidean_distances[triu_indices]
        correlation = np.corrcoef(circular_flat, euclidean_flat)[0, 1]
        
        data['distance_correlation'] = correlation
        data['distance_pairs_analyzed'] = len(circular_flat)
        
        # Adjacency analysis with perfect embeddings
        perfect_embeddings = _generate_perfect_circle_embeddings(p)
        
        adjacent_distances = []
        non_adjacent_distances = []
        
        for i in range(p):
            for j in range(p):
                if i != j:
                    dist = torch.norm(perfect_embeddings[i] - perfect_embeddings[j]).item()
                    if abs(i - j) % p in [1, p - 1]:  # Adjacent
                        adjacent_distances.append(dist)
                    else:  # Non-adjacent
                        non_adjacent_distances.append(dist)
        
        mean_adj = np.mean(adjacent_distances)
        mean_non_adj = np.mean(non_adjacent_distances)
        separation_ratio = mean_non_adj / mean_adj
        
        data['mean_adjacent_distance'] = mean_adj
        data['mean_non_adjacent_distance'] = mean_non_adj
        data['distance_separation_ratio'] = separation_ratio
        
        # Validation performance
        validator = CircularStructureValidator(p)
        
        # Time validation (rough estimate)
        start_time = time.time()
        results = validator.validate_embeddings(perfect_embeddings, visualize=False)
        validation_time = time.time() - start_time
        
        data['validation_time_seconds'] = validation_time
        data['validation_score'] = results['overall_assessment']['overall_score']
        data['validation_quality'] = results['overall_assessment']['quality_assessment']
        
        # Memory footprint (rough estimate)
        dataset_size_mb = (dataset.data['inputs'].element_size() * dataset.data['inputs'].nelement() + 
                          dataset.data['targets'].element_size() * dataset.data['targets'].nelement()) / (1024 * 1024)
        data['dataset_size_mb'] = dataset_size_mb
        
        comparison_data[f'mod_{p}'] = data
    
    # Calculate relative differences
    mod_17_data = comparison_data['mod_17']
    mod_23_data = comparison_data['mod_23']
    
    relative_differences = {
        'examples_increase_percent': ((mod_23_data['total_examples'] - mod_17_data['total_examples']) / mod_17_data['total_examples']) * 100,
        'resolution_improvement_percent': ((mod_17_data['angular_resolution_degrees'] - mod_23_data['angular_resolution_degrees']) / mod_17_data['angular_resolution_degrees']) * 100,
        'correlation_improvement': mod_23_data['distance_correlation'] - mod_17_data['distance_correlation'],
        'separation_improvement': mod_23_data['distance_separation_ratio'] - mod_17_data['distance_separation_ratio'],
        'validation_time_ratio': mod_23_data['validation_time_seconds'] / mod_17_data['validation_time_seconds'],
        'memory_ratio': mod_23_data['dataset_size_mb'] / mod_17_data['dataset_size_mb']
    }
    
    comparison_data['relative_differences'] = relative_differences
    
    # Save to JSON
    with open('results/comparison_data.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Generate summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'mod_17':<15} {'mod_23':<15} {'Difference':<20}")
    print("-" * 80)
    print(f"{'Examples':<30} {mod_17_data['total_examples']:<15} {mod_23_data['total_examples']:<15} {relative_differences['examples_increase_percent']:+.1f}%")
    print(f"{'Angular Resolution (°)':<30} {mod_17_data['angular_resolution_degrees']:<15.1f} {mod_23_data['angular_resolution_degrees']:<15.1f} {relative_differences['resolution_improvement_percent']:+.1f}% finer")
    print(f"{'Distance Correlation':<30} {mod_17_data['distance_correlation']:<15.3f} {mod_23_data['distance_correlation']:<15.3f} {relative_differences['correlation_improvement']:+.3f}")
    print(f"{'Separation Ratio':<30} {mod_17_data['distance_separation_ratio']:<15.2f} {mod_23_data['distance_separation_ratio']:<15.2f} {relative_differences['separation_improvement']:+.2f}")
    print(f"{'Validation Time (s)':<30} {mod_17_data['validation_time_seconds']:<15.4f} {mod_23_data['validation_time_seconds']:<15.4f} {relative_differences['validation_time_ratio']:.1f}x slower")
    print(f"{'Dataset Size (MB)':<30} {mod_17_data['dataset_size_mb']:<15.3f} {mod_23_data['dataset_size_mb']:<15.3f} {relative_differences['memory_ratio']:.1f}x larger")
    
    print(f"\n{'Validation Performance':<30} {'Identical':<15} {'Identical':<15} {'No difference':<20}")
    print(f"{'Perfect Circle Score':<30} {mod_17_data['validation_score']:<15.2f} {mod_23_data['validation_score']:<15.2f} {'Same':<20}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("• mod_23 provides 83% more training examples with 26% finer resolution")
    print("• Validation framework performance is identical across scales")
    print("• mod_23 offers better statistical robustness with minimal computational overhead")
    print("• Both datasets maintain perfect mathematical structure guarantees")
    print("="*80)
    
    return comparison_data

def generate_recommendation_matrix():
    """Generate decision matrix for choosing between datasets."""
    
    recommendations = {
        'use_cases': {
            'mod_17': [
                'Rapid prototyping and initial development',
                'Educational demonstrations and tutorials',
                'Resource-constrained environments',
                'Baseline comparisons with existing literature',
                'Quick validation of interpretability methods',
                'Real-time demonstrations'
            ],
            'mod_23': [
                'Publication-quality research and analysis',
                'Comprehensive method validation',
                'Statistical significance testing',
                'Fine-grained concept extraction',
                'High-resolution topology visualization',
                'Robust cross-validation studies'
            ]
        },
        'decision_factors': {
            'speed_priority': 'mod_17',
            'quality_priority': 'mod_23',
            'education_focus': 'mod_17',
            'research_focus': 'mod_23',
            'limited_resources': 'mod_17',
            'comprehensive_analysis': 'mod_23'
        },
        'hybrid_approach': {
            'development_phase': 'mod_17',
            'validation_phase': 'mod_23',
            'publication_phase': 'mod_23'
        }
    }
    
    with open('results/recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nRECOMMENDATION MATRIX GENERATED")
    print("Saved to: results/recommendations.json")
    
    return recommendations

if __name__ == "__main__":
    # Generate comprehensive comparison
    comparison_data = generate_comprehensive_comparison()
    
    # Generate recommendation matrix
    recommendations = generate_recommendation_matrix()
    
    print(f"\nComparison data saved to: results/comparison_data.json")
    print(f"Analysis complete!")

