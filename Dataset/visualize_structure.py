"""
Visualize the circular structure of the mod_17 dataset.

This script creates comprehensive visualizations showing:
1. The modular addition table
2. Expected circular structure of numbers 0-16
3. Validation of perfect circular embeddings
4. Distance relationships and adjacency structure
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator, _generate_perfect_circle_embeddings
from utils import visualize_addition_table, visualize_circular_structure
import seaborn as sns

def create_comprehensive_visualizations():
    """Create all visualizations for the mod_17 dataset."""
    
    print("Loading mod_17 dataset...")
    dataset = ModularArithmeticDataset.load('data/mod_17_dataset.pkl')
    p = dataset.p
    
    print(f"Dataset loaded: {dataset.data['num_examples']} examples for mod {p}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Modular Addition Table
    ax1 = plt.subplot(2, 3, 1)
    addition_table = dataset.metadata['algebraic_properties']['addition_table'].numpy()
    im1 = ax1.imshow(addition_table, cmap='viridis', aspect='equal')
    ax1.set_title(f'Modular Addition Table (mod {p})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('b')
    ax1.set_ylabel('a')
    
    # Add text annotations for readability
    for a in range(p):
        for b in range(p):
            text = ax1.text(b, a, f'{int(addition_table[a, b])}', 
                           ha='center', va='center', 
                           color='white' if addition_table[a, b] < p/2 else 'black',
                           fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='(a + b) mod p')
    
    # 2. Expected Circular Structure
    ax2 = plt.subplot(2, 3, 2)
    angles = np.linspace(0, 2*np.pi, p, endpoint=False)
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    
    scatter = ax2.scatter(circle_x, circle_y, c=range(p), cmap='hsv', s=150, alpha=0.8)
    
    # Add number labels
    for i in range(p):
        ax2.annotate(str(i), (circle_x[i], circle_y[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw circle outline
    circle_outline = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax2.add_patch(circle_outline)
    
    ax2.set_title('Expected Circular Structure', fontsize=14, fontweight='bold')
    ax2.set_xlabel('cos(2πi/p)')
    ax2.set_ylabel('sin(2πi/p)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    
    # 3. Adjacency Relationships
    ax3 = plt.subplot(2, 3, 3)
    adjacency_pairs = dataset.metadata['circular_structure']['adjacency_pairs']
    
    # Plot the circle again
    ax3.scatter(circle_x, circle_y, c=range(p), cmap='hsv', s=150, alpha=0.8)
    
    # Draw adjacency connections
    for i, next_i in adjacency_pairs:
        x1, y1 = circle_x[i], circle_y[i]
        x2, y2 = circle_x[next_i], circle_y[next_i]
        ax3.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, linewidth=2)
    
    # Add labels
    for i in range(p):
        ax3.annotate(str(i), (circle_x[i], circle_y[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax3.set_title('Adjacency Relationships\n(i → i+1 mod p)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('cos(2πi/p)')
    ax3.set_ylabel('sin(2πi/p)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    
    # 4. Distance Matrix Visualization
    ax4 = plt.subplot(2, 3, 4)
    circular_distances = dataset.metadata['distance_matrices']['circular_distance'].numpy()
    im4 = ax4.imshow(circular_distances, cmap='plasma', aspect='equal')
    ax4.set_title('Circular Distance Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number j')
    ax4.set_ylabel('Number i')
    plt.colorbar(im4, ax=ax4, label='Circular Distance')
    
    # 5. Perfect Circle Validation
    ax5 = plt.subplot(2, 3, 5)
    
    # Generate perfect circular embeddings
    perfect_embeddings = _generate_perfect_circle_embeddings(p)
    validator = CircularStructureValidator(p)
    results = validator.validate_embeddings(perfect_embeddings, visualize=False)
    
    # Plot the perfect embeddings
    embeddings_np = perfect_embeddings.numpy()
    scatter = ax5.scatter(embeddings_np[:, 0], embeddings_np[:, 1], 
                         c=range(p), cmap='hsv', s=150, alpha=0.8)
    
    # Add labels
    for i in range(p):
        ax5.annotate(str(i), (embeddings_np[i, 0], embeddings_np[i, 1]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add validation score
    score = results['overall_assessment']['overall_score']
    quality = results['overall_assessment']['quality_assessment']
    ax5.text(0.02, 0.98, f'Validation Score: {score:.2f}\n{quality}', 
             transform=ax5.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    ax5.set_title('Perfect Circle Validation', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Embedding Dimension 1')
    ax5.set_ylabel('Embedding Dimension 2')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Validation Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create text summary of validation results
    metrics = results['overall_assessment']['key_metrics']
    summary_text = f"""
Validation Results for Perfect Circle:

✓ Circular Ordering: {metrics['circular_ordering']}
✓ Distance Correlation: {metrics['distance_correlation']:.3f}
✓ Adjacency Test: {metrics['adjacency_test_passed']}
✓ Adjacency Ratio: {metrics['adjacency_ratio']:.3f}

Overall Score: {score:.2f}/1.0
Quality: {quality}

Dataset Properties:
• Total Examples: {dataset.data['num_examples']}
• Input Shape: {dataset.data['inputs'].shape}
• Representation: {dataset.representation}
• Prime Modulus: {p}

Structural Metadata:
• Adjacent Pairs: {len(dataset.metadata['circular_structure']['adjacency_pairs'])}
• Commutative Pairs: {len(dataset.metadata['algebraic_properties']['commutative_pairs'])}
• Identity Pairs: {len(dataset.metadata['algebraic_properties']['identity_pairs'])}
• Inverse Pairs: {len(dataset.metadata['algebraic_properties']['inverse_pairs'])}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    ax6.set_title('Dataset & Validation Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/mod_17_circular_structure_visualization.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Comprehensive visualization saved to: results/mod_17_circular_structure_visualization.png")
    
    return fig, results

def create_interactive_distance_analysis():
    """Create detailed distance analysis visualization."""
    
    dataset = ModularArithmeticDataset.load('data/mod_17_dataset.pkl')
    p = dataset.p
    
    # Create figure for distance analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Circular Distance Heatmap
    circular_distances = dataset.metadata['distance_matrices']['circular_distance'].numpy()
    im1 = ax1.imshow(circular_distances, cmap='viridis', aspect='equal')
    ax1.set_title('Circular Distance Matrix\n(Shortest path on circle)', fontweight='bold')
    ax1.set_xlabel('Number j')
    ax1.set_ylabel('Number i')
    plt.colorbar(im1, ax=ax1, label='Distance')
    
    # 2. Euclidean Distance Heatmap
    euclidean_distances = dataset.metadata['distance_matrices']['euclidean_distance'].numpy()
    im2 = ax2.imshow(euclidean_distances, cmap='plasma', aspect='equal')
    ax2.set_title('Euclidean Distance Matrix\n(On unit circle)', fontweight='bold')
    ax2.set_xlabel('Number j')
    ax2.set_ylabel('Number i')
    plt.colorbar(im2, ax=ax2, label='Distance')
    
    # 3. Distance Correlation Analysis
    # Flatten upper triangular matrices for correlation
    triu_indices = np.triu_indices(p, k=1)
    circular_flat = circular_distances[triu_indices]
    euclidean_flat = euclidean_distances[triu_indices]
    
    ax3.scatter(circular_flat, euclidean_flat, alpha=0.6, s=50)
    ax3.set_xlabel('Circular Distance')
    ax3.set_ylabel('Euclidean Distance')
    ax3.set_title('Distance Correlation Analysis', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(circular_flat, euclidean_flat)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax3.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 4. Adjacency Distance Distribution
    perfect_embeddings = _generate_perfect_circle_embeddings(p)
    
    # Calculate adjacent vs non-adjacent distances
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
    
    ax4.hist(adjacent_distances, bins=20, alpha=0.7, label='Adjacent pairs', color='green')
    ax4.hist(non_adjacent_distances, bins=20, alpha=0.7, label='Non-adjacent pairs', color='red')
    ax4.set_xlabel('Euclidean Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution\n(Perfect Circle Embeddings)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    mean_adj = np.mean(adjacent_distances)
    mean_non_adj = np.mean(non_adjacent_distances)
    ax4.axvline(mean_adj, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax4.axvline(mean_non_adj, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax4.text(0.05, 0.95, f'Mean Adjacent: {mean_adj:.3f}\nMean Non-Adjacent: {mean_non_adj:.3f}', 
             transform=ax4.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('results/mod_17_distance_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Distance analysis saved to: results/mod_17_distance_analysis.png")
    
    return fig

if __name__ == "__main__":
    print("Creating comprehensive visualizations for mod_17 dataset...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Create main visualization
    main_fig, validation_results = create_comprehensive_visualizations()
    
    # Create distance analysis
    distance_fig = create_interactive_distance_analysis()
    
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"✓ Main visualization: results/mod_17_circular_structure_visualization.png")
    print(f"✓ Distance analysis: results/mod_17_distance_analysis.png")
    print(f"✓ Validation score: {validation_results['overall_assessment']['overall_score']:.2f}")
    print(f"✓ Quality assessment: {validation_results['overall_assessment']['quality_assessment']}")
    print("="*60)
    
    # Show plots
    plt.show()

