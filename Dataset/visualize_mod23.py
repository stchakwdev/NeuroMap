"""
Visualize the circular structure of the mod_23 dataset.

This script creates comprehensive visualizations showing:
1. The modular addition table
2. Expected circular structure of numbers 0-22
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

def create_comprehensive_visualizations_mod23():
    """Create all visualizations for the mod_23 dataset."""
    
    print("Loading mod_23 dataset...")
    dataset = ModularArithmeticDataset.load('data/mod_23_dataset.pkl')
    p = dataset.p
    
    print(f"Dataset loaded: {dataset.data['num_examples']} examples for mod {p}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Modular Addition Table
    ax1 = plt.subplot(2, 3, 1)
    addition_table = dataset.metadata['algebraic_properties']['addition_table'].numpy()
    im1 = ax1.imshow(addition_table, cmap='viridis', aspect='equal')
    ax1.set_title(f'Modular Addition Table (mod {p})', fontsize=16, fontweight='bold')
    ax1.set_xlabel('b', fontsize=12)
    ax1.set_ylabel('a', fontsize=12)
    
    # For p=23, don't add text annotations as they would be too crowded
    plt.colorbar(im1, ax=ax1, label='(a + b) mod p')
    
    # 2. Expected Circular Structure
    ax2 = plt.subplot(2, 3, 2)
    angles = np.linspace(0, 2*np.pi, p, endpoint=False)
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    
    scatter = ax2.scatter(circle_x, circle_y, c=range(p), cmap='hsv', s=120, alpha=0.8)
    
    # Add number labels (smaller font for p=23)
    for i in range(p):
        ax2.annotate(str(i), (circle_x[i], circle_y[i]),
                    xytext=(6, 6), textcoords='offset points', 
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Draw circle outline
    circle_outline = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax2.add_patch(circle_outline)
    
    ax2.set_title('Expected Circular Structure', fontsize=16, fontweight='bold')
    ax2.set_xlabel('cos(2πi/p)', fontsize=12)
    ax2.set_ylabel('sin(2πi/p)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    
    # 3. Adjacency Relationships (show every 3rd connection to avoid clutter)
    ax3 = plt.subplot(2, 3, 3)
    adjacency_pairs = dataset.metadata['circular_structure']['adjacency_pairs']
    
    # Plot the circle again
    ax3.scatter(circle_x, circle_y, c=range(p), cmap='hsv', s=120, alpha=0.8)
    
    # Draw adjacency connections (every 3rd to reduce visual clutter)
    for idx, (i, next_i) in enumerate(adjacency_pairs):
        if idx % 3 == 0:  # Show every 3rd connection
            x1, y1 = circle_x[i], circle_y[i]
            x2, y2 = circle_x[next_i], circle_y[next_i]
            ax3.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, linewidth=1.5)
    
    # Add labels for key numbers only
    key_numbers = [0, 5, 10, 15, 20]
    for i in key_numbers:
        ax3.annotate(str(i), (circle_x[i], circle_y[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax3.set_title('Adjacency Relationships\n(i → i+1 mod p, every 3rd shown)', fontsize=16, fontweight='bold')
    ax3.set_xlabel('cos(2πi/p)', fontsize=12)
    ax3.set_ylabel('sin(2πi/p)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    
    # 4. Distance Matrix Visualization
    ax4 = plt.subplot(2, 3, 4)
    circular_distances = dataset.metadata['distance_matrices']['circular_distance'].numpy()
    im4 = ax4.imshow(circular_distances, cmap='plasma', aspect='equal')
    ax4.set_title('Circular Distance Matrix', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Number j', fontsize=12)
    ax4.set_ylabel('Number i', fontsize=12)
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
                         c=range(p), cmap='hsv', s=120, alpha=0.8)
    
    # Add labels for key numbers only
    for i in key_numbers:
        ax5.annotate(str(i), (embeddings_np[i, 0], embeddings_np[i, 1]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # Add validation score
    score = results['overall_assessment']['overall_score']
    quality = results['overall_assessment']['quality_assessment']
    ax5.text(0.02, 0.98, f'Validation Score: {score:.2f}\n{quality}', 
             transform=ax5.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    ax5.set_title('Perfect Circle Validation', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax5.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Validation Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create text summary of validation results
    metrics = results['overall_assessment']['key_metrics']
    summary_text = f"""
Validation Results for Perfect Circle (mod {p}):

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

Scale Comparison:
• mod 17: 289 examples
• mod 23: 529 examples (+83% more data)
• Larger circle → finer angular resolution
• More complex adjacency patterns
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    ax6.set_title('Dataset & Validation Summary', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/mod_23_circular_structure_visualization.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Comprehensive visualization saved to: results/mod_23_circular_structure_visualization.png")
    
    return fig, results

def create_interactive_distance_analysis_mod23():
    """Create detailed distance analysis visualization for mod_23."""
    
    dataset = ModularArithmeticDataset.load('data/mod_23_dataset.pkl')
    p = dataset.p
    
    # Create figure for distance analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Circular Distance Heatmap
    circular_distances = dataset.metadata['distance_matrices']['circular_distance'].numpy()
    im1 = ax1.imshow(circular_distances, cmap='viridis', aspect='equal')
    ax1.set_title(f'Circular Distance Matrix (mod {p})\n(Shortest path on circle)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Number j', fontsize=12)
    ax1.set_ylabel('Number i', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Distance')
    
    # 2. Euclidean Distance Heatmap
    euclidean_distances = dataset.metadata['distance_matrices']['euclidean_distance'].numpy()
    im2 = ax2.imshow(euclidean_distances, cmap='plasma', aspect='equal')
    ax2.set_title(f'Euclidean Distance Matrix (mod {p})\n(On unit circle)', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Number j', fontsize=12)
    ax2.set_ylabel('Number i', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Distance')
    
    # 3. Distance Correlation Analysis
    # Flatten upper triangular matrices for correlation
    triu_indices = np.triu_indices(p, k=1)
    circular_flat = circular_distances[triu_indices]
    euclidean_flat = euclidean_distances[triu_indices]
    
    ax3.scatter(circular_flat, euclidean_flat, alpha=0.4, s=30)
    ax3.set_xlabel('Circular Distance', fontsize=12)
    ax3.set_ylabel('Euclidean Distance', fontsize=12)
    ax3.set_title(f'Distance Correlation Analysis (mod {p})', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(circular_flat, euclidean_flat)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nData points: {len(circular_flat)}', 
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
    
    ax4.hist(adjacent_distances, bins=25, alpha=0.7, label=f'Adjacent pairs (n={len(adjacent_distances)})', color='green')
    ax4.hist(non_adjacent_distances, bins=25, alpha=0.7, label=f'Non-adjacent pairs (n={len(non_adjacent_distances)})', color='red')
    ax4.set_xlabel('Euclidean Distance', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'Distance Distribution (mod {p})\n(Perfect Circle Embeddings)', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    mean_adj = np.mean(adjacent_distances)
    mean_non_adj = np.mean(non_adjacent_distances)
    ax4.axvline(mean_adj, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax4.axvline(mean_non_adj, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax4.text(0.05, 0.95, f'Mean Adjacent: {mean_adj:.3f}\nMean Non-Adjacent: {mean_non_adj:.3f}\nSeparation: {mean_non_adj/mean_adj:.2f}x', 
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('results/mod_23_distance_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Distance analysis saved to: results/mod_23_distance_analysis.png")
    
    return fig

def create_scale_comparison():
    """Create a comparison between mod_17 and mod_23 datasets."""
    
    # Load both datasets
    dataset_17 = ModularArithmeticDataset.load('data/mod_17_dataset.pkl')
    dataset_23 = ModularArithmeticDataset.load('data/mod_23_dataset.pkl')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    datasets = [dataset_17, dataset_23]
    titles = ['mod 17 (289 examples)', 'mod 23 (529 examples)']
    
    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        p = dataset.p
        
        # Top row: Circular structures
        ax_top = axes[0, i]
        angles = np.linspace(0, 2*np.pi, p, endpoint=False)
        circle_x = np.cos(angles)
        circle_y = np.sin(angles)
        
        scatter = ax_top.scatter(circle_x, circle_y, c=range(p), cmap='hsv', s=100, alpha=0.8)
        
        # Add some number labels
        label_step = max(1, p // 8)  # Show ~8 labels
        for j in range(0, p, label_step):
            ax_top.annotate(str(j), (circle_x[j], circle_y[j]),
                          xytext=(6, 6), textcoords='offset points', 
                          fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        circle_outline = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax_top.add_patch(circle_outline)
        
        ax_top.set_title(f'Circular Structure: {title}', fontsize=14, fontweight='bold')
        ax_top.set_xlabel('cos(2πi/p)')
        ax_top.set_ylabel('sin(2πi/p)')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_aspect('equal')
        ax_top.set_xlim(-1.3, 1.3)
        ax_top.set_ylim(-1.3, 1.3)
        
        # Bottom row: Distance matrices
        ax_bottom = axes[1, i]
        circular_distances = dataset.metadata['distance_matrices']['circular_distance'].numpy()
        im = ax_bottom.imshow(circular_distances, cmap='viridis', aspect='equal')
        ax_bottom.set_title(f'Distance Matrix: {title}', fontsize=14, fontweight='bold')
        ax_bottom.set_xlabel('Number j')
        ax_bottom.set_ylabel('Number i')
        plt.colorbar(im, ax=ax_bottom, label='Circular Distance')
        
        # Add dataset statistics
        stats_text = f"""
Examples: {dataset.data['num_examples']}
Angular resolution: {360/p:.1f}°
Max distance: {p//2}
Adjacency pairs: {len(dataset.metadata['circular_structure']['adjacency_pairs'])}
        """
        ax_bottom.text(0.02, 0.98, stats_text.strip(), transform=ax_bottom.transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/mod_17_vs_mod_23_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Scale comparison saved to: results/mod_17_vs_mod_23_comparison.png")
    
    return fig

if __name__ == "__main__":
    print("Creating comprehensive visualizations for mod_23 dataset...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Create main visualization
    main_fig, validation_results = create_comprehensive_visualizations_mod23()
    
    # Create distance analysis
    distance_fig = create_interactive_distance_analysis_mod23()
    
    # Create scale comparison
    comparison_fig = create_scale_comparison()
    
    print("\n" + "="*70)
    print("MOD_23 VISUALIZATION SUMMARY")
    print("="*70)
    print(f"✓ Main visualization: results/mod_23_circular_structure_visualization.png")
    print(f"✓ Distance analysis: results/mod_23_distance_analysis.png")
    print(f"✓ Scale comparison: results/mod_17_vs_mod_23_comparison.png")
    print(f"✓ Validation score: {validation_results['overall_assessment']['overall_score']:.2f}")
    print(f"✓ Quality assessment: {validation_results['overall_assessment']['quality_assessment']}")
    print(f"✓ Dataset size: 529 examples (83% larger than mod_17)")
    print(f"✓ Angular resolution: {360/23:.1f}° per step")
    print("="*70)
    
    # Show plots
    plt.show()

