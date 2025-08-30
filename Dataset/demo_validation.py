"""
Demonstration of the validation framework with different embedding types.

This script shows how the CircularStructureValidator works with:
1. Perfect circular embeddings (should score 1.0)
2. Noisy circular embeddings (should score lower)
3. Random embeddings (should score near 0.0)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from validation import CircularStructureValidator, _generate_perfect_circle_embeddings, _generate_noisy_circle_embeddings, _generate_random_embeddings

def demonstrate_validation():
    """Demonstrate validation with different embedding types."""
    
    p = 17
    validator = CircularStructureValidator(p)
    
    # Generate different types of embeddings
    perfect_embeddings = _generate_perfect_circle_embeddings(p)
    noisy_embeddings = _generate_noisy_circle_embeddings(p, noise_level=0.2)
    random_embeddings = _generate_random_embeddings(p, dim=2)  # 2D for visualization
    
    # Validate each type
    print("Validating different embedding types...")
    
    perfect_results = validator.validate_embeddings(perfect_embeddings, visualize=False)
    noisy_results = validator.validate_embeddings(noisy_embeddings, visualize=False)
    random_results = validator.validate_embeddings(random_embeddings, visualize=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot embeddings
    embeddings_list = [perfect_embeddings, noisy_embeddings, random_embeddings]
    results_list = [perfect_results, noisy_results, random_results]
    titles = ['Perfect Circle', 'Noisy Circle', 'Random Embeddings']
    
    for i, (embeddings, results, title) in enumerate(zip(embeddings_list, results_list, titles)):
        # Top row: scatter plots
        ax_top = axes[0, i]
        embeddings_np = embeddings.numpy()
        
        scatter = ax_top.scatter(embeddings_np[:, 0], embeddings_np[:, 1], 
                               c=range(p), cmap='hsv', s=100, alpha=0.8)
        
        # Add number labels
        for j in range(p):
            ax_top.annotate(str(j), (embeddings_np[j, 0], embeddings_np[j, 1]),
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=8, fontweight='bold')
        
        ax_top.set_title(f'{title}\nScore: {results["overall_assessment"]["overall_score"]:.2f}', 
                        fontsize=12, fontweight='bold')
        ax_top.set_xlabel('Dimension 1')
        ax_top.set_ylabel('Dimension 2')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_aspect('equal')
        
        # Add quality assessment
        quality = results['overall_assessment']['quality_assessment']
        ax_top.text(0.02, 0.98, quality, transform=ax_top.transAxes, 
                   fontsize=10, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='lightgreen' if results["overall_assessment"]["overall_score"] > 0.8 
                           else 'yellow' if results["overall_assessment"]["overall_score"] > 0.4 
                           else 'lightcoral', alpha=0.8))
        
        # Bottom row: metrics breakdown
        ax_bottom = axes[1, i]
        ax_bottom.axis('off')
        
        metrics = results['overall_assessment']['key_metrics']
        metrics_text = f"""
Validation Metrics:

Circular Ordering: {'✓' if metrics['circular_ordering'] else '✗'}
Distance Correlation: {metrics['distance_correlation']:.3f}
Adjacency Test: {'✓' if metrics['adjacency_test_passed'] else '✗'}
Adjacency Ratio: {metrics['adjacency_ratio']:.3f}

Overall Score: {results['overall_assessment']['overall_score']:.2f}/1.0

Recommendations:
"""
        
        for rec in results['overall_assessment']['recommendations'][:3]:  # Show first 3
            metrics_text += f"• {rec}\n"
        
        ax_bottom.text(0.05, 0.95, metrics_text, transform=ax_bottom.transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/validation_framework_demo.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Validation framework demo saved to: results/validation_framework_demo.png")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    for title, results in zip(titles, results_list):
        score = results['overall_assessment']['overall_score']
        quality = results['overall_assessment']['quality_assessment']
        print(f"{title:20} | Score: {score:.2f} | {quality}")
    
    print("="*60)
    
    return fig

if __name__ == "__main__":
    print("Demonstrating validation framework with different embedding types...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run demonstration
    demo_fig = demonstrate_validation()
    
    plt.show()

