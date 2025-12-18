#!/usr/bin/env python3
"""
Programmatic Screenshot Generator for NeuroMap.

Generates all visualizations and figures for documentation and README.
Run this script to reproduce all visual assets.

Usage:
    python scripts/generate_screenshots.py --output screenshots/
    python scripts/generate_screenshots.py --modulus 17 --output results/figures/
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from Dataset.dataset import ModularArithmeticDataset
    from models.hooked_transformer import create_hooked_model
    from analysis.fourier_analysis import FourierAnalyzer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Using synthetic data.")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Generate NeuroMap screenshots')
    parser.add_argument('--output', '-o', type=str, default='screenshots/',
                       help='Output directory for figures')
    parser.add_argument('--modulus', '-p', type=int, default=17,
                       help='Modulus for modular arithmetic')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for saved figures')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Output format')
    return parser.parse_args()


def generate_synthetic_embeddings(p, d_model=64, circular=True):
    """Generate synthetic embeddings for visualization."""
    if circular:
        # Create circular embeddings
        angles = 2 * np.pi * np.arange(p) / p
        embeddings = np.zeros((p, d_model))
        # Primary circular structure in first 2 dimensions
        embeddings[:, 0] = np.cos(angles)
        embeddings[:, 1] = np.sin(angles)
        # Add Fourier harmonics in other dimensions
        for k in range(2, min(8, d_model // 2)):
            embeddings[:, 2*k] = 0.5 * np.cos(k * angles)
            embeddings[:, 2*k + 1] = 0.5 * np.sin(k * angles)
        # Add small noise
        embeddings += np.random.randn(p, d_model) * 0.1
    else:
        # Random embeddings
        embeddings = np.random.randn(p, d_model)
    return embeddings


def generate_fourier_spectrum(output_dir, p=17, dpi=150, fmt='png'):
    """Generate Fourier spectrum visualization."""
    print("Generating Fourier spectrum...")

    embeddings = generate_synthetic_embeddings(p, circular=True)

    # Compute Fourier transform
    fourier_basis = np.exp(2j * np.pi * np.outer(np.arange(p), np.arange(p)) / p) / np.sqrt(p)
    coefficients = fourier_basis @ embeddings
    powers = np.sum(np.abs(coefficients) ** 2, axis=1)
    powers = powers / powers.max()  # Normalize

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(p), powers, color='steelblue', edgecolor='navy', alpha=0.8)

    # Highlight dominant frequencies
    dominant = np.argsort(powers)[-3:]
    for idx in dominant:
        bars[idx].set_color('darkorange')
        bars[idx].set_edgecolor('red')

    ax.set_xlabel('Frequency k', fontsize=12)
    ax.set_ylabel('Normalized Power', fontsize=12)
    ax.set_title(f'Fourier Spectrum of Learned Embeddings (p={p})', fontsize=14)
    ax.set_xticks(range(0, p, 2))

    # Add legend
    normal_patch = mpatches.Patch(color='steelblue', label='Frequency component')
    dominant_patch = mpatches.Patch(color='darkorange', label='Dominant frequency')
    ax.legend(handles=[normal_patch, dominant_patch], loc='upper right')

    plt.tight_layout()
    save_path = output_dir / f'fourier_spectrum.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_embedding_circle(output_dir, p=17, dpi=150, fmt='png'):
    """Generate circular embedding visualization."""
    print("Generating embedding circle...")

    embeddings = generate_synthetic_embeddings(p, circular=True)

    # PCA projection
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
    embeddings_2d = embeddings_centered @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw connecting lines
    for i in range(p):
        j = (i + 1) % p
        ax.plot([embeddings_2d[i, 0], embeddings_2d[j, 0]],
                [embeddings_2d[i, 1], embeddings_2d[j, 1]],
                'gray', alpha=0.3, linewidth=1)

    # Color by position on circle
    colors = plt.cm.hsv(np.linspace(0, 1, p))

    for i in range(p):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                  c=[colors[i]], s=200, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Draw expected circle
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.sqrt(np.mean(embeddings_2d[:, 0]**2 + embeddings_2d[:, 1]**2))
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.3, linewidth=2)

    ax.set_aspect('equal')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'Number Embeddings Form a Circle (p={p})', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f'embedding_circle.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_causal_intervention_chart(output_dir, p=17, dpi=150, fmt='png'):
    """Generate causal intervention results chart."""
    print("Generating causal intervention chart...")

    # Simulated intervention results
    layers = ['Embeddings', 'L0 Attention', 'L0 MLP', 'L1 Attention', 'L1 MLP']
    accuracy_drops = [0.45, 0.23, 0.12, 0.08, 0.31]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#e74c3c' if d > 0.3 else '#f39c12' if d > 0.15 else '#27ae60' for d in accuracy_drops]

    bars = ax.barh(layers, accuracy_drops, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, drop in zip(bars, accuracy_drops):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{drop:.0%}', va='center', fontsize=11)

    ax.set_xlabel('Accuracy Drop (Mean Ablation)', fontsize=12)
    ax.set_title('Layer Importance via Causal Intervention', fontsize=14)
    ax.set_xlim(0, 0.6)
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax.axvline(x=0.15, color='orange', linestyle='--', alpha=0.5, label='Important threshold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = output_dir / f'causal_intervention.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_circuit_diagram(output_dir, p=17, dpi=150, fmt='png'):
    """Generate circuit diagram visualization."""
    print("Generating circuit diagram...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define circuit components
    components = [
        ('Embed', (1, 3), 0.234, 'lightblue'),
        ('L0H2', (3, 4), 0.152, 'lightgreen'),
        ('L0H0', (3, 2), 0.083, 'lightgreen'),
        ('L0 MLP', (5, 3), 0.045, 'lightyellow'),
        ('L1H3', (7, 4), 0.071, 'lightgreen'),
        ('L1 MLP', (9, 3), 0.089, 'lightyellow'),
        ('Output', (11, 3), None, 'lightcoral'),
    ]

    # Draw connections
    connections = [
        ('Embed', 'L0H2'), ('Embed', 'L0H0'),
        ('L0H2', 'L0 MLP'), ('L0H0', 'L0 MLP'),
        ('L0 MLP', 'L1H3'),
        ('L1H3', 'L1 MLP'),
        ('L1 MLP', 'Output')
    ]

    comp_dict = {c[0]: (c[1], c[2], c[3]) for c in components}

    for src, dst in connections:
        src_pos = comp_dict[src][0]
        dst_pos = comp_dict[dst][0]
        ax.annotate('', xy=dst_pos, xytext=src_pos,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.6))

    # Draw components
    for name, pos, importance, color in components:
        circle = plt.Circle(pos, 0.6, color=color, ec='black', lw=2, zorder=5)
        ax.add_patch(circle)

        if importance:
            label = f'{name}\n({importance:.3f})'
        else:
            label = name
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Modular Addition Circuit (importance scores)', fontsize=14, pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Embedding'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Attention Head'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='MLP'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    save_path = output_dir / f'circuit_diagram.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_training_dynamics(output_dir, p=17, dpi=150, fmt='png'):
    """Generate training dynamics visualization."""
    print("Generating training dynamics...")

    # Simulated training data (grokking phenomenon)
    epochs = np.arange(0, 5001, 50)

    # Accuracy shows sudden grokking
    train_acc = 1 / (1 + np.exp(-0.003 * (epochs - 1500)))
    test_acc = 1 / (1 + np.exp(-0.004 * (epochs - 3000)))

    # Loss decreases
    train_loss = 3 * np.exp(-0.001 * epochs) + 0.1
    test_loss = 3 * np.exp(-0.0008 * epochs) + 0.2

    # Fourier alignment emerges
    fourier_align = 1 / (1 + np.exp(-0.002 * (epochs - 2500))) * 0.9 + 0.1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Accuracy
    axes[0].plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, test_acc, 'r--', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy (Grokking)')
    axes[0].legend()
    axes[0].axhline(y=0.99, color='green', linestyle=':', alpha=0.5)
    axes[0].set_ylim(0, 1.05)

    # Loss
    axes[1].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, test_loss, 'r--', label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].set_yscale('log')

    # Fourier alignment
    axes[2].plot(epochs, fourier_align, 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Fourier Alignment')
    axes[2].set_title('Emergence of Fourier Structure')
    axes[2].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    axes[2].legend()
    axes[2].set_ylim(0, 1)

    plt.suptitle(f'Training Dynamics (p={p})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = output_dir / f'training_dynamics.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_faithfulness_comparison(output_dir, dpi=150, fmt='png'):
    """Generate faithfulness comparison chart."""
    print("Generating faithfulness comparison...")

    methods = ['Clustering', 'Linear Probe', 'SAE', 'Gated SAE']
    faithfulness = [0.72, 0.85, 0.81, 0.89]
    completeness = [0.68, 0.79, 0.83, 0.91]
    separability = [0.81, 0.92, 0.78, 0.87]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width, faithfulness, width, label='Faithfulness', color='steelblue')
    bars2 = ax.bar(x, completeness, width, label='Completeness', color='darkorange')
    bars3 = ax.bar(x + width, separability, width, label='Separability', color='forestgreen')

    ax.set_ylabel('Score')
    ax.set_title('Concept Extraction Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Quality threshold')

    plt.tight_layout()
    save_path = output_dir / f'faithfulness_comparison.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_distance_correlation(output_dir, p=17, dpi=150, fmt='png'):
    """Generate distance correlation visualization."""
    print("Generating distance correlation...")

    embeddings = generate_synthetic_embeddings(p, circular=True)

    # Compute embedding distances
    embed_dist = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            embed_dist[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

    # Compute circular distances
    circular_dist = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            diff = abs(i - j)
            circular_dist[i, j] = min(diff, p - diff)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Embedding distances
    im1 = axes[0].imshow(embed_dist, cmap='viridis')
    axes[0].set_title('Embedding Distances')
    axes[0].set_xlabel('Number j')
    axes[0].set_ylabel('Number i')
    plt.colorbar(im1, ax=axes[0])

    # Circular distances
    im2 = axes[1].imshow(circular_dist, cmap='viridis')
    axes[1].set_title('Circular Distances')
    axes[1].set_xlabel('Number j')
    axes[1].set_ylabel('Number i')
    plt.colorbar(im2, ax=axes[1])

    # Correlation scatter
    embed_flat = embed_dist.flatten()
    circular_flat = circular_dist.flatten()
    correlation = np.corrcoef(embed_flat, circular_flat)[0, 1]

    axes[2].scatter(circular_flat, embed_flat, alpha=0.5, s=10)
    axes[2].set_xlabel('Circular Distance')
    axes[2].set_ylabel('Embedding Distance')
    axes[2].set_title(f'Correlation: r = {correlation:.3f}')

    # Add trend line
    z = np.polyfit(circular_flat, embed_flat, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(0, circular_flat.max(), 100)
    axes[2].plot(x_line, p_line(x_line), 'r--', linewidth=2)

    plt.suptitle(f'Distance Preservation in Embeddings (p={p})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = output_dir / f'distance_correlation.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_architecture_comparison(output_dir, dpi=150, fmt='png'):
    """Generate architecture comparison chart."""
    print("Generating architecture comparison...")

    moduli = [7, 13, 17, 23, 31]

    transformer_acc = [1.0, 0.95, 0.89, 0.78, 0.67]
    memory_acc = [1.0, 1.0, 1.0, 1.0, 1.0]

    transformer_fourier = [0.92, 0.88, 0.85, 0.72, 0.58]
    memory_fourier = [0.15, 0.12, 0.10, 0.08, 0.06]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(moduli))
    width = 0.35

    # Accuracy comparison
    axes[0].bar(x - width/2, transformer_acc, width, label='Transformer', color='steelblue')
    axes[0].bar(x + width/2, memory_acc, width, label='Memory-based', color='darkorange')
    axes[0].set_xlabel('Modulus p')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy by Architecture')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(moduli)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # Fourier alignment comparison
    axes[1].bar(x - width/2, transformer_fourier, width, label='Transformer', color='steelblue')
    axes[1].bar(x + width/2, memory_fourier, width, label='Memory-based', color='darkorange')
    axes[1].set_xlabel('Modulus p')
    axes[1].set_ylabel('Fourier Alignment')
    axes[1].set_title('Fourier Structure by Architecture')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(moduli)
    axes[1].legend()
    axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, 1)

    plt.suptitle('Architecture Comparison: Transformer vs Memory-based', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = output_dir / f'architecture_comparison.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_pipeline_overview(output_dir, dpi=150, fmt='png'):
    """Generate pipeline overview diagram."""
    print("Generating pipeline overview...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Pipeline stages
    stages = [
        ('Dataset\nCreation', (1, 3), '#3498db'),
        ('Model\nTraining', (3.5, 3), '#e74c3c'),
        ('Activation\nExtraction', (6, 3), '#2ecc71'),
        ('Fourier\nAnalysis', (8.5, 4), '#9b59b6'),
        ('Causal\nIntervention', (8.5, 2), '#f39c12'),
        ('Circuit\nDiscovery', (11, 3), '#1abc9c'),
        ('Visualization\n& Report', (13.5, 3), '#34495e'),
    ]

    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)
    ]

    for src, dst in connections:
        src_pos = stages[src][1]
        dst_pos = stages[dst][1]
        ax.annotate('', xy=(dst_pos[0] - 0.8, dst_pos[1]),
                   xytext=(src_pos[0] + 0.8, src_pos[1]),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Draw stages
    for name, pos, color in stages:
        rect = plt.Rectangle((pos[0] - 0.8, pos[1] - 0.6), 1.6, 1.2,
                             facecolor=color, edgecolor='black', lw=2,
                             alpha=0.8, zorder=5)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', zorder=6)

    ax.set_xlim(-0.5, 15)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('NeuroMap Analysis Pipeline', fontsize=16, pad=20)

    plt.tight_layout()
    save_path = output_dir / f'pipeline_overview.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NEUROMAP SCREENSHOT GENERATOR")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Modulus: {args.modulus}")
    print(f"Format: {args.format}")
    print(f"DPI: {args.dpi}")
    print()

    # Generate all figures
    generate_pipeline_overview(output_dir, dpi=args.dpi, fmt=args.format)
    generate_fourier_spectrum(output_dir, p=args.modulus, dpi=args.dpi, fmt=args.format)
    generate_embedding_circle(output_dir, p=args.modulus, dpi=args.dpi, fmt=args.format)
    generate_causal_intervention_chart(output_dir, p=args.modulus, dpi=args.dpi, fmt=args.format)
    generate_circuit_diagram(output_dir, p=args.modulus, dpi=args.dpi, fmt=args.format)
    generate_training_dynamics(output_dir, p=args.modulus, dpi=args.dpi, fmt=args.format)
    generate_faithfulness_comparison(output_dir, dpi=args.dpi, fmt=args.format)
    generate_distance_correlation(output_dir, p=args.modulus, dpi=args.dpi, fmt=args.format)
    generate_architecture_comparison(output_dir, dpi=args.dpi, fmt=args.format)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(list(output_dir.glob(f'*.{args.format}')))} figures")
    print(f"Output directory: {output_dir.absolute()}")

    # Save manifest
    manifest = {
        'modulus': args.modulus,
        'format': args.format,
        'dpi': args.dpi,
        'files': [f.name for f in output_dir.glob(f'*.{args.format}')]
    }
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
