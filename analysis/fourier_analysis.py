"""
Fourier Analysis Module for NeuroMap.

Implements Fourier circuit analysis following Neel Nanda's "Progress Measures
for Grokking Via Mechanistic Interpretability" paper.

The key insight is that transformers learning modular arithmetic develop
Fourier-based algorithms where:
1. Embeddings encode numbers as sinusoidal waves
2. Attention computes Fourier multiplication
3. MLPs combine frequency components to produce outputs

This module provides tools to:
1. Extract Fourier components from embeddings
2. Measure alignment with expected Fourier basis
3. Visualize frequency structure
4. Verify Fourier algorithm mechanistically
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.stats import pearsonr, spearmanr


@dataclass
class FourierComponents:
    """Extracted Fourier components from neural representations."""
    frequencies: np.ndarray  # Frequency indices
    magnitudes: np.ndarray  # Magnitude of each frequency
    phases: np.ndarray  # Phase of each frequency
    dominant_frequency: int  # Most active frequency
    reconstruction_error: float  # Error from Fourier reconstruction


@dataclass
class FourierAlignmentResult:
    """Result of Fourier alignment analysis."""
    alignment_score: float  # How well activations align with Fourier basis
    frequency_importance: np.ndarray  # Importance of each frequency
    key_frequencies: List[int]  # Most important frequencies
    is_fourier_based: bool  # Does the model use Fourier algorithm?


class FourierBasis:
    """
    Discrete Fourier Transform basis for modular arithmetic.

    For modular arithmetic mod p, the relevant Fourier basis consists of
    complex exponentials exp(2*pi*i*k*n/p) for frequencies k = 0, 1, ..., p-1.

    The model should learn to represent numbers using these frequencies,
    particularly the frequencies that satisfy: k + k' = k'' (mod p) aligns
    with the Fourier multiplication property.
    """

    def __init__(self, p: int):
        self.p = p

        # Build Fourier basis matrix
        # F[k, n] = exp(2*pi*i*k*n/p) / sqrt(p)
        n = np.arange(p)
        k = np.arange(p)
        self.F = np.exp(2j * np.pi * np.outer(k, n) / p) / np.sqrt(p)

        # Real and imaginary parts (for real-valued activations)
        self.F_cos = np.cos(2 * np.pi * np.outer(k, n) / p)
        self.F_sin = np.sin(2 * np.pi * np.outer(k, n) / p)

        # Stacked real basis: [cos(k1), sin(k1), cos(k2), sin(k2), ...]
        self.F_real = self._build_real_basis()

    def _build_real_basis(self) -> np.ndarray:
        """Build real-valued Fourier basis."""
        basis_vectors = []

        # DC component (k=0)
        basis_vectors.append(np.ones(self.p) / np.sqrt(self.p))

        # For each frequency k = 1, ..., p//2
        for k in range(1, (self.p + 1) // 2):
            # Cosine component
            cos_k = np.cos(2 * np.pi * k * np.arange(self.p) / self.p)
            cos_k = cos_k / np.linalg.norm(cos_k)
            basis_vectors.append(cos_k)

            # Sine component
            sin_k = np.sin(2 * np.pi * k * np.arange(self.p) / self.p)
            sin_k = sin_k / np.linalg.norm(sin_k)
            basis_vectors.append(sin_k)

        # Nyquist component for even p
        if self.p % 2 == 0:
            nyquist = np.cos(np.pi * np.arange(self.p))
            nyquist = nyquist / np.linalg.norm(nyquist)
            basis_vectors.append(nyquist)

        return np.array(basis_vectors)

    def project(self, activations: np.ndarray) -> np.ndarray:
        """
        Project activations onto Fourier basis.

        Args:
            activations: Array of shape (p, d) where each row is representation of a number

        Returns:
            Fourier coefficients of shape (n_basis, d)
        """
        return self.F_real @ activations

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct activations from Fourier coefficients.

        Args:
            coefficients: Fourier coefficients

        Returns:
            Reconstructed activations
        """
        return self.F_real.T @ coefficients

    def get_frequency_strength(self, activations: np.ndarray) -> np.ndarray:
        """
        Get strength of each frequency in the activations.

        Args:
            activations: Array of shape (p, d)

        Returns:
            Frequency strengths of shape (p//2 + 1,)
        """
        # Use FFT for efficiency
        fft_result = fft(activations, axis=0)
        magnitudes = np.abs(fft_result)

        # Average over dimensions
        return magnitudes[:self.p // 2 + 1].mean(axis=1)


class FourierAnalyzer:
    """
    Analyze neural network representations for Fourier structure.

    This is the main interface for Fourier analysis in NeuroMap.
    """

    def __init__(self, p: int):
        self.p = p
        self.basis = FourierBasis(p)

    def extract_fourier_components(self,
                                   embeddings: torch.Tensor) -> FourierComponents:
        """
        Extract Fourier components from embedding matrix.

        Args:
            embeddings: Tensor of shape (p, d_model) with embeddings for numbers 0..p-1

        Returns:
            FourierComponents with frequency analysis
        """
        # Convert to numpy
        emb_np = embeddings.detach().cpu().numpy()

        # Compute FFT along the number dimension
        fft_result = fft(emb_np, axis=0)

        # Get magnitudes and phases
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)

        # Average over embedding dimensions
        avg_magnitudes = magnitudes.mean(axis=1)

        # Find dominant frequency
        dominant_freq = np.argmax(avg_magnitudes[1:]) + 1  # Exclude DC

        # Compute reconstruction error
        reconstructed = ifft(fft_result, axis=0).real
        reconstruction_error = np.mean((reconstructed - emb_np) ** 2)

        return FourierComponents(
            frequencies=np.arange(self.p),
            magnitudes=avg_magnitudes,
            phases=phases.mean(axis=1),
            dominant_frequency=int(dominant_freq),
            reconstruction_error=float(reconstruction_error)
        )

    def measure_fourier_alignment(self,
                                  embeddings: torch.Tensor) -> FourierAlignmentResult:
        """
        Measure how well embeddings align with Fourier basis.

        High alignment suggests the model is using a Fourier-based algorithm.

        Args:
            embeddings: Tensor of shape (p, d_model)

        Returns:
            FourierAlignmentResult with alignment metrics
        """
        emb_np = embeddings.detach().cpu().numpy()

        # Project onto Fourier basis
        coefficients = self.basis.project(emb_np)

        # Compute explained variance for each frequency
        total_var = np.var(emb_np)
        frequency_importance = np.zeros(len(coefficients))

        for i, coef in enumerate(coefficients):
            # Variance explained by this frequency
            reconstructed = np.outer(self.basis.F_real[i], coef)
            freq_var = np.var(reconstructed)
            frequency_importance[i] = freq_var / (total_var + 1e-8)

        # Overall alignment = fraction of variance explained by Fourier basis
        full_reconstruction = self.basis.reconstruct(coefficients)
        reconstruction_var = np.var(emb_np - full_reconstruction)
        alignment_score = 1.0 - (reconstruction_var / (total_var + 1e-8))

        # Key frequencies (explain > 10% of variance each)
        key_frequencies = np.where(frequency_importance > 0.1)[0].tolist()

        # Is it Fourier-based? (alignment > 0.8 and has key frequencies)
        is_fourier_based = alignment_score > 0.8 and len(key_frequencies) > 0

        return FourierAlignmentResult(
            alignment_score=float(alignment_score),
            frequency_importance=frequency_importance,
            key_frequencies=key_frequencies,
            is_fourier_based=is_fourier_based
        )

    def compute_fourier_basis_similarity(self,
                                         embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute similarity between learned embeddings and ideal Fourier basis.

        Args:
            embeddings: Tensor of shape (p, d_model)

        Returns:
            Dictionary with similarity metrics
        """
        emb_np = embeddings.detach().cpu().numpy()

        # For each dimension, find best matching Fourier frequency
        n_dims = emb_np.shape[1]
        best_correlations = []

        for d in range(n_dims):
            dim_activations = emb_np[:, d]

            # Compute correlation with each Fourier basis vector
            correlations = []
            for basis_vec in self.basis.F_real:
                corr, _ = pearsonr(dim_activations, basis_vec)
                correlations.append(abs(corr))

            best_correlations.append(max(correlations))

        return {
            'mean_best_correlation': float(np.mean(best_correlations)),
            'max_best_correlation': float(np.max(best_correlations)),
            'min_best_correlation': float(np.min(best_correlations)),
            'std_best_correlation': float(np.std(best_correlations)),
            'n_highly_aligned_dims': int(sum(c > 0.9 for c in best_correlations)),
        }

    def detect_circular_structure(self,
                                  embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Detect if embeddings form a circular structure (key Fourier property).

        In Fourier representations, numbers should be arranged in a circle
        where adjacent numbers are close.

        Args:
            embeddings: Tensor of shape (p, d_model)

        Returns:
            Dictionary with circular structure metrics
        """
        emb_np = embeddings.detach().cpu().numpy()

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(emb_np, emb_np, metric='euclidean')

        # Expected circular distances
        circular_distances = np.minimum(
            np.abs(np.arange(self.p)[:, None] - np.arange(self.p)[None, :]),
            self.p - np.abs(np.arange(self.p)[:, None] - np.arange(self.p)[None, :])
        )

        # Correlation between learned and expected distances
        # (Upper triangle only to avoid double-counting)
        upper_tri = np.triu_indices(self.p, k=1)
        learned_dists = distances[upper_tri]
        expected_dists = circular_distances[upper_tri].astype(float)

        distance_correlation, _ = pearsonr(learned_dists, expected_dists)

        # Check adjacency preservation
        adjacent_dists = []
        non_adjacent_dists = []

        for i in range(self.p):
            j = (i + 1) % self.p
            adjacent_dists.append(distances[i, j])

            for k in range(self.p):
                if k != i and k != j and k != (i - 1) % self.p:
                    non_adjacent_dists.append(distances[i, k])

        mean_adjacent = np.mean(adjacent_dists)
        mean_non_adjacent = np.mean(non_adjacent_dists)
        adjacency_ratio = mean_adjacent / (mean_non_adjacent + 1e-8)

        return {
            'distance_correlation': float(distance_correlation),
            'mean_adjacent_distance': float(mean_adjacent),
            'mean_non_adjacent_distance': float(mean_non_adjacent),
            'adjacency_ratio': float(adjacency_ratio),
            'is_circular': distance_correlation > 0.7 and adjacency_ratio < 0.5
        }

    def visualize_fourier_spectrum(self,
                                   embeddings: torch.Tensor,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the Fourier spectrum of embeddings.

        Args:
            embeddings: Tensor of shape (p, d_model)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        components = self.extract_fourier_components(embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Magnitude spectrum
        ax1 = axes[0]
        ax1.bar(components.frequencies, components.magnitudes)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Fourier Spectrum of Embeddings')
        ax1.axvline(components.dominant_frequency, color='r', linestyle='--',
                   label=f'Dominant: k={components.dominant_frequency}')
        ax1.legend()

        # Phase spectrum
        ax2 = axes[1]
        ax2.bar(components.frequencies, components.phases)
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Phase Spectrum of Embeddings')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def visualize_embedding_circle(self,
                                   embeddings: torch.Tensor,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize embeddings projected onto first 2 principal components.

        If Fourier-based, should form a circle.

        Args:
            embeddings: Tensor of shape (p, d_model)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        from sklearn.decomposition import PCA

        emb_np = embeddings.detach().cpu().numpy()

        # PCA to 2D
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb_np)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot points
        for i in range(self.p):
            ax.scatter(emb_2d[i, 0], emb_2d[i, 1], s=100, zorder=2)
            ax.annotate(str(i), (emb_2d[i, 0], emb_2d[i, 1]),
                       fontsize=10, ha='center', va='center')

        # Connect adjacent numbers
        for i in range(self.p):
            j = (i + 1) % self.p
            ax.plot([emb_2d[i, 0], emb_2d[j, 0]],
                   [emb_2d[i, 1], emb_2d[j, 1]],
                   'k-', alpha=0.3, zorder=1)

        # Draw ideal circle for reference
        theta = np.linspace(0, 2 * np.pi, 100)
        radius = np.sqrt(emb_2d[:, 0] ** 2 + emb_2d[:, 1] ** 2).mean()
        ax.plot(radius * np.cos(theta), radius * np.sin(theta),
               'r--', alpha=0.3, label='Ideal circle')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Embedding Structure (mod {self.p})')
        ax.axis('equal')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def generate_fourier_report(analyzer: FourierAnalyzer,
                           embeddings: torch.Tensor) -> str:
    """
    Generate a comprehensive Fourier analysis report.

    Args:
        analyzer: FourierAnalyzer instance
        embeddings: Embedding tensor

    Returns:
        Formatted report string
    """
    components = analyzer.extract_fourier_components(embeddings)
    alignment = analyzer.measure_fourier_alignment(embeddings)
    similarity = analyzer.compute_fourier_basis_similarity(embeddings)
    circular = analyzer.detect_circular_structure(embeddings)

    lines = []
    lines.append("=" * 60)
    lines.append("FOURIER ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("FOURIER COMPONENTS")
    lines.append("-" * 40)
    lines.append(f"  Dominant frequency: k = {components.dominant_frequency}")
    lines.append(f"  Reconstruction error: {components.reconstruction_error:.6f}")
    lines.append("")

    lines.append("FOURIER ALIGNMENT")
    lines.append("-" * 40)
    lines.append(f"  Alignment score: {alignment.alignment_score:.4f}")
    lines.append(f"  Key frequencies: {alignment.key_frequencies}")
    lines.append(f"  Is Fourier-based: {alignment.is_fourier_based}")
    lines.append("")

    lines.append("BASIS SIMILARITY")
    lines.append("-" * 40)
    lines.append(f"  Mean best correlation: {similarity['mean_best_correlation']:.4f}")
    lines.append(f"  Highly aligned dims: {similarity['n_highly_aligned_dims']}")
    lines.append("")

    lines.append("CIRCULAR STRUCTURE")
    lines.append("-" * 40)
    lines.append(f"  Distance correlation: {circular['distance_correlation']:.4f}")
    lines.append(f"  Adjacency ratio: {circular['adjacency_ratio']:.4f}")
    lines.append(f"  Is circular: {circular['is_circular']}")
    lines.append("")

    # Interpretation
    lines.append("INTERPRETATION")
    lines.append("-" * 40)

    if alignment.is_fourier_based and circular['is_circular']:
        lines.append("  [CONFIRMED] Model uses Fourier algorithm for modular arithmetic")
        lines.append("  Embeddings form circular structure with clear frequency components")
    elif alignment.alignment_score > 0.5:
        lines.append("  [PARTIAL] Model shows some Fourier structure")
        lines.append("  May be learning a hybrid or approximate algorithm")
    else:
        lines.append("  [NO FOURIER] Model does not appear to use Fourier algorithm")
        lines.append("  May use memorization or alternative computation")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    print("Testing Fourier Analysis Module")
    print("=" * 50)

    p = 17

    # Create analyzer
    analyzer = FourierAnalyzer(p)
    print(f"Created Fourier analyzer for p = {p}")

    # Create synthetic embeddings (perfect Fourier structure)
    print("\nTesting with perfect Fourier embeddings:")
    angles = 2 * np.pi * np.arange(p) / p
    perfect_embeddings = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], axis=1),
                                       dtype=torch.float32)

    components = analyzer.extract_fourier_components(perfect_embeddings)
    print(f"  Dominant frequency: {components.dominant_frequency}")

    alignment = analyzer.measure_fourier_alignment(perfect_embeddings)
    print(f"  Alignment score: {alignment.alignment_score:.4f}")
    print(f"  Is Fourier-based: {alignment.is_fourier_based}")

    circular = analyzer.detect_circular_structure(perfect_embeddings)
    print(f"  Distance correlation: {circular['distance_correlation']:.4f}")
    print(f"  Is circular: {circular['is_circular']}")

    # Test with random embeddings
    print("\nTesting with random embeddings:")
    random_embeddings = torch.randn(p, 64)

    alignment = analyzer.measure_fourier_alignment(random_embeddings)
    print(f"  Alignment score: {alignment.alignment_score:.4f}")
    print(f"  Is Fourier-based: {alignment.is_fourier_based}")

    circular = analyzer.detect_circular_structure(random_embeddings)
    print(f"  Distance correlation: {circular['distance_correlation']:.4f}")
    print(f"  Is circular: {circular['is_circular']}")

    # Generate report
    print("\nGenerating full report:")
    report = generate_fourier_report(analyzer, perfect_embeddings)
    print(report)

    print("\nFourier analysis module implementation complete!")
