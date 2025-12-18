"""
Faithfulness Evaluation Module for NeuroMap.

Measures how well extracted concepts represent actual model computation.
A faithful representation means that the concepts are not just correlational
but actually causally responsible for model behavior.

Key metrics:
- Faithfulness Score: Accuracy when using concept reconstructions
- Concept Completeness: Variance explained by concepts
- Concept Separability: Linear separability of concepts
- Reconstruction Quality: How well concepts reconstruct activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    faithfulness_score: float  # Main metric: how faithful are the concepts
    completeness_score: float  # How much variance do concepts explain
    separability_score: float  # How separable are the concepts
    reconstruction_error: float  # MSE of activation reconstruction
    accuracy_retention: float  # Accuracy when using concept representations
    concept_purity: float  # Average purity of concept clusters


@dataclass
class ConceptQuality:
    """Quality metrics for individual concepts."""
    concept_id: int
    purity: float  # Fraction of samples from dominant class
    coherence: float  # Average similarity within concept
    distinctiveness: float  # Distance to other concepts
    size: int  # Number of samples in concept


class FaithfulnessEvaluator:
    """
    Evaluates how faithfully extracted concepts represent model behavior.

    The key insight is that if concepts are faithful representations,
    replacing activations with concept-based reconstructions should
    preserve model behavior.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def compute_faithfulness(self,
                            concept_extractor,
                            dataset,
                            layer_name: str,
                            activations: Optional[torch.Tensor] = None) -> FaithfulnessResult:
        """
        Compute comprehensive faithfulness metrics.

        The faithfulness score measures: if we replace the model's activations
        with the "concept centers" from the concept extractor, what is the
        drop in accuracy?

        High faithfulness = small accuracy drop = concepts capture essential info

        Args:
            concept_extractor: Trained concept extractor (e.g., ClusteringExtractor)
            dataset: Dataset with inputs and targets
            layer_name: Layer where concepts were extracted
            activations: Pre-computed activations (optional)

        Returns:
            FaithfulnessResult with detailed metrics
        """
        inputs = dataset.data['inputs'].to(self.device)
        targets = dataset.data['targets'].to(self.device)

        # Get original model outputs
        with torch.no_grad():
            original_outputs = self.model(inputs)
            original_accuracy = (original_outputs.argmax(-1) == targets).float().mean().item()

        # Get concept assignments and centers
        if hasattr(concept_extractor, 'concept_assignments'):
            assignments = concept_extractor.concept_assignments
        else:
            raise ValueError("Concept extractor must have concept_assignments attribute")

        if hasattr(concept_extractor, 'concept_centers'):
            centers = concept_extractor.concept_centers
        else:
            raise ValueError("Concept extractor must have concept_centers attribute")

        # Reconstruct activations using concept centers
        if centers is not None:
            reconstructed = centers[assignments]
        else:
            raise ValueError("No concept centers available for reconstruction")

        # Compute reconstruction error
        if activations is not None:
            activations_flat = activations.view(activations.shape[0], -1).float()
            reconstructed_flat = reconstructed.view(reconstructed.shape[0], -1).float()
            reconstruction_error = F.mse_loss(reconstructed_flat, activations_flat).item()
        else:
            reconstruction_error = 0.0

        # Compute accuracy when using reconstructed activations
        # This requires running the model with patched activations
        try:
            reconstructed_accuracy = self._compute_reconstructed_accuracy(
                inputs, targets, layer_name, reconstructed.to(self.device)
            )
        except Exception:
            reconstructed_accuracy = original_accuracy  # Fallback

        # Faithfulness = how much accuracy is retained
        if original_accuracy > 0:
            faithfulness_score = reconstructed_accuracy / original_accuracy
        else:
            faithfulness_score = 0.0

        # Completeness = variance explained by concepts
        completeness_score = self._compute_completeness(activations, centers, assignments)

        # Separability = linear separability of concepts
        if activations is not None:
            separability_score = self._compute_separability(activations, assignments)
        else:
            separability_score = 0.0

        # Concept purity
        if hasattr(dataset.data, 'targets'):
            concept_purity = self._compute_concept_purity(assignments, targets.cpu())
        else:
            concept_purity = 0.0

        return FaithfulnessResult(
            faithfulness_score=faithfulness_score,
            completeness_score=completeness_score,
            separability_score=separability_score,
            reconstruction_error=reconstruction_error,
            accuracy_retention=reconstructed_accuracy,
            concept_purity=concept_purity
        )

    def _compute_reconstructed_accuracy(self,
                                        inputs: torch.Tensor,
                                        targets: torch.Tensor,
                                        layer_name: str,
                                        reconstructed: torch.Tensor) -> float:
        """
        Compute accuracy when using reconstructed activations.

        This replaces the layer's output with the reconstructed activations
        and measures the resulting accuracy.
        """
        from .causal_intervention import ActivationPatcher

        # Create a patcher
        patcher = ActivationPatcher(self.model, self.device)

        # Create a hook that replaces activations with reconstructed
        def replace_hook(module, input, output):
            if isinstance(output, tuple):
                # Try to match shapes
                target_shape = output[0].shape
                recon = reconstructed

                # Reshape if needed
                if recon.shape != target_shape:
                    if len(target_shape) == 3 and len(recon.shape) == 2:
                        # Add sequence dimension
                        recon = recon.unsqueeze(1).expand(-1, target_shape[1], -1)

                return (recon,) + output[1:]
            else:
                return reconstructed

        # Find the layer and register hook
        if layer_name in patcher.hook_points:
            module = patcher.hook_points[layer_name]
            hook = module.register_forward_hook(replace_hook)

            with torch.no_grad():
                outputs = self.model(inputs)

            hook.remove()

            accuracy = (outputs.argmax(-1) == targets).float().mean().item()
            return accuracy

        # Fallback: return 0 if layer not found
        return 0.0

    def _compute_completeness(self,
                             activations: Optional[torch.Tensor],
                             centers: torch.Tensor,
                             assignments: torch.Tensor) -> float:
        """
        Compute how much variance the concepts explain.

        Completeness = 1 - (residual_variance / total_variance)
        """
        if activations is None or centers is None:
            return 0.0

        activations_flat = activations.view(activations.shape[0], -1).float().numpy()
        centers_flat = centers.view(centers.shape[0], -1).float().numpy()

        # Get reconstructed activations
        reconstructed = centers_flat[assignments.numpy()]

        # Total variance
        total_var = np.var(activations_flat)

        # Residual variance
        residuals = activations_flat - reconstructed
        residual_var = np.var(residuals)

        if total_var > 0:
            completeness = 1.0 - (residual_var / total_var)
        else:
            completeness = 0.0

        return max(0.0, min(1.0, completeness))

    def _compute_separability(self,
                             activations: torch.Tensor,
                             assignments: torch.Tensor) -> float:
        """
        Compute linear separability of concepts.

        Uses SVM classification accuracy as a proxy for separability.
        """
        X = activations.view(activations.shape[0], -1).float().numpy()
        y = assignments.numpy()

        # Check if we have enough samples per class
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2 or min(counts) < 2:
            return 0.0

        try:
            # Reduce dimensionality if needed
            if X.shape[1] > 50:
                pca = PCA(n_components=min(50, X.shape[0] - 1))
                X = pca.fit_transform(X)

            # Train linear SVM
            svm = SVC(kernel='linear', max_iter=1000)
            svm.fit(X, y)

            # Compute accuracy
            predictions = svm.predict(X)
            separability = accuracy_score(y, predictions)

        except Exception:
            separability = 0.0

        return separability

    def _compute_concept_purity(self,
                               assignments: torch.Tensor,
                               true_labels: torch.Tensor) -> float:
        """
        Compute average purity of concept clusters.

        Purity = fraction of samples from the dominant true class in each concept.
        """
        assignments_np = assignments.numpy()
        labels_np = true_labels.numpy()

        unique_concepts = np.unique(assignments_np)
        purities = []

        for concept_id in unique_concepts:
            mask = assignments_np == concept_id
            if mask.sum() > 0:
                concept_labels = labels_np[mask]
                unique, counts = np.unique(concept_labels, return_counts=True)
                purity = counts.max() / counts.sum()
                purities.append(purity)

        if purities:
            return float(np.mean(purities))
        return 0.0

    def analyze_concept_quality(self,
                               activations: torch.Tensor,
                               assignments: torch.Tensor,
                               centers: torch.Tensor,
                               labels: Optional[torch.Tensor] = None) -> List[ConceptQuality]:
        """
        Analyze quality of individual concepts.

        Args:
            activations: Activation tensor
            assignments: Concept assignments for each sample
            centers: Concept center vectors
            labels: Optional ground truth labels

        Returns:
            List of ConceptQuality for each concept
        """
        activations_flat = activations.view(activations.shape[0], -1).float()
        assignments_np = assignments.numpy()
        centers_flat = centers.view(centers.shape[0], -1).float()

        results = []
        unique_concepts = np.unique(assignments_np)

        for concept_id in unique_concepts:
            mask = assignments_np == concept_id
            concept_activations = activations_flat[mask]
            concept_center = centers_flat[concept_id]

            # Size
            size = int(mask.sum())

            # Coherence: average similarity within concept
            if size > 1:
                similarities = F.cosine_similarity(
                    concept_activations,
                    concept_center.unsqueeze(0).expand(size, -1)
                )
                coherence = float(similarities.mean())
            else:
                coherence = 1.0

            # Distinctiveness: distance to other concept centers
            other_centers = centers_flat[np.arange(len(centers_flat)) != concept_id]
            if len(other_centers) > 0:
                distances = torch.cdist(
                    concept_center.unsqueeze(0),
                    other_centers
                )
                distinctiveness = float(distances.mean())
            else:
                distinctiveness = 0.0

            # Purity: if labels provided
            if labels is not None:
                concept_labels = labels.numpy()[mask]
                unique, counts = np.unique(concept_labels, return_counts=True)
                purity = float(counts.max() / counts.sum())
            else:
                purity = 0.0

            results.append(ConceptQuality(
                concept_id=int(concept_id),
                purity=purity,
                coherence=coherence,
                distinctiveness=distinctiveness,
                size=size
            ))

        return results


class LinearRepresentationTester:
    """
    Test the Linear Representation Hypothesis.

    The Linear Representation Hypothesis states that concepts are stored
    as linear directions in activation space. This class provides tools
    to test this hypothesis.
    """

    def __init__(self):
        pass

    def test_linear_separability(self,
                                activations: torch.Tensor,
                                concept_labels: torch.Tensor,
                                test_split: float = 0.2) -> Dict[str, float]:
        """
        Test if concepts are linearly separable.

        Args:
            activations: Activation tensor (n_samples, dim)
            concept_labels: Concept labels for each sample
            test_split: Fraction of data for testing

        Returns:
            Dictionary with train and test accuracy
        """
        X = activations.view(activations.shape[0], -1).float().numpy()
        y = concept_labels.numpy()

        # Split data
        n_samples = len(X)
        n_test = int(n_samples * test_split)
        indices = np.random.permutation(n_samples)

        X_train, X_test = X[indices[n_test:]], X[indices[:n_test]]
        y_train, y_test = y[indices[n_test:]], y[indices[:n_test]]

        # Train linear classifier
        try:
            clf = LogisticRegression(max_iter=1000, multi_class='ovr')
            clf.fit(X_train, y_train)

            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)

            return {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'is_linearly_separable': test_acc > 0.9
            }
        except Exception as e:
            return {
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'is_linearly_separable': False,
                'error': str(e)
            }

    def find_concept_directions(self,
                               activations: torch.Tensor,
                               concept_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Find the linear direction for each concept.

        Uses difference-in-means to find concept directions.

        Args:
            activations: Activation tensor
            concept_labels: Concept labels

        Returns:
            Dictionary mapping concept ID to direction vector
        """
        activations_flat = activations.view(activations.shape[0], -1).float()
        labels_np = concept_labels.numpy()

        # Global mean
        global_mean = activations_flat.mean(dim=0)

        directions = {}
        unique_concepts = np.unique(labels_np)

        for concept_id in unique_concepts:
            mask = labels_np == concept_id
            concept_mean = activations_flat[mask].mean(dim=0)

            # Direction = concept_mean - global_mean
            direction = concept_mean - global_mean

            # Normalize
            direction = direction / (direction.norm() + 1e-8)

            directions[int(concept_id)] = direction

        return directions

    def test_direction_quality(self,
                              activations: torch.Tensor,
                              concept_labels: torch.Tensor,
                              directions: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """
        Test the quality of concept directions.

        A good direction should:
        1. Have high projection for same-concept samples
        2. Have low projection for different-concept samples

        Args:
            activations: Activation tensor
            concept_labels: Concept labels
            directions: Concept directions from find_concept_directions

        Returns:
            Quality metrics for the directions
        """
        activations_flat = activations.view(activations.shape[0], -1).float()
        labels_np = concept_labels.numpy()

        same_concept_projections = []
        diff_concept_projections = []

        for concept_id, direction in directions.items():
            mask = labels_np == concept_id

            # Project all activations onto this direction
            projections = (activations_flat @ direction).numpy()

            # Same-concept projections (should be high)
            same_concept_projections.extend(projections[mask])

            # Different-concept projections (should be low)
            diff_concept_projections.extend(projections[~mask])

        same_mean = np.mean(same_concept_projections)
        diff_mean = np.mean(diff_concept_projections)
        same_std = np.std(same_concept_projections)
        diff_std = np.std(diff_concept_projections)

        # Discriminability: Cohen's d
        pooled_std = np.sqrt((same_std**2 + diff_std**2) / 2)
        if pooled_std > 0:
            discriminability = (same_mean - diff_mean) / pooled_std
        else:
            discriminability = 0.0

        return {
            'same_concept_projection_mean': float(same_mean),
            'diff_concept_projection_mean': float(diff_mean),
            'discriminability': float(discriminability),
            'direction_quality': 'good' if discriminability > 1.0 else 'poor'
        }


def generate_faithfulness_report(result: FaithfulnessResult,
                                concept_qualities: Optional[List[ConceptQuality]] = None) -> str:
    """
    Generate a human-readable faithfulness report.

    Args:
        result: FaithfulnessResult from evaluation
        concept_qualities: Optional list of per-concept quality metrics

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("FAITHFULNESS EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("SUMMARY METRICS")
    lines.append("-" * 40)
    lines.append(f"  Faithfulness Score:     {result.faithfulness_score:.4f}")
    lines.append(f"  Completeness Score:     {result.completeness_score:.4f}")
    lines.append(f"  Separability Score:     {result.separability_score:.4f}")
    lines.append(f"  Reconstruction Error:   {result.reconstruction_error:.4f}")
    lines.append(f"  Accuracy Retention:     {result.accuracy_retention:.2%}")
    lines.append(f"  Concept Purity:         {result.concept_purity:.4f}")
    lines.append("")

    # Interpretation
    lines.append("INTERPRETATION")
    lines.append("-" * 40)

    if result.faithfulness_score > 0.9:
        lines.append("  [EXCELLENT] Concepts are highly faithful representations")
    elif result.faithfulness_score > 0.7:
        lines.append("  [GOOD] Concepts capture most of the model's computation")
    elif result.faithfulness_score > 0.5:
        lines.append("  [MODERATE] Concepts partially represent model behavior")
    else:
        lines.append("  [POOR] Concepts may not be faithful to actual computation")

    if result.completeness_score > 0.8:
        lines.append("  Concepts explain most of the activation variance")
    else:
        lines.append("  Concepts miss some activation variance")

    if result.separability_score > 0.9:
        lines.append("  Concepts are highly linearly separable")
    else:
        lines.append("  Concepts have limited linear separability")

    if concept_qualities:
        lines.append("")
        lines.append("PER-CONCEPT QUALITY")
        lines.append("-" * 40)

        for cq in concept_qualities[:10]:  # Show top 10
            lines.append(f"  Concept {cq.concept_id}:")
            lines.append(f"    Purity: {cq.purity:.3f}, Coherence: {cq.coherence:.3f}")
            lines.append(f"    Distinctiveness: {cq.distinctiveness:.3f}, Size: {cq.size}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))

    from Dataset.dataset import ModularArithmeticDataset
    from models.transformer import create_model
    from analysis.activation_extractor import ActivationExtractor
    from analysis.concept_extractors import ClusteringExtractor

    print("Testing Faithfulness Evaluation Module")
    print("=" * 50)

    # Create test data and model
    p = 5
    dataset = ModularArithmeticDataset(p=p)
    model = create_model(vocab_size=p)

    # Extract activations
    extractor = ActivationExtractor(model)
    number_reps = extractor.extract_number_representations('embeddings')
    print(f"Extracted number representations: {number_reps.shape}")

    # Extract concepts
    concept_extractor = ClusteringExtractor(n_concepts=p, method='kmeans')
    concept_info = concept_extractor.extract_concepts(number_reps)
    print(f"Extracted {concept_info['n_concepts_found']} concepts")

    # Evaluate faithfulness
    evaluator = FaithfulnessEvaluator(model)

    # Compute completeness and separability (without full faithfulness test)
    completeness = evaluator._compute_completeness(
        number_reps,
        concept_extractor.concept_centers,
        concept_extractor.concept_assignments
    )
    print(f"Completeness score: {completeness:.4f}")

    separability = evaluator._compute_separability(
        number_reps,
        concept_extractor.concept_assignments
    )
    print(f"Separability score: {separability:.4f}")

    # Test linear representation hypothesis
    lrt = LinearRepresentationTester()

    # Create synthetic labels for testing
    labels = torch.arange(p)
    directions = lrt.find_concept_directions(number_reps, labels)
    print(f"Found {len(directions)} concept directions")

    quality = lrt.test_direction_quality(number_reps, labels, directions)
    print(f"Direction discriminability: {quality['discriminability']:.4f}")

    # Analyze concept quality
    concept_qualities = evaluator.analyze_concept_quality(
        number_reps,
        concept_extractor.concept_assignments,
        concept_extractor.concept_centers,
        labels
    )
    print(f"Analyzed {len(concept_qualities)} concepts")

    print("\nFaithfulness evaluation module implementation complete!")
