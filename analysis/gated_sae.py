"""
Gated Sparse Autoencoder for NeuroMap.

Implements a production-grade Gated SAE following Anthropic's "Scaling
Monosemanticity" approach. The gating mechanism avoids the shrinkage
problem where L1 penalties reduce activation magnitudes.

Key improvements over basic SAE:
1. Gating mechanism to preserve activation magnitude
2. Proper initialization
3. Dead feature detection and resampling
4. Feature interpretation tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json


@dataclass
class SAEConfig:
    """Configuration for Gated SAE."""
    d_model: int  # Input/output dimension
    d_sae: int  # SAE hidden dimension (usually 4-8x d_model)
    l1_coefficient: float = 1e-3
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    dead_feature_threshold: int = 10_000_000  # Tokens before feature considered dead
    dead_feature_window: int = 1000  # Window for measuring feature activity


@dataclass
class SAEOutput:
    """Output from SAE forward pass."""
    reconstruction: torch.Tensor
    feature_acts: torch.Tensor  # Activated features
    l2_loss: float
    l1_loss: float
    total_loss: float
    dead_features: int  # Count of dead features


class GatedSAE(nn.Module):
    """
    Gated Sparse Autoencoder.

    Architecture:
    - Encoder produces (pre_activation, gate_activation)
    - Feature activation = ReLU(pre_activation) * sigmoid(gate_activation)
    - Decoder reconstructs from feature activations

    The gating mechanism allows the model to learn when to activate features
    without the L1 penalty causing magnitude shrinkage.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder weights
        self.W_enc = nn.Parameter(
            torch.randn(config.d_model, config.d_sae) / np.sqrt(config.d_model)
        )
        self.b_enc = nn.Parameter(torch.zeros(config.d_sae))

        # Gating parameters
        self.r_mag = nn.Parameter(torch.zeros(config.d_sae))
        self.b_gate = nn.Parameter(torch.zeros(config.d_sae))

        # Decoder weights (tied to encoder by default, but can untie)
        self.W_dec = nn.Parameter(
            torch.randn(config.d_sae, config.d_model) / np.sqrt(config.d_sae)
        )
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))

        # Feature activation tracking
        self.register_buffer(
            'feature_activation_counts',
            torch.zeros(config.d_sae, dtype=torch.long)
        )
        self.register_buffer('total_tokens_seen', torch.tensor(0, dtype=torch.long))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse feature activations.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Feature activations of shape (..., d_sae)
        """
        # Center input
        x_centered = x - self.b_dec

        # Compute pre-activations
        pre_acts = x_centered @ self.W_enc + self.b_enc

        # Gating mechanism
        # Gate activation uses scaled pre-activations
        gate_acts = pre_acts * torch.exp(self.r_mag) + self.b_gate

        # Feature activations = ReLU(pre) * sigmoid(gate)
        feature_acts = F.relu(pre_acts) * torch.sigmoid(gate_acts)

        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode feature activations to reconstruction.

        Args:
            feature_acts: Feature activations of shape (..., d_sae)

        Returns:
            Reconstruction of shape (..., d_model)
        """
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        """
        Forward pass with loss computation.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            SAEOutput with reconstruction, features, and losses
        """
        # Encode
        feature_acts = self.encode(x)

        # Decode
        reconstruction = self.decode(feature_acts)

        # Compute losses
        l2_loss = F.mse_loss(reconstruction, x)
        l1_loss = feature_acts.abs().sum(dim=-1).mean()
        total_loss = l2_loss + self.config.l1_coefficient * l1_loss

        # Track feature activations
        if self.training:
            active_features = (feature_acts > 0).float().sum(dim=0)
            self.feature_activation_counts += active_features.long().sum(dim=0) if active_features.dim() > 1 else active_features.long()
            self.total_tokens_seen += x.shape[0] * (x.shape[1] if x.dim() > 2 else 1)

        # Count dead features
        dead_features = (self.feature_activation_counts == 0).sum().item()

        return SAEOutput(
            reconstruction=reconstruction,
            feature_acts=feature_acts,
            l2_loss=l2_loss.item(),
            l1_loss=l1_loss.item(),
            total_loss=total_loss.item(),
            dead_features=dead_features
        )

    def get_feature_sparsity(self) -> torch.Tensor:
        """Get the activation frequency of each feature."""
        if self.total_tokens_seen > 0:
            return self.feature_activation_counts.float() / self.total_tokens_seen.float()
        return torch.zeros(self.config.d_sae)

    def get_dead_features(self) -> torch.Tensor:
        """Get indices of dead features (never activated)."""
        return torch.where(self.feature_activation_counts == 0)[0]

    def resample_dead_features(self, activations: torch.Tensor):
        """
        Resample dead features using high-loss examples.

        Args:
            activations: Training activations to use for resampling
        """
        dead_indices = self.get_dead_features()

        if len(dead_indices) == 0:
            return

        # Find high-loss examples
        with torch.no_grad():
            output = self(activations)
            losses = ((output.reconstruction - activations) ** 2).sum(dim=-1)

            # Sample from high-loss examples
            probs = losses / losses.sum()
            sample_indices = torch.multinomial(probs.flatten(), len(dead_indices), replacement=True)

            # Get the activations for these examples
            sampled_activations = activations.view(-1, activations.shape[-1])[sample_indices]

            # Reinitialize dead features
            for i, dead_idx in enumerate(dead_indices):
                # New encoder weights point toward high-loss example
                direction = sampled_activations[i] - self.b_dec
                direction = direction / (direction.norm() + 1e-8)

                self.W_enc.data[:, dead_idx] = direction
                self.W_dec.data[dead_idx, :] = direction

                # Reset bias
                self.b_enc.data[dead_idx] = 0.0

            # Reset activation counts for resampled features
            self.feature_activation_counts[dead_indices] = 0

        print(f"Resampled {len(dead_indices)} dead features")


class SAETrainer:
    """
    Trainer for Gated SAE.
    """

    def __init__(self, sae: GatedSAE, config: SAEConfig):
        self.sae = sae
        self.config = config

        self.optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.training_history = {
            'l2_loss': [],
            'l1_loss': [],
            'total_loss': [],
            'dead_features': []
        }

    def train_step(self, activations: torch.Tensor) -> SAEOutput:
        """
        Single training step.

        Args:
            activations: Batch of activations to train on

        Returns:
            SAEOutput from forward pass
        """
        self.sae.train()
        self.optimizer.zero_grad()

        output = self.sae(activations)

        # Backprop
        loss = output.l2_loss + self.config.l1_coefficient * output.l1_loss
        loss_tensor = F.mse_loss(output.reconstruction, activations) + \
                      self.config.l1_coefficient * output.feature_acts.abs().sum(dim=-1).mean()
        loss_tensor.backward()

        self.optimizer.step()

        # Normalize decoder weights to unit norm
        with torch.no_grad():
            self.sae.W_dec.data = F.normalize(self.sae.W_dec.data, dim=1)

        # Track history
        self.training_history['l2_loss'].append(output.l2_loss)
        self.training_history['l1_loss'].append(output.l1_loss)
        self.training_history['total_loss'].append(output.total_loss)
        self.training_history['dead_features'].append(output.dead_features)

        return output

    def train(self,
              activations: torch.Tensor,
              n_epochs: int = 100,
              batch_size: int = 256,
              resample_dead_every: int = 25000,
              log_every: int = 100) -> Dict[str, List[float]]:
        """
        Train the SAE on a dataset of activations.

        Args:
            activations: Full dataset of activations
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            resample_dead_every: Resample dead features every N steps
            log_every: Log progress every N steps

        Returns:
            Training history dictionary
        """
        n_samples = activations.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        step = 0
        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            activations = activations[perm]

            epoch_l2 = 0.0
            epoch_l1 = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch = activations[start:end]

                output = self.train_step(batch)
                epoch_l2 += output.l2_loss
                epoch_l1 += output.l1_loss

                step += 1

                # Resample dead features
                if step % resample_dead_every == 0:
                    self.sae.resample_dead_features(activations)

                # Logging
                if step % log_every == 0:
                    dead = output.dead_features
                    sparsity = (output.feature_acts > 0).float().mean().item()
                    print(f"Step {step}: L2={output.l2_loss:.4f}, "
                          f"L1={output.l1_loss:.4f}, Dead={dead}, "
                          f"Sparsity={sparsity:.4f}")

            avg_l2 = epoch_l2 / n_batches
            avg_l1 = epoch_l1 / n_batches
            print(f"Epoch {epoch + 1}/{n_epochs}: L2={avg_l2:.4f}, L1={avg_l1:.4f}")

        return self.training_history


class SAEFeatureAnalyzer:
    """
    Analyze and interpret SAE features.
    """

    def __init__(self, sae: GatedSAE):
        self.sae = sae

    def get_top_activating_examples(self,
                                    feature_idx: int,
                                    activations: torch.Tensor,
                                    inputs: Optional[torch.Tensor] = None,
                                    k: int = 10) -> Dict[str, Any]:
        """
        Find examples that most strongly activate a feature.

        Args:
            feature_idx: Index of feature to analyze
            activations: Dataset activations
            inputs: Optional corresponding inputs
            k: Number of top examples to return

        Returns:
            Dictionary with top examples and their activations
        """
        with torch.no_grad():
            feature_acts = self.sae.encode(activations)
            feature_values = feature_acts[..., feature_idx]

            # Get top k
            if feature_values.dim() > 1:
                feature_values = feature_values.flatten()

            top_values, top_indices = torch.topk(feature_values, min(k, len(feature_values)))

        result = {
            'feature_idx': feature_idx,
            'top_activation_values': top_values.cpu().numpy().tolist(),
            'top_indices': top_indices.cpu().numpy().tolist(),
        }

        if inputs is not None:
            result['top_inputs'] = inputs.view(-1, inputs.shape[-1])[top_indices].cpu().numpy().tolist()

        return result

    def compute_feature_statistics(self,
                                  activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute statistics for all features.

        Args:
            activations: Dataset activations

        Returns:
            Dictionary of feature statistics
        """
        with torch.no_grad():
            feature_acts = self.sae.encode(activations)

            # Flatten batch dimensions
            if feature_acts.dim() > 2:
                feature_acts = feature_acts.view(-1, feature_acts.shape[-1])

            # Compute statistics
            mean_activation = feature_acts.mean(dim=0)
            max_activation = feature_acts.max(dim=0).values
            activation_frequency = (feature_acts > 0).float().mean(dim=0)
            std_activation = feature_acts.std(dim=0)

        return {
            'mean_activation': mean_activation,
            'max_activation': max_activation,
            'activation_frequency': activation_frequency,
            'std_activation': std_activation,
            'dead_features': (activation_frequency == 0).sum().item(),
            'ultra_sparse_features': (activation_frequency < 0.01).sum().item(),
        }

    def compute_feature_concept_correlation(self,
                                           activations: torch.Tensor,
                                           concept_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between SAE features and ground truth concepts.

        Args:
            activations: Dataset activations
            concept_labels: Ground truth concept labels

        Returns:
            Correlation matrix of shape (n_features, n_concepts)
        """
        with torch.no_grad():
            feature_acts = self.sae.encode(activations)

            if feature_acts.dim() > 2:
                feature_acts = feature_acts.view(-1, feature_acts.shape[-1])

            # One-hot encode concepts
            n_concepts = concept_labels.max().item() + 1
            concept_one_hot = F.one_hot(concept_labels.flatten(), n_concepts).float()

            # Compute correlation
            # Normalize both matrices
            feature_acts_centered = feature_acts - feature_acts.mean(dim=0, keepdim=True)
            concept_centered = concept_one_hot - concept_one_hot.mean(dim=0, keepdim=True)

            feature_std = feature_acts_centered.std(dim=0, keepdim=True) + 1e-8
            concept_std = concept_centered.std(dim=0, keepdim=True) + 1e-8

            feature_normalized = feature_acts_centered / feature_std
            concept_normalized = concept_centered / concept_std

            correlation = (feature_normalized.T @ concept_normalized) / feature_acts.shape[0]

        return correlation

    def find_concept_features(self,
                             activations: torch.Tensor,
                             concept_labels: torch.Tensor,
                             threshold: float = 0.5) -> Dict[int, List[int]]:
        """
        Find features that correspond to specific concepts.

        Args:
            activations: Dataset activations
            concept_labels: Ground truth concept labels
            threshold: Correlation threshold for assignment

        Returns:
            Dictionary mapping concept ID to list of feature indices
        """
        correlation = self.compute_feature_concept_correlation(activations, concept_labels)

        concept_features = {}
        n_concepts = correlation.shape[1]

        for concept_idx in range(n_concepts):
            # Find features strongly correlated with this concept
            corr_values = correlation[:, concept_idx]
            strong_features = torch.where(corr_values > threshold)[0].tolist()
            concept_features[concept_idx] = strong_features

        return concept_features


def create_sae(d_model: int,
               expansion_factor: int = 4,
               l1_coefficient: float = 1e-3) -> GatedSAE:
    """
    Create a Gated SAE with recommended settings.

    Args:
        d_model: Input dimension
        expansion_factor: SAE hidden dimension = d_model * expansion_factor
        l1_coefficient: L1 regularization strength

    Returns:
        Configured GatedSAE
    """
    config = SAEConfig(
        d_model=d_model,
        d_sae=d_model * expansion_factor,
        l1_coefficient=l1_coefficient
    )
    return GatedSAE(config)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    print("Testing Gated SAE Module")
    print("=" * 50)

    # Create SAE
    d_model = 64
    d_sae = 256
    config = SAEConfig(d_model=d_model, d_sae=d_sae, l1_coefficient=1e-3)
    sae = GatedSAE(config)
    print(f"Created Gated SAE: {d_model} -> {d_sae} -> {d_model}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, d_model)
    output = sae(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {output.reconstruction.shape}")
    print(f"Feature activations shape: {output.feature_acts.shape}")
    print(f"L2 loss: {output.l2_loss:.4f}")
    print(f"L1 loss: {output.l1_loss:.4f}")
    print(f"Dead features: {output.dead_features}")

    # Test sparsity
    sparsity = (output.feature_acts > 0).float().mean()
    print(f"Activation sparsity: {sparsity:.4f}")

    # Test training
    print("\nTraining SAE...")
    trainer = SAETrainer(sae, config)

    # Generate synthetic training data
    train_data = torch.randn(1000, d_model)

    history = trainer.train(
        train_data,
        n_epochs=5,
        batch_size=64,
        log_every=50
    )

    print(f"\nFinal L2 loss: {history['l2_loss'][-1]:.4f}")

    # Test feature analysis
    analyzer = SAEFeatureAnalyzer(sae)
    stats = analyzer.compute_feature_statistics(train_data)
    print(f"\nFeature statistics:")
    print(f"  Dead features: {stats['dead_features']}")
    print(f"  Ultra-sparse features: {stats['ultra_sparse_features']}")
    print(f"  Mean activation frequency: {stats['activation_frequency'].mean():.4f}")

    print("\nGated SAE module implementation complete!")
