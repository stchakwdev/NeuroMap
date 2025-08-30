"""
Concept extraction methods for neural network analysis.

This module implements various methods for extracting interpretable concepts
from neural network activations, including clustering, probing, and sparse
autoencoder approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict


class ClusteringExtractor:
    """
    Extract concepts using clustering methods on neural activations.
    
    This class implements various clustering approaches to identify
    distinct concept groups in the activation space.
    """
    
    def __init__(self, n_concepts: int = 17, method: str = 'kmeans', random_state: int = 42):
        self.n_concepts = n_concepts
        self.method = method
        self.random_state = random_state
        self.clusterer = None
        self.concept_centers = None
        self.concept_assignments = None
    
    def extract_concepts(self, 
                        activations: torch.Tensor,
                        labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Extract concepts using clustering.
        
        Args:
            activations: Tensor of shape (n_samples, activation_dim)
            labels: Optional ground truth labels for validation
            
        Returns:
            Dictionary with concept information
        """
        # Convert to numpy for sklearn
        X = activations.cpu().numpy()
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)  # Flatten
        
        # Apply clustering
        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_concepts, 
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = self.clusterer.fit_predict(X)
            self.concept_centers = torch.tensor(self.clusterer.cluster_centers_)
            
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = self.clusterer.fit_predict(X)
            # Get unique cluster labels (excluding noise points labeled as -1)
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]
            self.n_concepts = len(unique_labels)
            
            # Compute cluster centers manually for DBSCAN
            centers = []
            for label in unique_labels:
                mask = cluster_labels == label
                center = np.mean(X[mask], axis=0)
                centers.append(center)
            self.concept_centers = torch.tensor(np.array(centers)) if centers else None
            
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.concept_assignments = torch.tensor(cluster_labels)
        
        # Analyze concept quality
        concept_info = self._analyze_concepts(activations, cluster_labels, labels)
        
        return concept_info
    
    def _analyze_concepts(self, 
                         activations: torch.Tensor,
                         cluster_labels: np.ndarray,
                         true_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Analyze quality and interpretability of extracted concepts."""
        
        concept_info = {
            'n_concepts_found': len(np.unique(cluster_labels[cluster_labels != -1])),
            'concept_assignments': cluster_labels,
            'concept_centers': self.concept_centers,
            'concept_sizes': [],
            'concept_coherence': [],
            'silhouette_scores': []
        }
        
        # Compute concept sizes and coherence
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        
        for label in unique_labels:
            mask = cluster_labels == label
            concept_size = np.sum(mask)
            concept_info['concept_sizes'].append(concept_size)
            
            # Compute intra-cluster coherence (average distance to center)
            if self.concept_centers is not None and label < len(self.concept_centers):
                center = self.concept_centers[label].numpy()
                concept_activations = activations[mask].cpu().numpy()
                if concept_activations.ndim > 2:
                    concept_activations = concept_activations.reshape(concept_activations.shape[0], -1)
                
                distances = np.linalg.norm(concept_activations - center, axis=1)
                coherence = np.mean(distances)
                concept_info['concept_coherence'].append(coherence)
        
        # Compute purity with respect to ground truth if available
        if true_labels is not None:
            purity = self._compute_purity(cluster_labels, true_labels.cpu().numpy())
            concept_info['purity'] = purity
            
            # Concept-to-label mapping
            concept_info['concept_label_mapping'] = self._map_concepts_to_labels(
                cluster_labels, true_labels.cpu().numpy()
            )
        
        return concept_info
    
    def _compute_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute cluster purity with respect to true labels."""
        total_samples = len(cluster_labels)
        weighted_purity = 0
        
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_true_labels = true_labels[cluster_mask]
            
            # Find most common true label in this cluster
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            max_count = np.max(counts)
            cluster_size = len(cluster_true_labels)
            
            # Weight purity by cluster size
            weighted_purity += (max_count / cluster_size) * (cluster_size / total_samples)
        
        return weighted_purity
    
    def _map_concepts_to_labels(self, 
                              cluster_labels: np.ndarray, 
                              true_labels: np.ndarray) -> Dict[int, int]:
        """Map each concept to its most common true label."""
        concept_mapping = {}
        
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_true_labels = true_labels[cluster_mask]
            
            # Find most common true label
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            concept_mapping[cluster] = most_common_label
        
        return concept_mapping
    
    def get_concept_representatives(self, 
                                 activations: torch.Tensor,
                                 n_representatives: int = 5) -> Dict[int, List[int]]:
        """Get representative samples for each concept."""
        if self.concept_assignments is None:
            raise ValueError("Must extract concepts first")
        
        representatives = {}
        unique_concepts = torch.unique(self.concept_assignments)
        unique_concepts = unique_concepts[unique_concepts != -1]  # Exclude noise
        
        for concept in unique_concepts:
            concept_mask = self.concept_assignments == concept
            concept_indices = torch.where(concept_mask)[0]
            
            if len(concept_indices) > 0:
                # Get closest samples to concept center
                if self.concept_centers is not None and concept < len(self.concept_centers):
                    center = self.concept_centers[concept]
                    concept_activations = activations[concept_mask]
                    
                    if concept_activations.ndim > 2:
                        concept_activations = concept_activations.view(concept_activations.shape[0], -1)
                        center = center.view(-1)
                    
                    distances = torch.norm(concept_activations - center, dim=1)
                    _, closest_indices = torch.topk(distances, min(n_representatives, len(distances)), largest=False)
                    
                    representatives[concept.item()] = concept_indices[closest_indices].tolist()
                else:
                    # Random selection if no centers available
                    n_select = min(n_representatives, len(concept_indices))
                    selected = torch.randperm(len(concept_indices))[:n_select]
                    representatives[concept.item()] = concept_indices[selected].tolist()
        
        return representatives


class ProbeExtractor:
    """
    Extract concepts using linear probing for specific properties.
    
    This class trains linear classifiers to identify neurons that encode
    specific mathematical properties in modular arithmetic.
    """
    
    def __init__(self, vocab_size: int = 17, random_state: int = 42):
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.probes = {}
        self.probe_accuracies = {}
    
    def create_property_labels(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create labels for various mathematical properties.
        
        Args:
            inputs: Input tensor of shape (n_samples, 2) with (a, b) pairs
            
        Returns:
            Dictionary mapping property names to binary labels
        """
        a, b = inputs[:, 0], inputs[:, 1]
        p = self.vocab_size
        
        properties = {}
        
        # Even/odd properties
        properties['a_is_even'] = (a % 2 == 0).float()
        properties['b_is_even'] = (b % 2 == 0).float()
        properties['sum_is_even'] = ((a + b) % 2 == 0).float()
        
        # Zero properties
        properties['a_is_zero'] = (a == 0).float()
        properties['b_is_zero'] = (b == 0).float()
        properties['sum_is_zero'] = ((a + b) % p == 0).float()
        
        # Adjacency properties (numbers differ by 1 mod p)
        properties['adjacent_forward'] = ((b - a) % p == 1).float()
        properties['adjacent_backward'] = ((a - b) % p == 1).float()
        properties['adjacent_any'] = torch.logical_or(
            (b - a) % p == 1, (a - b) % p == 1
        ).float()
        
        # Equality
        properties['a_equals_b'] = (a == b).float()
        
        # Magnitude properties (for small p)
        if p <= 23:
            properties['a_greater_than_half'] = (a > p // 2).float()
            properties['b_greater_than_half'] = (b > p // 2).float()
        
        # Identity properties
        properties['identity_addition'] = torch.logical_or(a == 0, b == 0).float()
        
        return properties
    
    def train_probes(self, 
                    activations: torch.Tensor,
                    inputs: torch.Tensor,
                    layer_name: str = 'default',
                    test_ratio: float = 0.2) -> Dict[str, float]:
        """
        Train linear probes for various properties.
        
        Args:
            activations: Activation tensor of shape (n_samples, activation_dim)
            inputs: Input tensor of shape (n_samples, 2)
            layer_name: Name of the layer being probed
            test_ratio: Fraction of data to use for testing
            
        Returns:
            Dictionary of probe accuracies
        """
        # Flatten activations if needed
        if activations.ndim > 2:
            activations = activations.view(activations.shape[0], -1)
        
        X = activations.cpu().numpy()
        
        # Create property labels
        properties = self.create_property_labels(inputs)
        
        # Split data
        n_samples = len(X)
        n_train = int(n_samples * (1 - test_ratio))
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Train probes for each property
        layer_probes = {}
        layer_accuracies = {}
        
        for prop_name, labels in properties.items():
            y = labels.cpu().numpy()
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Skip properties with only one class
            if len(np.unique(y_train)) < 2:
                continue
            
            # Train logistic regression probe
            probe = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            probe.fit(X_train, y_train)
            
            # Evaluate
            y_pred = probe.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            layer_probes[prop_name] = probe
            layer_accuracies[prop_name] = accuracy
        
        # Store results
        self.probes[layer_name] = layer_probes
        self.probe_accuracies[layer_name] = layer_accuracies
        
        return layer_accuracies
    
    def get_important_neurons(self, 
                            layer_name: str,
                            property_name: str,
                            top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get most important neurons for a specific property.
        
        Args:
            layer_name: Name of the probed layer
            property_name: Name of the property
            top_k: Number of top neurons to return
            
        Returns:
            Tuple of (neuron_indices, importance_scores)
        """
        if layer_name not in self.probes or property_name not in self.probes[layer_name]:
            raise ValueError(f"Probe not found: {layer_name}.{property_name}")
        
        probe = self.probes[layer_name][property_name]
        coefficients = np.abs(probe.coef_[0])  # Take absolute value for importance
        
        top_indices = np.argsort(coefficients)[-top_k:][::-1]  # Sort descending
        top_scores = coefficients[top_indices]
        
        return top_indices, top_scores
    
    def create_concept_from_probe(self, 
                                layer_name: str,
                                property_name: str,
                                activations: torch.Tensor) -> torch.Tensor:
        """
        Create concept vector from trained probe.
        
        Args:
            layer_name: Name of the probed layer
            property_name: Name of the property
            activations: Activations to project
            
        Returns:
            Concept scores for each sample
        """
        if layer_name not in self.probes or property_name not in self.probes[layer_name]:
            raise ValueError(f"Probe not found: {layer_name}.{property_name}")
        
        probe = self.probes[layer_name][property_name]
        
        # Flatten activations if needed
        if activations.ndim > 2:
            activations = activations.view(activations.shape[0], -1)
        
        X = activations.cpu().numpy()
        concept_scores = probe.predict_proba(X)[:, 1]  # Probability of positive class
        
        return torch.tensor(concept_scores)


class SparseAutoencoderExtractor:
    """
    Extract concepts using Sparse Autoencoders (SAE).
    
    This class implements a simple sparse autoencoder for discovering
    sparse, interpretable features in neural activations.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 sparsity_lambda: float = 1e-3,
                 device: str = 'cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        self.device = device
        
        # Build autoencoder
        self.encoder = nn.Linear(input_dim, hidden_dim).to(device)
        self.decoder = nn.Linear(hidden_dim, input_dim).to(device)
        
        self.optimizer = None
        self.training_history = []
    
    def train(self, 
              activations: torch.Tensor,
              num_epochs: int = 100,
              batch_size: int = 64,
              learning_rate: float = 1e-3) -> List[float]:
        """
        Train sparse autoencoder on activations.
        
        Args:
            activations: Training activations
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training loss history
        """
        # Flatten activations if needed
        if activations.ndim > 2:
            activations = activations.view(activations.shape[0], -1)
        
        activations = activations.to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Training loop
        self.training_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for (batch_x,) in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                hidden = torch.relu(self.encoder(batch_x))
                reconstructed = self.decoder(hidden)
                
                # Reconstruction loss
                recon_loss = nn.MSELoss()(reconstructed, batch_x)
                
                # Sparsity regularization (L1 on hidden activations)
                sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(hidden))
                
                total_loss = recon_loss + sparsity_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_history.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return self.training_history
    
    def extract_sparse_features(self, activations: torch.Tensor) -> torch.Tensor:
        """Extract sparse features from activations."""
        if activations.ndim > 2:
            activations = activations.view(activations.shape[0], -1)
        
        activations = activations.to(self.device)
        
        with torch.no_grad():
            hidden = torch.relu(self.encoder(activations))
        
        return hidden
    
    def get_feature_statistics(self, activations: torch.Tensor) -> Dict[str, Any]:
        """Compute statistics for sparse features."""
        features = self.extract_sparse_features(activations)
        
        stats = {
            'feature_activations': features,
            'sparsity': torch.mean((features == 0).float(), dim=0),
            'mean_activation': torch.mean(features, dim=0),
            'max_activation': torch.max(features, dim=0)[0],
            'active_frequency': torch.mean((features > 0).float(), dim=0)
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from Dataset.dataset import ModularArithmeticDataset
    from models.transformer import create_model
    from analysis.activation_extractor import ActivationExtractor
    
    # Create test data
    dataset = ModularArithmeticDataset(p=7)
    model = create_model(vocab_size=7)
    
    # Extract activations
    extractor = ActivationExtractor(model)
    inputs = dataset.data['inputs']
    activations = extractor.extract_activations(inputs, ['aggregated'])
    agg_activations = activations['aggregated']
    
    print(f"Test activations shape: {agg_activations.shape}")
    
    # Test clustering
    clusterer = ClusteringExtractor(n_concepts=7)
    labels = dataset.data['targets']  # Use true labels for validation
    concept_info = clusterer.extract_concepts(agg_activations, labels)
    
    print(f"\nClustering results:")
    print(f"  Concepts found: {concept_info['n_concepts_found']}")
    print(f"  Concept sizes: {concept_info['concept_sizes']}")
    if 'purity' in concept_info:
        print(f"  Purity: {concept_info['purity']:.3f}")
    
    # Test probing
    prober = ProbeExtractor(vocab_size=7)
    probe_accs = prober.train_probes(agg_activations, inputs, 'aggregated')
    
    print(f"\nProbing results:")
    for prop, acc in probe_accs.items():
        print(f"  {prop}: {acc:.3f}")
    
    print("✅ Concept extractors implementation complete!")