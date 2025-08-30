"""
Activation extraction pipeline for neural network analysis.

This module provides tools for extracting and analyzing activations from
trained neural networks, specifically for concept analysis and visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict


class ActivationExtractor:
    """
    Extract activations from neural network models for analysis.
    
    This class provides a unified interface for extracting activations from
    both transformer and Mamba models, with support for batch processing
    and selective layer extraction.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Storage for activations
        self.activations = {}
        self.hooks = []
        
        # Available layers for extraction
        self.available_layers = self._get_available_layers()
    
    def _get_available_layers(self) -> List[str]:
        """Get list of available layers for activation extraction."""
        # Model-specific layer identification
        if hasattr(self.model, 'register_activation_hook'):
            # Our custom models with built-in hooks
            if hasattr(self.model, 'num_layers'):  # Mamba model
                layers = ['embeddings', 'aggregated'] + [f'layer_{i}' for i in range(self.model.num_layers)]
            else:  # Transformer model
                layers = ['embeddings', 'transformer_output', 'aggregated']
        else:
            # Generic model - identify layers automatically
            layers = []
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    layers.append(name)
        
        return layers
    
    def register_hooks(self, layer_names: List[str]):
        """Register forward hooks for specified layers."""
        self.clear_hooks()
        
        for layer_name in layer_names:
            if hasattr(self.model, 'register_activation_hook'):
                # Use model's built-in hook system
                self.model.register_activation_hook(layer_name)
            else:
                # Register manual hooks for generic models
                self._register_manual_hook(layer_name)
    
    def _register_manual_hook(self, layer_name: str):
        """Register manual forward hook for a layer."""
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach().cpu()
        
        # Find the module by name
        module = dict(self.model.named_modules()).get(layer_name)
        if module is not None:
            hook = module.register_forward_hook(hook_fn)
            self.hooks.append(hook)
    
    def clear_hooks(self):
        """Clear all registered hooks."""
        # Clear manual hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Clear model's built-in hooks
        if hasattr(self.model, 'clear_activations'):
            self.model.clear_activations()
        
        # Clear stored activations
        self.activations.clear()
    
    def extract_activations(self, 
                          inputs: torch.Tensor,
                          layer_names: Optional[List[str]] = None,
                          batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract activations for given inputs.
        
        Args:
            inputs: Input tensor of shape (num_samples, input_dim)
            layer_names: List of layer names to extract from
            batch_size: Batch size for processing (None = process all at once)
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if layer_names is None:
            layer_names = ['embeddings', 'aggregated']  # Default layers
        
        # Register hooks
        self.register_hooks(layer_names)
        
        # Process inputs
        inputs = inputs.to(self.device)
        
        if batch_size is None or len(inputs) <= batch_size:
            # Process all inputs at once
            with torch.no_grad():
                _ = self.model(inputs)
                
            # Get activations
            if hasattr(self.model, 'get_activations'):
                activations = self.model.get_activations()
            else:
                activations = self.activations.copy()
        else:
            # Process in batches
            all_activations = defaultdict(list)
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size]
                
                with torch.no_grad():
                    _ = self.model(batch_inputs)
                
                # Get batch activations
                if hasattr(self.model, 'get_activations'):
                    batch_activations = self.model.get_activations()
                else:
                    batch_activations = self.activations.copy()
                
                # Accumulate activations
                for layer_name, activation in batch_activations.items():
                    all_activations[layer_name].append(activation.cpu())
                
                # Clear activations for next batch
                if hasattr(self.model, 'clear_activations'):
                    self.model.clear_activations()
                self.activations.clear()
            
            # Concatenate batch results
            activations = {}
            for layer_name, activation_list in all_activations.items():
                activations[layer_name] = torch.cat(activation_list, dim=0)
        
        # Clean up
        self.clear_hooks()
        
        return activations
    
    def extract_number_representations(self, 
                                     layer_name: str = 'embeddings',
                                     vocab_size: Optional[int] = None) -> torch.Tensor:
        """
        Extract representations for all numbers 0 through vocab_size-1.
        
        Args:
            layer_name: Layer to extract representations from
            vocab_size: Vocabulary size (auto-detected if None)
            
        Returns:
            Tensor of shape (vocab_size, representation_dim)
        """
        if vocab_size is None:
            vocab_size = getattr(self.model, 'vocab_size', 17)
        
        if layer_name == 'embeddings' and hasattr(self.model, 'get_number_embeddings'):
            # Use model's built-in method for embeddings
            return self.model.get_number_embeddings()
        
        # Create inputs for all numbers (paired with 0)
        inputs = torch.stack([
            torch.arange(vocab_size),
            torch.zeros(vocab_size, dtype=torch.long)
        ], dim=1)
        
        # Extract activations
        activations = self.extract_activations(inputs, [layer_name])
        
        # Return representations for the first element in each pair
        if layer_name in activations:
            representations = activations[layer_name]
            if len(representations.shape) == 3:  # (batch, seq, hidden)
                representations = representations[:, 0, :]  # Take first token
            return representations
        else:
            raise ValueError(f"Could not extract representations from layer: {layer_name}")
    
    def extract_pairwise_activations(self, 
                                   pairs: List[Tuple[int, int]],
                                   layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract activations for specific input pairs.
        
        Args:
            pairs: List of (a, b) pairs
            layer_names: Layers to extract from
            
        Returns:
            Dictionary of activations for the specified pairs
        """
        inputs = torch.tensor(pairs, dtype=torch.long)
        return self.extract_activations(inputs, layer_names)
    
    def analyze_activation_statistics(self, 
                                    activations: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for extracted activations.
        
        Args:
            activations: Dictionary of layer activations
            
        Returns:
            Dictionary of statistics for each layer
        """
        stats = {}
        
        for layer_name, activation in activations.items():
            # Flatten activation for statistics
            flat_activation = activation.view(activation.shape[0], -1)
            
            stats[layer_name] = {
                'shape': list(activation.shape),
                'mean': float(torch.mean(flat_activation)),
                'std': float(torch.std(flat_activation)),
                'min': float(torch.min(flat_activation)),
                'max': float(torch.max(flat_activation)),
                'norm_mean': float(torch.mean(torch.norm(flat_activation, dim=1))),
                'sparsity': float(torch.mean((flat_activation == 0).float())),
                'total_elements': activation.numel()
            }
        
        return stats
    
    def save_activations(self, 
                        activations: Dict[str, torch.Tensor],
                        filepath: Path,
                        include_metadata: bool = True):
        """Save activations to file."""
        save_data = {
            'activations': activations,
            'layer_names': list(activations.keys()),
            'shapes': {k: list(v.shape) for k, v in activations.items()}
        }
        
        if include_metadata:
            save_data['statistics'] = self.analyze_activation_statistics(activations)
            save_data['model_info'] = {
                'model_type': type(self.model).__name__,
                'vocab_size': getattr(self.model, 'vocab_size', None),
                'd_model': getattr(self.model, 'd_model', None)
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_activations(self, filepath: Path) -> Dict[str, torch.Tensor]:
        """Load activations from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        return save_data['activations']


class BatchActivationExtractor:
    """
    Efficient batch processing for activation extraction across datasets.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.extractor = ActivationExtractor(model, device)
    
    def extract_dataset_activations(self,
                                  dataset,
                                  layer_names: List[str],
                                  batch_size: int = 64,
                                  save_path: Optional[Path] = None) -> Dict[str, torch.Tensor]:
        """
        Extract activations for an entire dataset.
        
        Args:
            dataset: ModularArithmeticDataset instance
            layer_names: Layers to extract from
            batch_size: Batch size for processing
            save_path: Path to save results
            
        Returns:
            Dictionary of activations for all samples
        """
        inputs = dataset.data['inputs']
        activations = self.extractor.extract_activations(
            inputs, layer_names, batch_size=batch_size
        )
        
        if save_path is not None:
            self.extractor.save_activations(activations, save_path)
        
        return activations
    
    def extract_concept_activations(self,
                                  concept_inputs: Dict[str, torch.Tensor],
                                  layer_names: List[str],
                                  batch_size: int = 64) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract activations for different concept groups.
        
        Args:
            concept_inputs: Dictionary mapping concept names to input tensors
            layer_names: Layers to extract from
            batch_size: Batch size for processing
            
        Returns:
            Nested dictionary: concept -> layer -> activations
        """
        concept_activations = {}
        
        for concept_name, inputs in concept_inputs.items():
            activations = self.extractor.extract_activations(
                inputs, layer_names, batch_size=batch_size
            )
            concept_activations[concept_name] = activations
        
        return concept_activations


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from Dataset.dataset import ModularArithmeticDataset
    from models.transformer import create_model
    
    # Create test data
    dataset = ModularArithmeticDataset(p=5)
    model = create_model(vocab_size=5)
    
    # Test activation extraction
    extractor = ActivationExtractor(model)
    print(f"Available layers: {extractor.available_layers}")
    
    # Extract activations for a few samples
    test_inputs = dataset.data['inputs'][:8]
    activations = extractor.extract_activations(
        test_inputs, 
        layer_names=['embeddings', 'aggregated']
    )
    
    print(f"Extracted activations:")
    for layer, activation in activations.items():
        print(f"  {layer}: {activation.shape}")
    
    # Test number representations
    number_reps = extractor.extract_number_representations('embeddings')
    print(f"Number representations: {number_reps.shape}")
    
    # Test statistics
    stats = extractor.analyze_activation_statistics(activations)
    for layer, layer_stats in stats.items():
        print(f"{layer} stats: mean={layer_stats['mean']:.3f}, std={layer_stats['std']:.3f}")
    
    print("✅ Activation extractor implementation complete!")