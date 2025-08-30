"""
Modular arithmetic dataset creation for neural network interpretability research.

This module creates complete datasets for learning f(a,b) = (a+b) mod p,
with emphasis on extracting and validating circular structure in learned representations.
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from config import DEFAULT_P, REPRESENTATION_TYPES, DATA_DIR

class ModularArithmeticDataset:
    """
    Complete modular arithmetic dataset with structural metadata.
    
    Creates all possible (a,b) -> (a+b) mod p examples plus rich metadata
    for validating that neural networks learn the correct circular structure.
    """
    
    def __init__(self, p: int = DEFAULT_P, representation: str = 'embedding'):
        """
        Initialize dataset.
        
        Args:
            p: Prime modulus (17 recommended for toy models)
            representation: How to encode inputs ('one_hot', 'embedding', 'integer')
        """
        if representation not in REPRESENTATION_TYPES:
            raise ValueError(f"representation must be one of {REPRESENTATION_TYPES}")
        
        self.p = p
        self.representation = representation
        self.data = None
        self.metadata = None
        
        # Generate dataset
        self._create_core_dataset()
        self._create_structural_metadata()
        
    def _create_core_dataset(self) -> None:
        """Generate all (a,b) -> (a+b) mod p examples."""
        
        # Generate all possible input pairs
        raw_inputs = []
        raw_targets = []
        
        for a in range(self.p):
            for b in range(self.p):
                raw_inputs.append((a, b))
                raw_targets.append((a + b) % self.p)
        
        # Convert to tensors based on representation
        if self.representation == 'one_hot':
            # Each number becomes p-dimensional one-hot vector
            input_tensor = torch.zeros(len(raw_inputs), 2, self.p)  # (p², 2, p)
            target_tensor = torch.zeros(len(raw_targets), self.p)    # (p², p)
            
            for i, (a, b) in enumerate(raw_inputs):
                input_tensor[i, 0, a] = 1  # First number one-hot
                input_tensor[i, 1, b] = 1  # Second number one-hot
                target_tensor[i, raw_targets[i]] = 1  # Target one-hot
                
        elif self.representation == 'embedding':
            # Use integer indices, let model learn embeddings
            input_tensor = torch.tensor(raw_inputs, dtype=torch.long)  # (p², 2)
            target_tensor = torch.tensor(raw_targets, dtype=torch.long)  # (p²,)
            
        elif self.representation == 'integer':
            # Direct integer representation
            input_tensor = torch.tensor(raw_inputs, dtype=torch.float32)  # (p², 2)
            target_tensor = torch.tensor(raw_targets, dtype=torch.long)   # (p²,)
            
        self.data = {
            'inputs': input_tensor,
            'targets': target_tensor,
            'raw_inputs': raw_inputs,
            'raw_targets': raw_targets,
            'p': self.p,
            'representation': self.representation,
            'num_examples': len(raw_inputs)
        }
        
    def _create_structural_metadata(self) -> None:
        """Create metadata about expected mathematical structure."""
        
        # Create metadata components in order
        circular_metadata = self._create_circular_metadata()
        algebraic_metadata = self._create_algebraic_metadata()
        distance_metadata = self._create_distance_metadata()
        
        # Set partial metadata first
        self.metadata = {
            'circular_structure': circular_metadata,
            'algebraic_properties': algebraic_metadata,
            'distance_matrices': distance_metadata,
        }
        
        # Now create validation sets that depend on other metadata
        validation_metadata = self._create_validation_sets()
        self.metadata['validation_sets'] = validation_metadata
        
    def _create_circular_metadata(self) -> Dict[str, Any]:
        """Metadata about expected circular arrangement of numbers."""
        
        # Adjacent pairs that should be close in embedding space
        adjacency_pairs = []
        for i in range(self.p):
            next_i = (i + 1) % self.p
            adjacency_pairs.append((i, next_i))
        
        # Diameter pairs (maximally distant on circle)
        diameter_pairs = []
        for i in range(self.p):
            opposite = (i + self.p // 2) % self.p
            if i < opposite:  # Avoid duplicates
                diameter_pairs.append((i, opposite))
        
        return {
            'adjacency_pairs': adjacency_pairs,
            'diameter_pairs': diameter_pairs,
            'expected_radius_consistency': True,
            'expected_angular_spacing': 2 * np.pi / self.p
        }
        
    def _create_algebraic_metadata(self) -> Dict[str, Any]:
        """Metadata about algebraic properties of modular addition."""
        
        # Commutative pairs: (a,b) and (b,a) should give same result
        commutative_pairs = []
        for a in range(self.p):
            for b in range(a + 1, self.p):  # Avoid duplicates and (a,a) cases
                commutative_pairs.append(((a, b), (b, a)))
        
        # Identity element pairs: (a, 0) should give a
        identity_pairs = [(a, 0) for a in range(self.p)]
        
        # Inverse pairs: (a, p-a) should give 0 (except for a=0)
        inverse_pairs = []
        for a in range(1, self.p):
            inverse = (-a) % self.p
            if a <= inverse:  # Avoid duplicates
                inverse_pairs.append((a, inverse))
        
        # Complete addition table for reference
        addition_table = torch.zeros(self.p, self.p, dtype=torch.long)
        for a in range(self.p):
            for b in range(self.p):
                addition_table[a, b] = (a + b) % self.p
        
        return {
            'commutative_pairs': commutative_pairs,
            'identity_pairs': identity_pairs,
            'inverse_pairs': inverse_pairs,
            'addition_table': addition_table,
            'is_abelian_group': True,
            'identity_element': 0
        }
        
    def _create_distance_metadata(self) -> Dict[str, torch.Tensor]:
        """Create various distance matrices for structure validation."""
        
        # Circular distance matrix (shortest path on circle)
        circular_distance = torch.zeros(self.p, self.p)
        for i in range(self.p):
            for j in range(self.p):
                clockwise = (j - i) % self.p
                counterclockwise = (i - j) % self.p
                circular_distance[i, j] = min(clockwise, counterclockwise)
        
        # Euclidean distance matrix (if numbers were on unit circle)
        angles = torch.tensor([2 * np.pi * i / self.p for i in range(self.p)])
        unit_circle_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        euclidean_distance = torch.cdist(unit_circle_coords, unit_circle_coords)
        
        return {
            'circular_distance': circular_distance,
            'euclidean_distance': euclidean_distance,
            'unit_circle_coords': unit_circle_coords,
            'angles': angles
        }
        
    def _create_validation_sets(self) -> Dict[str, Dict]:
        """Create specific test sets for model validation."""
        
        # Get indices for different test types
        def get_indices(input_pairs):
            indices = []
            for a, b in input_pairs:
                idx = a * self.p + b
                indices.append(idx)
            return indices
        
        validation_sets = {}
        
        # Commutativity test
        commutative_indices = []
        for (a, b), (b_rev, a_rev) in self.metadata['algebraic_properties']['commutative_pairs']:
            idx1 = a * self.p + b
            idx2 = b * self.p + a
            commutative_indices.append((idx1, idx2))
        
        validation_sets['commutativity'] = {
            'index_pairs': commutative_indices,
            'description': 'f(a,b) should equal f(b,a)',
            'expected_difference': 0
        }
        
        # Identity test
        identity_indices = get_indices(self.metadata['algebraic_properties']['identity_pairs'])
        validation_sets['identity'] = {
            'indices': identity_indices,
            'expected_outputs': list(range(self.p)),
            'description': 'f(a,0) should equal a'
        }
        
        # Inverse test
        inverse_indices = get_indices(self.metadata['algebraic_properties']['inverse_pairs'])
        validation_sets['inverse'] = {
            'indices': inverse_indices,
            'expected_output': 0,
            'description': 'f(a,p-a) should equal 0'
        }
        
        return validation_sets
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save complete dataset to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'metadata': self.metadata,
                'p': self.p,
                'representation': self.representation
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModularArithmeticDataset':
        """Load dataset from file."""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.data = saved_data['data']
        instance.metadata = saved_data['metadata']
        instance.p = saved_data['p']
        instance.representation = saved_data['representation']
        
        return instance
    
    def export_metadata_json(self, filepath: Union[str, Path]) -> None:
        """Export human-readable metadata to JSON."""
        
        def convert_tensors(obj):
            """Convert tensors to lists for JSON serialization."""
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        json_metadata = convert_tensors(self.metadata)
        json_metadata['dataset_info'] = {
            'p': self.p,
            'representation': self.representation,
            'num_examples': self.data['num_examples'],
            'input_shape': list(self.data['inputs'].shape),
            'target_shape': list(self.data['targets'].shape)
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_metadata, f, indent=2)

def create_mod_p_datasets(p_values: List[int] = [13, 17, 23]) -> Dict[int, ModularArithmeticDataset]:
    """Create datasets for multiple modulus values."""
    
    datasets = {}
    for p in p_values:
        print(f"Creating mod {p} dataset...")
        dataset = ModularArithmeticDataset(p=p, representation='embedding')
        datasets[p] = dataset
        
        # Save to files
        data_dir = Path(DATA_DIR)
        data_dir.mkdir(exist_ok=True)
        
        dataset.save(data_dir / f'mod_{p}_dataset.pkl')
        dataset.export_metadata_json(data_dir / f'mod_{p}_metadata.json')
        
        print(f"  Saved {dataset.data['num_examples']} examples")
    
    return datasets

if __name__ == "__main__":
    # Create primary dataset
    print("Creating modular arithmetic datasets...")
    datasets = create_mod_p_datasets([17])  # Start with just p=17
    
    # Quick validation
    dataset = datasets[17]
    print(f"\nDataset validation:")
    print(f"  Shape: {dataset.data['inputs'].shape}")
    print(f"  Examples: {dataset.data['num_examples']}")
    print(f"  Representation: {dataset.representation}")
    
    # Show first few examples
    print(f"\nFirst 5 examples:")
    for i in range(5):
        a, b = dataset.data['raw_inputs'][i]
        result = dataset.data['raw_targets'][i]
        print(f"  {a} + {b} = {result} (mod {dataset.p})")
    
    print(f"\nFiles saved to {DATA_DIR}/")

