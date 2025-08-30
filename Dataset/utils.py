"""Utility functions for modular arithmetic dataset handling."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple

def print_dataset_summary(dataset) -> None:
    """Print comprehensive summary of dataset."""
    
    print(f"=== Modular Arithmetic Dataset (p={dataset.p}) ===")
    print(f"Representation: {dataset.representation}")
    print(f"Total examples: {dataset.data['num_examples']}")
    print(f"Input shape: {dataset.data['inputs'].shape}")
    print(f"Target shape: {dataset.data['targets'].shape}")
    
    # Show sample examples
    print(f"\nSample examples:")
    for i in range(min(10, dataset.data['num_examples'])):
        a, b = dataset.data['raw_inputs'][i]
        result = dataset.data['raw_targets'][i]
        print(f"  {a:2d} + {b:2d} = {result:2d} (mod {dataset.p})")
    
    # Structural information
    print(f"\nStructural metadata:")
    metadata = dataset.metadata
    print(f"  Adjacent pairs: {len(metadata['circular_structure']['adjacency_pairs'])}")
    print(f"  Commutative pairs: {len(metadata['algebraic_properties']['commutative_pairs'])}")
    print(f"  Identity pairs: {len(metadata['algebraic_properties']['identity_pairs'])}")
    print(f"  Inverse pairs: {len(metadata['algebraic_properties']['inverse_pairs'])}")

def visualize_addition_table(p: int, save_path: str = None) -> None:
    """Create visualization of modular addition table."""
    
    # Create addition table
    table = np.zeros((p, p))
    for a in range(p):
        for b in range(p):
            table[a, b] = (a + b) % p
    
    # Create heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(table, cmap='viridis', aspect='equal')
    plt.colorbar(label='(a + b) mod p')
    plt.xlabel('b')
    plt.ylabel('a')
    plt.title(f'Modular Addition Table (mod {p})')
    
    # Add text annotations for small p
    if p <= 20:
        for a in range(p):
            for b in range(p):
                plt.text(b, a, f'{int(table[a, b])}', 
                        ha='center', va='center', 
                        color='white' if table[a, b] < p/2 else 'black')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_circular_structure(embeddings: torch.Tensor, p: int, 
                               save_path: str = None) -> None:
    """Visualize embeddings in 2D to check circular structure."""
    
    from sklearn.decomposition import PCA
    
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Project to 2D if needed
    if embeddings_np.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)
        title_suffix = f" (PCA: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})"
    else:
        embeddings_2d = embeddings_np
        title_suffix = ""
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: scatter with labels
    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=range(p), cmap='hsv', s=100)
    
    # Add number labels
    for i in range(p):
        ax1.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_title(f'Learned Embeddings (p={p}){title_suffix}')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right plot: reference circle
    angles = np.linspace(0, 2*np.pi, p, endpoint=False)
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    
    ax2.scatter(circle_x, circle_y, c=range(p), cmap='hsv', s=100)
    for i in range(p):
        ax2.annotate(str(i), (circle_x[i], circle_y[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_title('Expected Circular Structure')
    ax2.set_xlabel('cos(2πi/p)')
    ax2.set_ylabel('sin(2πi/p)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def check_dataset_consistency(dataset) -> Dict[str, bool]:
    """Run consistency checks on dataset."""
    
    checks = {}
    p = dataset.p
    
    # Check total number of examples
    expected_count = p * p
    actual_count = dataset.data['num_examples']
    checks['correct_example_count'] = (expected_count == actual_count)
    
    # Check that all (a,b) pairs are present exactly once
    seen_pairs = set()
    for a, b in dataset.data['raw_inputs']:
        if (a, b) in seen_pairs:
            checks['no_duplicate_pairs'] = False
            break
        seen_pairs.add((a, b))
    else:
        checks['no_duplicate_pairs'] = True
    
    # Check that all targets are correct
    correct_targets = True
    for i, (a, b) in enumerate(dataset.data['raw_inputs']):
        expected = (a + b) % p
        actual = dataset.data['raw_targets'][i]
        if expected != actual:
            correct_targets = False
            break
    checks['correct_targets'] = correct_targets
    
    # Check tensor shapes
    if dataset.representation == 'embedding':
        expected_input_shape = (p*p, 2)
        expected_target_shape = (p*p,)
    elif dataset.representation == 'one_hot':
        expected_input_shape = (p*p, 2, p)
        expected_target_shape = (p*p, p)
    else:  # integer
        expected_input_shape = (p*p, 2)
        expected_target_shape = (p*p,)
    
    checks['correct_input_shape'] = (dataset.data['inputs'].shape == expected_input_shape)
    checks['correct_target_shape'] = (dataset.data['targets'].shape == expected_target_shape)
    
    # Check metadata consistency
    checks['metadata_adjacency_count'] = (
        len(dataset.metadata['circular_structure']['adjacency_pairs']) == p
    )
    
    return checks

def create_training_splits(dataset, train_ratio: float = 0.8) -> Dict[str, torch.Tensor]:
    """Create train/val splits (though full dataset is small enough to use entirely)."""
    
    total_size = dataset.data['num_examples']
    train_size = int(total_size * train_ratio)
    
    # Random permutation
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return {
        'train_inputs': dataset.data['inputs'][train_indices],
        'train_targets': dataset.data['targets'][train_indices],
        'val_inputs': dataset.data['inputs'][val_indices],
        'val_targets': dataset.data['targets'][val_indices],
        'train_indices': train_indices,
        'val_indices': val_indices
    }

def analyze_algebraic_properties(dataset) -> Dict[str, Any]:
    """Analyze algebraic properties of the dataset."""
    
    p = dataset.p
    analysis = {}
    
    # Test commutativity
    commutative_violations = []
    for a in range(p):
        for b in range(p):
            idx_ab = a * p + b
            idx_ba = b * p + a
            result_ab = dataset.data['raw_targets'][idx_ab]
            result_ba = dataset.data['raw_targets'][idx_ba]
            if result_ab != result_ba:
                commutative_violations.append(((a, b), result_ab, result_ba))
    
    analysis['commutativity'] = {
        'is_commutative': len(commutative_violations) == 0,
        'violations': commutative_violations
    }
    
    # Test identity element
    identity_violations = []
    for a in range(p):
        idx = a * p + 0  # (a, 0)
        result = dataset.data['raw_targets'][idx]
        if result != a:
            identity_violations.append((a, result))
    
    analysis['identity'] = {
        'has_identity': len(identity_violations) == 0,
        'identity_element': 0,
        'violations': identity_violations
    }
    
    # Test inverse elements
    inverse_analysis = {}
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            result = dataset.data['raw_targets'][idx]
            if result == 0:  # Found inverse
                inverse_analysis[a] = b
                break
    
    analysis['inverses'] = {
        'all_have_inverses': len(inverse_analysis) == p,
        'inverse_map': inverse_analysis
    }
    
    return analysis

def export_dataset_summary(dataset, filepath: str) -> None:
    """Export comprehensive dataset summary to text file."""
    
    with open(filepath, 'w') as f:
        f.write(f"Modular Arithmetic Dataset Summary\n")
        f.write(f"==================================\n\n")
        
        f.write(f"Basic Information:\n")
        f.write(f"  Prime modulus (p): {dataset.p}\n")
        f.write(f"  Representation: {dataset.representation}\n")
        f.write(f"  Total examples: {dataset.data['num_examples']}\n")
        f.write(f"  Input shape: {dataset.data['inputs'].shape}\n")
        f.write(f"  Target shape: {dataset.data['targets'].shape}\n\n")
        
        # Consistency checks
        checks = check_dataset_consistency(dataset)
        f.write(f"Consistency Checks:\n")
        for check_name, result in checks.items():
            status = "PASS" if result else "FAIL"
            f.write(f"  {check_name}: {status}\n")
        f.write("\n")
        
        # Algebraic properties
        algebra = analyze_algebraic_properties(dataset)
        f.write(f"Algebraic Properties:\n")
        f.write(f"  Commutative: {algebra['commutativity']['is_commutative']}\n")
        f.write(f"  Has identity: {algebra['identity']['has_identity']}\n")
        f.write(f"  All have inverses: {algebra['inverses']['all_have_inverses']}\n\n")
        
        # Structural metadata summary
        metadata = dataset.metadata
        f.write(f"Structural Metadata:\n")
        f.write(f"  Adjacent pairs: {len(metadata['circular_structure']['adjacency_pairs'])}\n")
        f.write(f"  Commutative pairs: {len(metadata['algebraic_properties']['commutative_pairs'])}\n")
        f.write(f"  Identity pairs: {len(metadata['algebraic_properties']['identity_pairs'])}\n")
        f.write(f"  Inverse pairs: {len(metadata['algebraic_properties']['inverse_pairs'])}\n\n")
        
        # Sample examples
        f.write(f"Sample Examples:\n")
        for i in range(min(20, dataset.data['num_examples'])):
            a, b = dataset.data['raw_inputs'][i]
            result = dataset.data['raw_targets'][i]
            f.write(f"  {a:2d} + {b:2d} = {result:2d} (mod {dataset.p})\n")

def load_and_validate_dataset(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """Load dataset and run comprehensive validation."""
    
    from dataset import ModularArithmeticDataset
    
    # Load dataset
    dataset = ModularArithmeticDataset.load(filepath)
    
    # Run all validation checks
    validation_results = {
        'consistency_checks': check_dataset_consistency(dataset),
        'algebraic_properties': analyze_algebraic_properties(dataset)
    }
    
    return dataset, validation_results

if __name__ == "__main__":
    # Test utility functions with a sample dataset
    from dataset import ModularArithmeticDataset
    from validation import _generate_perfect_circle_embeddings
    
    print("Testing utility functions...")
    
    # Create test dataset
    dataset = ModularArithmeticDataset(p=7, representation='embedding')
    
    # Test dataset summary
    print("\n1. Dataset Summary:")
    print_dataset_summary(dataset)
    
    # Test consistency checks
    print("\n2. Consistency Checks:")
    checks = check_dataset_consistency(dataset)
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  {check_name}: {status}")
    
    # Test algebraic analysis
    print("\n3. Algebraic Properties:")
    algebra = analyze_algebraic_properties(dataset)
    print(f"  Commutative: {algebra['commutativity']['is_commutative']}")
    print(f"  Has identity: {algebra['identity']['has_identity']}")
    print(f"  All have inverses: {algebra['inverses']['all_have_inverses']}")
    
    # Test training splits
    print("\n4. Training Splits:")
    splits = create_training_splits(dataset, train_ratio=0.8)
    print(f"  Train size: {splits['train_inputs'].shape[0]}")
    print(f"  Val size: {splits['val_inputs'].shape[0]}")
    
    print("\nAll utility functions working correctly!")

