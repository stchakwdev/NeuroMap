#!/usr/bin/env python3
"""
Training script for alternative approaches to modular arithmetic learning.

Tests memory-augmented and cyclic encoding models on larger datasets (p=13, 17, 23).
These approaches should overcome the scaling challenges faced by standard neural networks.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple

# Add paths
sys.path.insert(0, 'Dataset')
sys.path.insert(0, 'models/alternative_approaches')

from dataset import ModularArithmeticDataset
from memory_models import create_memory_model
from cyclic_models import create_cyclic_model

# Enable deterministic behavior
torch.manual_seed(42)
np.random.seed(42)


def get_alternative_configs() -> List[Dict]:
    """Get configurations for alternative approaches."""
    
    configs = [
        # Memory-based approaches (should achieve 100% accuracy)
        {
            'approach': 'memory',
            'model_type': 'direct_lookup',
            'optimizer': 'adam',
            'learning_rate': 0.01,  # Can use higher LR for lookup tables
            'batch_size': 'full',   # Use full batch for memorization
            'weight_decay': 0.0,
            'name': 'DirectLookup_Adam'
        },
        {
            'approach': 'memory',
            'model_type': 'memory_augmented',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-5,
            'name': 'MemoryAugmented_Adam'
        },
        {
            'approach': 'memory',
            'model_type': 'hybrid_memory',
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'batch_size': 8,
            'weight_decay': 1e-4,
            'name': 'HybridMemory_AdamW'
        },
        
        # Cyclic encoding approaches (should learn patterns better)
        {
            'approach': 'cyclic',
            'model_type': 'cyclic_basic',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-4,
            'name': 'CyclicBasic_Adam'
        },
        {
            'approach': 'cyclic',
            'model_type': 'cyclic_enhanced',
            'optimizer': 'adamw',
            'learning_rate': 0.0005,
            'batch_size': 8,
            'weight_decay': 1e-4,
            'name': 'CyclicEnhanced_AdamW'
        },
        {
            'approach': 'cyclic',
            'model_type': 'cyclic_residual',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-4,
            'name': 'CyclicResidual_Adam'
        }
    ]
    
    return configs


def create_model(approach: str, model_type: str, vocab_size: int, device: str):
    """Create model based on approach and type."""
    
    if approach == 'memory':
        model = create_memory_model(model_type, vocab_size)
    elif approach == 'cyclic':
        model = create_cyclic_model(model_type, vocab_size)
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    return model.to(device)


def train_model(model, train_loader, val_loader, config: Dict, vocab_size: int, device: str) -> Dict:
    """Train a single model configuration."""
    
    print(f"\n--- Training {config['name']} ---")
    print(f"Approach: {config['approach']}, Model: {config['model_type']}")
    print(f"Learning rate: {config['learning_rate']}, Batch size: {config['batch_size']}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                            lr=config['learning_rate'],
                            momentum=0.9,
                            weight_decay=config['weight_decay'])
    
    # Training parameters
    max_epochs = 200
    patience = 100  # More patience for complex models
    best_accuracy = 0
    patience_counter = 0
    
    train_losses = []
    val_accuracies = []
    
    model.train()
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Training
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        # Check for improvement
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            best_state_dict = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress 
        if (epoch + 1) % 20 == 0 or accuracy >= 0.95 or epoch < 5:
            print(f"  Epoch {epoch+1:3d}: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}, "
                  f"Best {best_accuracy:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        # Perfect accuracy reached
        if accuracy >= 0.999:
            print(f"  🎉 Perfect accuracy reached at epoch {epoch+1}!")
            break
        
        model.train()
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(best_state_dict)
    
    success_marker = "🎉" if best_accuracy >= 0.95 else "✅" if best_accuracy >= 0.90 else "⚠️"
    print(f"  {success_marker} Training complete: Best accuracy {best_accuracy:.4f} in {training_time:.1f}s")
    
    return {
        'best_accuracy': best_accuracy,
        'final_loss': train_losses[-1] if train_losses else float('inf'),
        'epochs_trained': epoch + 1,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'model_state_dict': best_state_dict,
        'config': config
    }


def train_for_vocab_size(vocab_size: int) -> Dict:
    """Train all alternative approaches for a given vocabulary size."""
    
    print(f"\n{'='*70}")
    print(f"ALTERNATIVE APPROACHES FOR p={vocab_size}")
    print(f"Dataset size: {vocab_size * vocab_size} examples")
    print(f"Testing memory-based and cyclic encoding models")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = Path(f'data/mod_{vocab_size}_dataset.pkl')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = ModularArithmeticDataset.load(dataset_path)
    
    # Create train/val split (90/10 for larger datasets)
    inputs = dataset.data['inputs']
    targets = dataset.data['targets']
    dataset_size = len(inputs)
    
    # Shuffle indices with fixed seed
    torch.manual_seed(42)
    indices = torch.randperm(dataset_size)
    train_size = int(0.9 * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]
    
    print(f"Train examples: {len(train_inputs)}")
    print(f"Val examples: {len(val_inputs)}")
    
    # Get configurations
    configs = get_alternative_configs()
    results = {}
    
    # Create output directory
    output_dir = Path('models/alternative')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each configuration
    for config in configs:
        print(f"\n" + "-" * 50)
        
        # Adjust batch size
        batch_size = config['batch_size']
        if batch_size == 'full':
            batch_size = len(train_inputs)
        
        try:
            # Create model
            model = create_model(config['approach'], config['model_type'], vocab_size, device)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {param_count:,}")
            
            # Create data loaders
            train_dataset = TensorDataset(train_inputs, train_targets)
            val_dataset = TensorDataset(val_inputs, val_targets)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            # Train model
            result = train_model(model, train_loader, val_loader, config, vocab_size, device)
            result['parameter_count'] = param_count
            
            # Save successful models
            if result['best_accuracy'] >= 0.85:
                model_name = f"{config['name']}_p{vocab_size}"
                model_path = output_dir / f"{model_name}.pt"
                
                torch.save({
                    'model_state_dict': result['model_state_dict'],
                    'config': config,
                    'vocab_size': vocab_size,
                    'accuracy': result['best_accuracy'],
                    'parameter_count': param_count,
                    'training_results': result
                }, model_path)
                
                print(f"  💾 Saved model: {model_path}")
                result['saved_path'] = str(model_path)
            else:
                print(f"  ⚠️  Accuracy {result['best_accuracy']:.3f} below save threshold")
            
            results[config['name']] = result
            
        except Exception as e:
            print(f"  ❌ Error training {config['name']}: {e}")
            results[config['name']] = {'error': str(e), 'best_accuracy': 0.0}
            continue
    
    return results


def main():
    """Main training function for alternative approaches."""
    
    print("ALTERNATIVE APPROACHES TO MODULAR ARITHMETIC")
    print("Testing memory-based and cyclic encoding models")
    print("Should overcome scaling challenges of standard neural networks")
    print("=" * 80)
    
    vocab_sizes = [13]  # Start with p=13, then expand if successful
    all_results = {}
    
    for vocab_size in vocab_sizes:
        try:
            results = train_for_vocab_size(vocab_size)
            all_results[vocab_size] = results
        except Exception as e:
            print(f"❌ Error training p={vocab_size}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("ALTERNATIVE APPROACHES SUMMARY")
    print(f"{'='*80}")
    
    success_count = 0
    total_count = 0
    excellent_count = 0
    
    for vocab_size, results in all_results.items():
        print(f"\np={vocab_size} ({vocab_size*vocab_size} examples):")
        print(f"{'Model':<25} {'Accuracy':<10} {'Params':<8} {'Time':<6} {'Status'}")
        print("-" * 60)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"{model_name:<25} ERROR      -        -      ❌")
                total_count += 1
                continue
            
            total_count += 1
            accuracy = result['best_accuracy']
            params = result.get('parameter_count', 0)
            time_taken = result.get('training_time', 0)
            
            if accuracy >= 0.95:
                status = "🎉 EXCELLENT"
                excellent_count += 1
                success_count += 1
            elif accuracy >= 0.90:
                status = "✅ GOOD"
                success_count += 1
            elif accuracy >= 0.80:
                status = "⚠️ FAIR"
            else:
                status = "❌ POOR"
            
            saved = " [SAVED]" if 'saved_path' in result else ""
            print(f"{model_name:<25} {accuracy:<10.3f} {params/1000:<7.1f}K {time_taken:<6.1f}s {status}{saved}")
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"  • Excellent (≥95%): {excellent_count}/{total_count} ({100*excellent_count/total_count:.0f}%)")
    print(f"  • Success (≥90%):   {success_count}/{total_count} ({100*success_count/total_count:.0f}%)")
    
    # Save detailed results
    output_dir = Path('models/alternative')
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        'total_models': total_count,
        'successful_models': success_count,
        'excellent_models': excellent_count,
        'success_rate': success_count / total_count if total_count > 0 else 0,
        'excellent_rate': excellent_count / total_count if total_count > 0 else 0,
        'results_by_vocab_size': {}
    }
    
    for vocab_size, results in all_results.items():
        summary['results_by_vocab_size'][vocab_size] = {
            model_name: {
                'accuracy': result.get('best_accuracy', 0),
                'epochs': result.get('epochs_trained', 0),
                'time': result.get('training_time', 0),
                'saved': 'saved_path' in result,
                'error': result.get('error')
            }
            for model_name, result in results.items()
        }
    
    with open(output_dir / 'alternative_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir}/alternative_training_summary.json")
    
    # Final assessment
    if excellent_count > 0:
        print(f"\n🎉 BREAKTHROUGH! {excellent_count} models achieved ≥95% accuracy!")
        print("Alternative approaches successfully overcome scaling challenges!")
        
        if vocab_size == 13 and excellent_count > 0:
            print("\n🚀 NEXT STEPS:")
            print("1. Apply successful approaches to p=17 and p=23")
            print("2. Analyze which approach works best")
            print("3. Proceed with neural topology visualization")
    elif success_count > 0:
        print(f"\n✅ PROGRESS! {success_count} models achieved ≥90% accuracy!")
        print("Alternative approaches show significant improvement!")
    else:
        print(f"\n📚 LEARNING: Alternative approaches tested.")
        print("Results provide insights for future research directions.")


if __name__ == "__main__":
    main()