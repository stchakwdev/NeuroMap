#!/usr/bin/env python3
"""
Train scaled models for larger datasets using proven hyperparameter approach.

This script applies the lessons learned from p=7 to train models for p=13, 17, 23.
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

# Add Dataset to path
sys.path.insert(0, 'Dataset')

from dataset import ModularArithmeticDataset
from models.scaled_optimized_models import create_scaled_model

# Enable deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

def get_optimal_configs(vocab_size: int, dataset_size: int) -> List[Dict]:
    """Get optimal training configurations based on proven approach."""
    
    configs = []
    
    # Configuration 1: Full batch SGD (inspired by Linear_SGD_FullBatch success)
    configs.append({
        'model_type': 'scaled_linear',
        'optimizer': 'sgd',
        'learning_rate': max(0.02, 0.1 * (49 / dataset_size)),  # Scale down LR with dataset size
        'batch_size': dataset_size,  # Full batch
        'weight_decay': 1e-5,
        'name_suffix': 'FullBatch'
    })
    
    # Configuration 2: Small batch Adam (inspired by Linear_Adam_SmallBatch success)
    configs.append({
        'model_type': 'scaled_linear',
        'optimizer': 'adam',
        'learning_rate': max(0.0003, 0.001 * (49 / dataset_size)),  # Scale down slightly
        'batch_size': max(2, int(np.sqrt(dataset_size) / 4)),  # Very small batch
        'weight_decay': 1e-5,
        'name_suffix': 'SmallBatch'
    })
    
    # Configuration 3: MLP with medium batch (new approach for larger datasets)
    configs.append({
        'model_type': 'scaled_mlp',
        'optimizer': 'adamw',
        'learning_rate': max(0.0005, 0.002 * (49 / dataset_size)),
        'batch_size': max(4, int(np.sqrt(dataset_size) / 2)),  # Medium batch
        'weight_decay': 1e-4,
        'name_suffix': 'MediumBatch'
    })
    
    return configs


def train_model(model, train_loader, val_loader, config: Dict, vocab_size: int, device: str) -> Dict:
    """Train a single model configuration."""
    
    print(f"\nTraining {config['model_type']} with {config['optimizer']} "
          f"(lr={config['learning_rate']}, batch_size={config['batch_size']})")
    
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                            lr=config['learning_rate'], 
                            momentum=0.9,
                            weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    
    # Training parameters
    max_epochs = 200
    patience = 50
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
        
        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            best_state_dict = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}, "
                  f"Best {best_accuracy:.4f} (patience {patience_counter})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        model.train()
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(best_state_dict)
    
    print(f"  ✅ Training complete: Best accuracy {best_accuracy:.4f} in {training_time:.1f}s")
    
    return {
        'best_accuracy': best_accuracy,
        'final_loss': train_losses[-1],
        'epochs_trained': epoch + 1,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'model_state_dict': best_state_dict,
        'config': config
    }


def train_for_vocab_size(vocab_size: int) -> Dict:
    """Train all model configurations for a given vocabulary size."""
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS FOR p={vocab_size}")
    print(f"Dataset size: {vocab_size * vocab_size} examples")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = Path(f'data/mod_{vocab_size}_dataset.pkl')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = ModularArithmeticDataset.load(dataset_path)
    
    # Create train/val split (80/20)
    inputs = dataset.data['inputs']
    targets = dataset.data['targets']
    dataset_size = len(inputs)
    
    # Shuffle indices
    indices = torch.randperm(dataset_size)
    train_size = int(0.8 * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]
    
    print(f"Train examples: {len(train_inputs)}")
    print(f"Val examples: {len(val_inputs)}")
    
    # Get configurations
    configs = get_optimal_configs(vocab_size, dataset_size)
    results = {}
    
    # Create output directory
    output_dir = Path('models/scaled')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each configuration
    for config in configs:
        print(f"\n--- Configuration: {config['model_type']}_{config['optimizer'].upper()}_{config['name_suffix']} ---")
        
        # Create model
        model = create_scaled_model(config['model_type'], vocab_size, device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # Train model
        result = train_model(model, train_loader, val_loader, config, vocab_size, device)
        
        # Save model if accuracy is good
        if result['best_accuracy'] >= 0.90:
            model_name = f"{config['model_type']}_{config['optimizer'].upper()}_{config['name_suffix']}_p{vocab_size}"
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
            print(f"  ⚠️  Accuracy too low ({result['best_accuracy']:.3f}) - not saved")
        
        results[f"{config['model_type']}_{config['optimizer']}_{config['name_suffix']}"] = result
    
    return results


def main():
    """Main training function for all vocabulary sizes."""
    
    print("SCALED MODEL TRAINING")
    print("Training models for p=13, 17, 23 using proven approach")
    print("(Based on successful hyperparameters from p=7)")
    print("=" * 80)
    
    vocab_sizes = [13, 17, 23]  # Skip 7 since it's already trained
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
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    success_count = 0
    total_count = 0
    
    for vocab_size, results in all_results.items():
        print(f"\np={vocab_size} ({vocab_size*vocab_size} examples):")
        
        for config_name, result in results.items():
            total_count += 1
            accuracy = result['best_accuracy']
            status = "✅" if accuracy >= 0.95 else "⚠️" if accuracy >= 0.90 else "❌"
            
            if accuracy >= 0.90:
                success_count += 1
            
            saved = " [SAVED]" if 'saved_path' in result else ""
            print(f"  {status} {config_name}: {accuracy:.3f}{saved}")
    
    print(f"\n📊 Overall Success Rate: {success_count}/{total_count} "
          f"({100*success_count/total_count:.0f}%)")
    
    # Save summary
    summary = {
        'success_rate': success_count / total_count,
        'total_models': total_count,
        'successful_models': success_count,
        'results_by_vocab_size': {}
    }
    
    for vocab_size, results in all_results.items():
        summary['results_by_vocab_size'][vocab_size] = {
            config_name: {
                'accuracy': result['best_accuracy'],
                'epochs': result['epochs_trained'],
                'saved': 'saved_path' in result
            }
            for config_name, result in results.items()
        }
    
    with open('models/scaled/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Training summary saved to models/scaled/training_summary.json")
    
    if success_count > 0:
        print(f"\n🎉 SUCCESS! {success_count} models achieved 90%+ accuracy")
        print("Ready for neural topology visualization research!")
    else:
        print(f"\n⚠️  No models achieved 90%+ accuracy. Consider adjusting hyperparameters.")


if __name__ == "__main__":
    main()