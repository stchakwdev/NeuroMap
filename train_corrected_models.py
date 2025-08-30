#!/usr/bin/env python3
"""
Corrected training approach for larger datasets.

The first attempt failed because models were still too complex.
This version uses the exact same approach that worked for p=7.
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

# Enable deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

class SimpleLinearModel(nn.Module):
    """Very simple linear model - exactly like the one that worked for p=7."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # Keep embedding dimension small but scale slightly with vocab size
        hidden_dim = max(32, min(64, vocab_size * 3))  
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        embeddings = self.embedding(x)
        flattened = embeddings.view(embeddings.size(0), -1)
        logits = self.classifier(flattened)
        return logits


class SimpleMLP(nn.Module):
    """Simple MLP that worked for p=7."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        # Keep it simple - direct integer input
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, vocab_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x_float = x.float()
        logits = self.network(x_float)
        return logits


def get_corrected_configs(vocab_size: int, dataset_size: int) -> List[Dict]:
    """Get configurations based exactly on what worked for p=7."""
    
    configs = []
    
    # Config 1: Exact copy of Linear_SGD_FullBatch (100% accuracy for p=7)
    configs.append({
        'model_type': 'simple_linear',
        'optimizer': 'sgd',
        'learning_rate': 0.1,  # Use exact same LR that worked
        'batch_size': dataset_size,  # Full batch
        'weight_decay': 1e-5,
        'name_suffix': 'FullBatch'
    })
    
    # Config 2: Exact copy of Linear_Adam_SmallBatch (100% accuracy for p=7)
    configs.append({
        'model_type': 'simple_linear',
        'optimizer': 'adam',
        'learning_rate': 0.001,  # Use exact same LR that worked
        'batch_size': 2,  # Same small batch size
        'weight_decay': 1e-5,
        'name_suffix': 'SmallBatch'
    })
    
    # Config 3: Simple MLP approach
    configs.append({
        'model_type': 'simple_mlp',
        'optimizer': 'adamw',
        'learning_rate': 0.001,
        'batch_size': 4,  # Small batch
        'weight_decay': 1e-4,
        'name_suffix': 'SmallBatch'
    })
    
    # Config 4: Try very high learning rate with SGD (sometimes works for larger datasets)
    configs.append({
        'model_type': 'simple_linear',
        'optimizer': 'sgd',
        'learning_rate': 0.5,  # Even higher LR
        'batch_size': dataset_size,
        'weight_decay': 1e-5,
        'name_suffix': 'HighLR'
    })
    
    return configs


def create_simple_model(model_type: str, vocab_size: int, device: str = 'cpu'):
    """Create simple models like the ones that worked for p=7."""
    
    if model_type == 'simple_linear':
        model = SimpleLinearModel(vocab_size)
    elif model_type == 'simple_mlp':
        model = SimpleMLP(vocab_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


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
    
    # Training parameters - more aggressive
    max_epochs = 300  # More epochs
    patience = 100    # More patience
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
        
        # Print progress more frequently
        if (epoch + 1) % 25 == 0 or best_accuracy > 0.5:
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
    print(f"CORRECTED TRAINING FOR p={vocab_size}")
    print(f"Dataset size: {vocab_size * vocab_size} examples")
    print(f"Using simple models and proven hyperparameters")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = Path(f'data/mod_{vocab_size}_dataset.pkl')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = ModularArithmeticDataset.load(dataset_path)
    
    # Create train/val split (80/20) with same random seed for consistency
    inputs = dataset.data['inputs']
    targets = dataset.data['targets']
    dataset_size = len(inputs)
    
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
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
    configs = get_corrected_configs(vocab_size, dataset_size)
    results = {}
    
    # Create output directory
    output_dir = Path('models/corrected')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each configuration
    for config in configs:
        print(f"\n--- Configuration: {config['model_type']}_{config['optimizer'].upper()}_{config['name_suffix']} ---")
        
        # Create model
        model = create_simple_model(config['model_type'], vocab_size, device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        print(f"Param/Data ratio: {param_count / dataset_size:.2f}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # Train model
        result = train_model(model, train_loader, val_loader, config, vocab_size, device)
        
        # Save model if accuracy is reasonable
        if result['best_accuracy'] >= 0.80:  # Lower threshold initially
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
    
    print("CORRECTED SCALED MODEL TRAINING")
    print("Using simple models with exact hyperparameters that worked for p=7")
    print("=" * 80)
    
    vocab_sizes = [13, 17, 23]  
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
    print("CORRECTED TRAINING SUMMARY")
    print(f"{'='*80}")
    
    success_count = 0
    total_count = 0
    high_acc_count = 0
    
    for vocab_size, results in all_results.items():
        print(f"\np={vocab_size} ({vocab_size*vocab_size} examples):")
        
        for config_name, result in results.items():
            total_count += 1
            accuracy = result['best_accuracy']
            
            if accuracy >= 0.95:
                status = "🎉"
                high_acc_count += 1
                success_count += 1
            elif accuracy >= 0.90:
                status = "✅"
                success_count += 1
            elif accuracy >= 0.80:
                status = "⚠️"
            else:
                status = "❌"
            
            saved = " [SAVED]" if 'saved_path' in result else ""
            print(f"  {status} {config_name}: {accuracy:.3f}{saved}")
    
    print(f"\n📊 Success Rates:")
    print(f"  • High accuracy (≥95%): {high_acc_count}/{total_count} ({100*high_acc_count/total_count:.0f}%)")
    print(f"  • Good accuracy (≥90%): {success_count}/{total_count} ({100*success_count/total_count:.0f}%)")
    
    # Save summary
    summary = {
        'high_accuracy_rate': high_acc_count / total_count,
        'success_rate': success_count / total_count,
        'total_models': total_count,
        'high_accuracy_models': high_acc_count,
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
    
    with open('models/corrected/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Training summary saved to models/corrected/training_summary.json")
    
    if success_count > 0:
        print(f"\n🎉 SUCCESS! {success_count} models achieved 90%+ accuracy")
        if high_acc_count > 0:
            print(f"   BONUS: {high_acc_count} models achieved 95%+ accuracy!")
        print("Ready for neural topology visualization research!")
    else:
        print(f"\n⚠️  Training improved but still needs work. Best models:")
        best_models = []
        for vocab_size, results in all_results.items():
            for config_name, result in results.items():
                best_models.append((vocab_size, config_name, result['best_accuracy']))
        
        best_models.sort(key=lambda x: x[2], reverse=True)
        for vocab_size, config_name, acc in best_models[:3]:
            print(f"     p={vocab_size} {config_name}: {acc:.3f}")


if __name__ == "__main__":
    main()