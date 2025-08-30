#!/usr/bin/env python3
"""
Quick hyperparameter search with fewer epochs to identify promising configurations.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
sys.path.insert(0, 'Dataset')

from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator
from models.optimized_models import create_optimized_model


def quick_train(model, train_loader, config, max_epochs=200):
    """Quick training with early stopping."""
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                             momentum=0.9, weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    patience_counter = 0
    patience = 50
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_x.size(0)
        
        accuracy = correct / total
        
        if accuracy > best_acc:
            best_acc = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience or accuracy >= 0.99:
            break
    
    return best_acc, epoch + 1


def quick_search():
    """Run quick hyperparameter search."""
    
    print("QUICK HYPERPARAMETER SEARCH")
    print("=" * 40)
    
    # Create dataset
    dataset = ModularArithmeticDataset(p=7, representation='embedding')
    
    # Promising configurations based on analysis
    configs = [
        # Small batch, high LR, no validation split
        {'model_type': 'mlp', 'learning_rate': 0.01, 'batch_size': 1, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        {'model_type': 'mlp', 'learning_rate': 0.1, 'batch_size': 1, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        {'model_type': 'linear', 'learning_rate': 0.01, 'batch_size': 1, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        {'model_type': 'linear', 'learning_rate': 0.1, 'batch_size': 1, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        
        # Full batch gradient descent
        {'model_type': 'mlp', 'learning_rate': 0.1, 'batch_size': 49, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        {'model_type': 'linear', 'learning_rate': 0.1, 'batch_size': 49, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        {'model_type': 'tiny_transformer', 'learning_rate': 0.01, 'batch_size': 49, 'optimizer': 'sgd', 'weight_decay': 0.0, 'use_validation': False},
        
        # Adam variants
        {'model_type': 'mlp', 'learning_rate': 0.001, 'batch_size': 2, 'optimizer': 'adam', 'weight_decay': 1e-4, 'use_validation': False},
        {'model_type': 'linear', 'learning_rate': 0.001, 'batch_size': 2, 'optimizer': 'adam', 'weight_decay': 1e-4, 'use_validation': False},
        
        # Smaller models with different optimizers
        {'model_type': 'tiny_transformer', 'learning_rate': 0.001, 'batch_size': 2, 'optimizer': 'adamw', 'weight_decay': 1e-3, 'use_validation': False},
        {'model_type': 'tiny_mamba', 'learning_rate': 0.001, 'batch_size': 2, 'optimizer': 'adamw', 'weight_decay': 1e-3, 'use_validation': False},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Experiment {i}/{len(configs)} ---")
        print(f"Model: {config['model_type']}, LR: {config['learning_rate']}, "
              f"Batch: {config['batch_size']}, Optimizer: {config['optimizer']}")
        
        start_time = time.time()
        
        try:
            # Create model
            model = create_optimized_model(config['model_type'], vocab_size=7, device='cpu')
            
            # Create data loader
            if config['use_validation']:
                # Use train/val split
                total_size = dataset.data['num_examples']
                train_size = int(total_size * 0.8)
                indices = torch.randperm(total_size)
                train_inputs = dataset.data['inputs'][indices[:train_size]]
                train_targets = dataset.data['targets'][indices[:train_size]]
            else:
                # Use full dataset
                train_inputs = dataset.data['inputs']
                train_targets = dataset.data['targets']
            
            train_dataset = TensorDataset(train_inputs, train_targets)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            
            # Train
            best_acc, epochs = quick_train(model, train_loader, config)
            
            # Test circular structure
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'embedding'):
                    embeddings = model.embedding(torch.arange(7))
                else:
                    # For MLP, use input representations  
                    test_inputs = torch.arange(7).unsqueeze(1).float()
                    test_inputs = torch.cat([test_inputs, torch.zeros_like(test_inputs)], dim=1)
                    embeddings = model.network[0](test_inputs)
            
            validator = CircularStructureValidator(7)
            structure_results = validator.validate_embeddings(embeddings, visualize=False)
            circular_score = structure_results['overall_assessment']['overall_score']
            
            training_time = time.time() - start_time
            
            result = {
                'config': config,
                'accuracy': best_acc,
                'circular_score': circular_score,
                'epochs': epochs,
                'time': training_time,
                'success': best_acc >= 0.90
            }
            
            results.append(result)
            
            status = "🎉 EXCELLENT" if best_acc >= 0.99 else "✅ GOOD" if best_acc >= 0.90 else "⚠️  POOR"
            print(f"{status}: Acc={best_acc:.3f}, Circular={circular_score:.3f}, "
                  f"Epochs={epochs}, Time={training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            result = {'config': config, 'error': str(e), 'success': False}
            results.append(result)
    
    # Analysis
    print(f"\n{'='*60}")
    print("QUICK SEARCH RESULTS")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('success', False)]
    excellent = [r for r in results if r.get('accuracy', 0) >= 0.99]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful (≥90%): {len(successful)}")
    print(f"Excellent (≥99%): {len(excellent)}")
    
    if successful:
        print(f"\n🏆 TOP RESULTS:")
        successful.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        for i, result in enumerate(successful[:5], 1):
            config = result['config']
            print(f"{i}. {config['model_type']} | LR={config['learning_rate']} | "
                  f"Batch={config['batch_size']} | {config['optimizer']} -> "
                  f"Acc={result['accuracy']:.3f}")
    
    if excellent:
        print(f"\n🎊 EXCELLENT CONFIGURATIONS (99%+ accuracy):")
        for result in excellent:
            config = result['config']
            print(f"   {config['model_type']} | LR={config['learning_rate']} | "
                  f"Batch={config['batch_size']} | {config['optimizer']} | "
                  f"Acc={result['accuracy']:.3f} | Circular={result['circular_score']:.3f}")
    
    return results, excellent[0] if excellent else (successful[0] if successful else None)


if __name__ == "__main__":
    results, best = quick_search()
    
    if best and best.get('accuracy', 0) >= 0.99:
        print(f"\n🎉 SUCCESS: Found configuration achieving {best['accuracy']:.1%} accuracy!")
        print("Ready for full training with optimal hyperparameters.")
    elif best:
        print(f"\n✅ GOOD: Best configuration achieved {best['accuracy']:.1%} accuracy") 
        print("May need further tuning for 99% accuracy.")
    else:
        print(f"\n💥 No successful configurations found. Need to expand search space.")