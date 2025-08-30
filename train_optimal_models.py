#!/usr/bin/env python3
"""
Train models with optimal hyperparameters found from search.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path
sys.path.insert(0, 'Dataset')

from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator
from models.optimized_models import create_optimized_model


def train_optimal_model(config, save_path, max_epochs=1000):
    """Train a model with optimal hyperparameters."""
    
    print(f"\n{'='*60}")
    print(f"Training Optimized Model: {config['name']}")
    print(f"{'='*60}")
    print(f"Model: {config['model_type']}")
    print(f"Learning Rate: {config['learning_rate']}")  
    print(f"Batch Size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer']}")
    
    start_time = time.time()
    
    # Create dataset
    dataset = ModularArithmeticDataset(p=7, representation='embedding')
    
    # Create model
    model = create_optimized_model(config['model_type'], vocab_size=7, device='cpu')
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Create optimizer
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                             momentum=0.9, weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    
    # Create data loader
    train_dataset = TensorDataset(dataset.data['inputs'], dataset.data['targets'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    patience = 100
    
    print("\nTraining Progress:")
    print("Epoch    Loss      Accuracy   Time")
    print("-" * 35)
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_x.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch % 20 == 0 or accuracy >= 0.99:
            print(f"{epoch:5d}   {avg_loss:7.4f}   {accuracy:8.3f}   {epoch_time:.2f}s")
        
        # Check for improvement
        if accuracy > best_acc:
            best_acc = accuracy
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'accuracy': accuracy,
                'epoch': epoch,
                'loss': avg_loss
            }, save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Perfect accuracy reached
        if accuracy >= 0.999:
            print(f"Perfect accuracy reached at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    print(f"\nTraining Complete:")
    print(f"Best Accuracy: {best_acc:.3f}")
    print(f"Total Time: {training_time:.1f}s")
    print(f"Epochs: {epoch + 1}")
    
    # Load best model for validation
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Validate circular structure
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'embedding'):
            embeddings = model.embedding(torch.arange(7))
        else:
            # For MLP and Linear models, use input representations
            test_inputs = torch.arange(7).unsqueeze(1).float()
            test_inputs = torch.cat([test_inputs, torch.zeros_like(test_inputs)], dim=1)
            
            if hasattr(model, 'network'):  # MLP
                embeddings = model.network[0](test_inputs)
            else:  # Linear model
                embeddings = model.embedding(torch.arange(7))
    
    validator = CircularStructureValidator(7)
    structure_results = validator.validate_embeddings(embeddings, visualize=False)
    circular_score = structure_results['overall_assessment']['overall_score']
    quality = structure_results['overall_assessment']['quality_assessment']
    
    print(f"\nCircular Structure Analysis:")
    print(f"Score: {circular_score:.3f}")
    print(f"Quality: {quality}")
    
    # Detailed metrics
    is_circular = structure_results['circular_ordering']['is_circular_order']
    dist_corr = structure_results['distance_consistency']['distance_correlation']
    passes_adj = structure_results['adjacency_structure']['passes_adjacency_test']
    
    print(f"Circular Ordering: {is_circular}")
    print(f"Distance Correlation: {dist_corr:.3f}")
    print(f"Adjacency Test: {passes_adj}")
    
    print(f"✅ Model saved: {save_path}")
    
    return {
        'accuracy': best_acc,
        'circular_score': circular_score,
        'training_time': training_time,
        'epochs': epoch + 1,
        'config': config
    }


def main():
    """Train all optimal configurations."""
    
    print("TRAINING OPTIMAL MODELS")
    print("Based on hyperparameter search results")
    
    # Optimal configurations found from quick search
    optimal_configs = [
        {
            'name': 'Linear_SGD_FullBatch',
            'model_type': 'linear',
            'learning_rate': 0.1,
            'batch_size': 49,
            'optimizer': 'sgd',
            'weight_decay': 0.0
        },
        {
            'name': 'Linear_Adam_SmallBatch', 
            'model_type': 'linear',
            'learning_rate': 0.001,
            'batch_size': 2,
            'optimizer': 'adam',
            'weight_decay': 1e-4
        },
        {
            'name': 'TinyTransformer_AdamW',
            'model_type': 'tiny_transformer',
            'learning_rate': 0.001,
            'batch_size': 2,
            'optimizer': 'adamw',
            'weight_decay': 1e-4
        }
    ]
    
    # Create output directory
    output_dir = Path('models/optimized')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for config in optimal_configs:
        save_path = output_dir / f"{config['name']}_p7.pt"
        result = train_optimal_model(config, save_path)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("OPTIMAL MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Model':<25} {'Accuracy':<10} {'Circular':<10} {'Time':<8} {'Quality'}")
    print("-" * 70)
    
    for result in results:
        config = result['config']
        name = config['name'][:24]
        acc = result['accuracy']
        circ = result['circular_score']
        time_val = result['training_time']
        
        if circ >= 0.8:
            quality = "Excellent"
        elif circ >= 0.5:
            quality = "Good"  
        elif circ >= 0.3:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"{name:<25} {acc:<10.3f} {circ:<10.3f} {time_val:<8.1f} {quality}")
    
    # Find best model
    best_acc_model = max(results, key=lambda x: x['accuracy'])
    best_circular_model = max(results, key=lambda x: x['circular_score'])
    
    print(f"\n🏆 Best Accuracy: {best_acc_model['config']['name']} ({best_acc_model['accuracy']:.3f})")
    print(f"🔵 Best Circular: {best_circular_model['config']['name']} ({best_circular_model['circular_score']:.3f})")
    
    if max(r['accuracy'] for r in results) >= 0.99:
        print(f"\n🎉 SUCCESS: Achieved 99%+ accuracy with optimal hyperparameters!")
        if max(r['circular_score'] for r in results) >= 0.5:
            print(f"🔵 BONUS: Also learned circular structure!")
        else:
            print(f"📝 NOTE: Models learned the task but not with circular representations")
            print(f"   This is still valuable - they solved the mathematical problem!")
    
    return results


if __name__ == "__main__":
    results = main()