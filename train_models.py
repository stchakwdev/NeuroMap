#!/usr/bin/env python3
"""
Proper training script to train and save models with circular structure validation.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import json
import time
sys.path.insert(0, 'Dataset')

from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator
from models.transformer import create_model as create_transformer
from models.mamba_model import create_mamba_model
from models.model_utils import ModelTrainer, ModelEvaluator

def train_and_save_model(model_name: str, p: int = 7):
    """Train a model to high accuracy and save it with validation results."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} model (p={p})")
    print(f"{'='*60}")
    
    # 1. Create dataset
    print("1. Creating dataset...")
    dataset = ModularArithmeticDataset(p=p, representation='embedding')
    print(f"   Dataset: {dataset.data['num_examples']} examples")
    
    # 2. Create model
    print("2. Creating model...")
    if model_name == 'transformer':
        model = create_transformer(vocab_size=p, device='cpu')
    elif model_name == 'mamba':
        model = create_mamba_model(vocab_size=p, device='cpu')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"   Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. Set up training
    trainer = ModelTrainer(model, device='cpu', learning_rate=3e-4)  # Lower learning rate
    
    # Create train/val split
    total_size = dataset.data['num_examples']
    train_size = int(total_size * 0.8)
    indices = torch.randperm(total_size)
    
    train_inputs = dataset.data['inputs'][indices[:train_size]]
    train_targets = dataset.data['targets'][indices[:train_size]]
    val_inputs = dataset.data['inputs'][indices[train_size:]]
    val_targets = dataset.data['targets'][indices[train_size:]]
    
    train_loader, val_loader = trainer.create_data_loaders(
        train_inputs, train_targets, val_inputs, val_targets, batch_size=8  # Smaller batch
    )
    
    # 4. Train model
    print("3. Training model...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,  # More epochs
        target_accuracy=0.80,  # Lower target initially
        patience=50,  # More patience
        verbose=True
    )
    
    training_time = time.time() - start_time
    final_acc = history['train_acc'][-1] if history['train_acc'] else 0.0
    
    print(f"   Training completed in {training_time:.1f} seconds")
    print(f"   Final accuracy: {final_acc:.3f}")
    print(f"   Epochs trained: {len(history['train_loss'])}")
    
    # 5. Extract representations
    print("4. Extracting representations...")
    model.eval()
    
    # Get number embeddings
    with torch.no_grad():
        number_embeddings = model.embedding(torch.arange(p)).detach()
    
    print(f"   Representations extracted: {number_embeddings.shape}")
    
    # 6. Validate circular structure
    print("5. Validating circular structure...")
    validator = CircularStructureValidator(p)
    results = validator.validate_embeddings(number_embeddings, visualize=False)
    
    score = results['overall_assessment']['overall_score']
    quality = results['overall_assessment']['quality_assessment']
    
    print(f"   Circular structure score: {score:.3f}")
    print(f"   Quality assessment: {quality}")
    
    # Show detailed metrics
    if 'circular_ordering' in results:
        is_circular = results['circular_ordering']['is_circular_order']
        print(f"   Circular ordering: {is_circular}")
    
    if 'distance_consistency' in results:
        dist_corr = results['distance_consistency']['distance_correlation']
        print(f"   Distance correlation: {dist_corr:.3f}")
    
    if 'adjacency_structure' in results:
        passes_adj = results['adjacency_structure']['passes_adjacency_test']
        adj_ratio = results['adjacency_structure']['adjacency_ratio']
        print(f"   Adjacency test: {passes_adj} (ratio: {adj_ratio:.3f})")
    
    # 7. Save everything
    print("6. Saving results...")
    
    # Save model
    model_path = f"models/{model_name}_p{p}.pt"
    Path("models").mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': model_name,
            'vocab_size': p,
            'd_model': model.embedding.embedding_dim if hasattr(model, 'embedding') else 64
        },
        'training_history': history,
        'final_accuracy': final_acc,
        'training_time': training_time
    }, model_path)
    print(f"   Model saved: {model_path}")
    
    # Save validation results
    results_path = f"results/{model_name}_p{p}_validation.json"
    Path("results").mkdir(exist_ok=True)
    
    # Convert tensors to lists for JSON
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(v) for v in obj]
        else:
            return obj
    
    json_results = tensor_to_list(results)
    json_results['training_info'] = {
        'model_type': model_name,
        'p': p,
        'final_accuracy': final_acc,
        'epochs_trained': len(history['train_loss']),
        'training_time': training_time,
        'parameters': sum(p.numel() for p in model.parameters())
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   Results saved: {results_path}")
    
    # Save embeddings
    embeddings_path = f"results/{model_name}_p{p}_embeddings.pt"
    torch.save(number_embeddings, embeddings_path)
    print(f"   Embeddings saved: {embeddings_path}")
    
    return {
        'model_name': model_name,
        'p': p,
        'accuracy': final_acc,
        'circular_score': score,
        'quality': quality,
        'training_time': training_time,
        'epochs': len(history['train_loss']),
        'model_path': model_path,
        'results_path': results_path
    }

def create_comparison_summary(results_list):
    """Create a summary comparison of all trained models."""
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    summary = {
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': []
    }
    
    print(f"{'Model':<15} {'Accuracy':<10} {'Circular':<10} {'Quality':<20} {'Time':<8}")
    print("-" * 70)
    
    for result in results_list:
        print(f"{result['model_name']:<15} {result['accuracy']:<10.3f} {result['circular_score']:<10.3f} {result['quality']:<20} {result['training_time']:<8.1f}s")
        summary['models'].append(result)
    
    # Save summary
    with open('results/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: results/training_summary.json")
    
    # Determine best model
    best_model = max(results_list, key=lambda x: x['circular_score'])
    print(f"\n🏆 Best circular structure: {best_model['model_name']} (score: {best_model['circular_score']:.3f})")
    
    high_acc_models = [r for r in results_list if r['accuracy'] >= 0.9]
    if high_acc_models:
        print(f"✅ Models with high accuracy (≥90%): {len(high_acc_models)}/{len(results_list)}")
    else:
        print(f"⚠️  No models reached 90% accuracy - may need hyperparameter tuning")

def main():
    print("NEURAL TOPOLOGY RESEARCH: MODEL TRAINING")
    print("Training models for circular structure validation")
    
    results = []
    
    # Train both models
    for model_name in ['transformer', 'mamba']:
        try:
            result = train_and_save_model(model_name, p=7)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        create_comparison_summary(results)
        print(f"\n🎉 Training completed! {len(results)} models saved.")
    else:
        print(f"\n💥 No models were successfully trained.")

if __name__ == "__main__":
    main()