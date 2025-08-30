#!/usr/bin/env python3
"""
Comprehensive hyperparameter search for modular arithmetic task.

This script systematically searches for optimal hyperparameters to achieve
high accuracy on the small dataset (49 examples).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
from pathlib import Path
from itertools import product
sys.path.insert(0, 'Dataset')

from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator
from models.optimized_models import create_optimized_model


class OptimizedTrainer:
    """Optimized trainer for small dataset with extensive hyperparameter options."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Create model
        self.model = create_optimized_model(
            config['model_type'], 
            vocab_size=config['vocab_size'], 
            device=self.device
        )
        
        # Create optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay']
            )
        
        # Learning rate scheduler
        if config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['num_epochs']
            )
        elif config.get('scheduler') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config['num_epochs']//4, gamma=0.5
            )
        else:
            self.scheduler = None
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train_model(self, train_loader, val_loader=None):
        """Train model with the given configuration."""
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'converged_epoch': -1
        }
        
        best_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_x.size(0)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        logits = self.model(batch_x)
                        loss = self.criterion(logits, batch_y)
                        
                        val_loss += loss.item() * batch_x.size(0)
                        predictions = torch.argmax(logits, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_x.size(0)
                
                val_acc = val_correct / val_total
                val_loss = val_loss / val_total
            else:
                val_acc = train_correct / train_total
                val_loss = train_loss / train_total
            
            train_acc = train_correct / train_total
            train_loss = train_loss / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Check for convergence
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                history['best_val_acc'] = best_acc
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                history['converged_epoch'] = epoch
                break
            
            # Target accuracy reached
            if val_acc >= self.config['target_accuracy']:
                history['converged_epoch'] = epoch
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history


def create_hyperparameter_grid():
    """Create comprehensive hyperparameter search space."""
    
    # Define search space
    hyperparams = {
        'model_type': ['tiny_transformer', 'linear', 'tiny_mamba', 'mlp'],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'batch_size': [1, 2, 4, 49],  # Including full batch
        'optimizer': ['sgd', 'adam', 'adamw'],
        'weight_decay': [0.0, 1e-4, 1e-3],
        'scheduler': [None, 'cosine', 'step'],
        'use_validation': [True, False]  # Whether to use train/val split
    }
    
    # Generate all combinations
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    configs = []
    for combination in product(*values):
        config = dict(zip(keys, combination))
        
        # Add fixed parameters
        config.update({
            'vocab_size': 7,
            'num_epochs': 2000,
            'patience': 200,
            'target_accuracy': 0.99,
            'device': 'cpu',
            'grad_clip': 1.0
        })
        
        # Skip some nonsensical combinations
        if config['batch_size'] == 49 and config['use_validation']:
            continue  # Can't use validation with full batch
        
        if config['optimizer'] == 'sgd':
            config['momentum'] = 0.9
        
        configs.append(config)
    
    return configs


def run_single_experiment(config, dataset, experiment_id):
    """Run a single hyperparameter configuration."""
    
    print(f"\n{'='*60}")
    print(f"Experiment {experiment_id}: {config['model_type']} | LR={config['learning_rate']} | "
          f"Batch={config['batch_size']} | Optimizer={config['optimizer']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Create data loaders
        total_size = dataset.data['num_examples']
        
        if config['use_validation']:
            # Use train/validation split
            train_size = int(total_size * 0.8)
            indices = torch.randperm(total_size)
            
            train_inputs = dataset.data['inputs'][indices[:train_size]]
            train_targets = dataset.data['targets'][indices[:train_size]]
            val_inputs = dataset.data['inputs'][indices[train_size:]]
            val_targets = dataset.data['targets'][indices[train_size:]]
            
            train_dataset = TensorDataset(train_inputs, train_targets)
            val_dataset = TensorDataset(val_inputs, val_targets)
        else:
            # Use full dataset for training
            train_dataset = TensorDataset(dataset.data['inputs'], dataset.data['targets'])
            val_dataset = TensorDataset(dataset.data['inputs'], dataset.data['targets'])  # Same for validation
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # Create trainer and train
        trainer = OptimizedTrainer(config)
        history = trainer.train_model(train_loader, val_loader)
        
        # Evaluate final model
        final_acc = history['val_acc'][-1]
        best_acc = history['best_val_acc']
        
        # Extract embeddings and validate circular structure
        trainer.model.eval()
        with torch.no_grad():
            if hasattr(trainer.model, 'embedding'):
                embeddings = trainer.model.embedding(torch.arange(config['vocab_size'])).cpu()
            else:
                # For MLP, use input representations
                test_inputs = torch.arange(config['vocab_size']).unsqueeze(1)
                test_inputs = torch.cat([test_inputs, torch.zeros_like(test_inputs)], dim=1)
                embeddings = trainer.model.network[0](test_inputs.float()).cpu()
        
        validator = CircularStructureValidator(config['vocab_size'])
        structure_results = validator.validate_embeddings(embeddings, visualize=False)
        circular_score = structure_results['overall_assessment']['overall_score']
        
        training_time = time.time() - start_time
        
        result = {
            'experiment_id': experiment_id,
            'config': config,
            'final_accuracy': final_acc,
            'best_accuracy': best_acc,
            'circular_structure_score': circular_score,
            'converged_epoch': history['converged_epoch'],
            'training_time': training_time,
            'success': best_acc >= 0.90,  # Consider 90%+ as success
            'excellent': best_acc >= 0.99,  # 99%+ is excellent
            'history': {
                'train_acc': history['train_acc'][-10:],  # Last 10 epochs
                'val_acc': history['val_acc'][-10:],
                'final_loss': history['train_loss'][-1]
            }
        }
        
        print(f"✅ Result: Acc={best_acc:.3f}, Circular={circular_score:.3f}, "
              f"Time={training_time:.1f}s, Epochs={len(history['train_acc'])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return {
            'experiment_id': experiment_id,
            'config': config,
            'error': str(e),
            'success': False,
            'excellent': False
        }


def run_hyperparameter_search(max_experiments=50):
    """Run systematic hyperparameter search."""
    
    print("HYPERPARAMETER SEARCH FOR MODULAR ARITHMETIC")
    print("=" * 60)
    
    # Create dataset
    dataset = ModularArithmeticDataset(p=7, representation='embedding')
    print(f"Dataset: {dataset.data['num_examples']} examples")
    
    # Generate configurations
    all_configs = create_hyperparameter_grid()
    print(f"Total configurations: {len(all_configs)}")
    
    # Limit experiments if needed
    if len(all_configs) > max_experiments:
        # Prioritize: small batch sizes, good optimizers, no validation split for tiny dataset
        all_configs = sorted(all_configs, key=lambda x: (
            x['use_validation'],  # Prefer no validation split
            x['batch_size'],      # Prefer smaller batch sizes
            -(['sgd', 'adamw', 'adam'].index(x['optimizer']) if x['optimizer'] in ['sgd', 'adamw', 'adam'] else 0)
        ))
        all_configs = all_configs[:max_experiments]
        print(f"Limited to: {max_experiments} experiments")
    
    # Run experiments
    results = []
    best_result = None
    best_score = 0.0
    
    for i, config in enumerate(all_configs, 1):
        result = run_single_experiment(config, dataset, i)
        results.append(result)
        
        # Track best result
        if result.get('best_accuracy', 0) > best_score:
            best_score = result.get('best_accuracy', 0)
            best_result = result
        
        # Early termination if we find excellent results
        excellent_results = [r for r in results if r.get('excellent', False)]
        if len(excellent_results) >= 3:
            print(f"\n🎉 Found {len(excellent_results)} excellent results (99%+ accuracy), stopping early!")
            break
    
    # Save results
    output_dir = Path('hyperparameter_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'search_results_{timestamp}.json'
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = result.copy()
        if 'config' in json_result:
            # Ensure all config values are JSON serializable
            json_result['config'] = {k: v for k, v in json_result['config'].items() 
                                   if isinstance(v, (str, int, float, bool, type(None)))}
        json_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📊 HYPERPARAMETER SEARCH COMPLETE")
    print(f"Results saved to: {results_file}")
    
    # Analysis
    successful_results = [r for r in results if r.get('success', False)]
    excellent_results = [r for r in results if r.get('excellent', False)]
    
    print(f"\nResults Summary:")
    print(f"Total experiments: {len(results)}")
    print(f"Successful (≥90%): {len(successful_results)}")
    print(f"Excellent (≥99%): {len(excellent_results)}")
    
    if best_result:
        print(f"\nBest Result:")
        print(f"Accuracy: {best_result.get('best_accuracy', 0):.3f}")
        print(f"Circular Score: {best_result.get('circular_structure_score', 0):.3f}")
        print(f"Model: {best_result['config']['model_type']}")
        print(f"LR: {best_result['config']['learning_rate']}")
        print(f"Batch: {best_result['config']['batch_size']}")
        print(f"Optimizer: {best_result['config']['optimizer']}")
    
    return results, best_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--max_experiments', type=int, default=30, 
                       help='Maximum number of experiments to run')
    
    args = parser.parse_args()
    
    results, best = run_hyperparameter_search(max_experiments=args.max_experiments)
    
    if best and best.get('excellent', False):
        print(f"\n🎉 SUCCESS: Found optimal hyperparameters achieving {best['best_accuracy']:.1%} accuracy!")
    elif best and best.get('success', False):
        print(f"\n✅ GOOD: Best configuration achieved {best['best_accuracy']:.1%} accuracy")
    else:
        print(f"\n⚠️  Need more tuning: Best accuracy was only {best.get('best_accuracy', 0):.1%}")