"""
Training script for neural networks on modular arithmetic tasks.

This script provides a complete training pipeline for both transformer and Mamba models,
with comprehensive logging and model comparison capabilities.
"""

import torch
import sys
import os
from pathlib import Path
import argparse
import json
import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from Dataset.dataset import ModularArithmeticDataset
from Dataset.validation import CircularStructureValidator
from models.transformer import create_model as create_transformer
from models.mamba_model import create_mamba_model
from models.model_utils import ModelTrainer, ModelEvaluator, create_training_splits


def train_model(model_type: str,
                dataset: ModularArithmeticDataset,
                device: str = 'cpu',
                num_epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 1e-3,
                train_ratio: float = 0.8,
                target_accuracy: float = 0.99,
                save_dir: Optional[Path] = None,
                verbose: bool = True) -> Dict[str, Any]:
    """
    Train a model on modular arithmetic task.
    
    Args:
        model_type: 'transformer' or 'mamba'
        dataset: ModularArithmeticDataset instance
        device: Device to train on
        num_epochs: Maximum number of epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        train_ratio: Fraction of data for training
        target_accuracy: Target accuracy to stop training
        save_dir: Directory to save models and results
        verbose: Print training progress
        
    Returns:
        Dictionary with training results and model analysis
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} on mod {dataset.p} arithmetic")
        print(f"{'='*60}")
    
    # Create model
    if model_type == 'transformer':
        model = create_transformer(vocab_size=dataset.p, device=device)
    elif model_type == 'mamba':
        model = create_mamba_model(vocab_size=dataset.p, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
    
    # Create data splits
    splits = create_training_splits(dataset, train_ratio=train_ratio)
    
    # Create trainer
    trainer = ModelTrainer(model, device=device, learning_rate=learning_rate)
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders(
        splits['train_inputs'], splits['train_targets'],
        splits['val_inputs'], splits['val_targets'],
        batch_size=batch_size
    )
    
    if verbose:
        print(f"Training samples: {len(splits['train_inputs'])}")
        print(f"Validation samples: {len(splits['val_inputs'])}")
        print(f"Target accuracy: {target_accuracy:.1%}")
    
    # Train model
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        target_accuracy=target_accuracy,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    # Final evaluation
    evaluator = ModelEvaluator(model, device=device)
    final_results = evaluator.evaluate_on_dataset(dataset)
    
    if verbose:
        print(f"\n📊 Final Results:")
        print(f"Overall accuracy: {final_results['overall_accuracy']:.3f}")
        print(f"Training time: {training_time:.1f}s")
    
    # Validate circular structure in learned embeddings
    validator = CircularStructureValidator(dataset.p)
    number_embeddings = evaluator.get_number_representations('embeddings')
    structure_validation = validator.validate_embeddings(number_embeddings, visualize=False)
    
    if verbose:
        score = structure_validation['overall_assessment']['overall_score']
        quality = structure_validation['overall_assessment']['quality_assessment']
        print(f"Circular structure score: {score:.2f}")
        print(f"Structure quality: {quality}")
    
    # Compile results
    results = {
        'model_type': model_type,
        'dataset_p': dataset.p,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'final_accuracy': final_results['overall_accuracy'],
        'training_history': history,
        'structure_validation': structure_validation,
        'final_results': final_results,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_ratio': train_ratio,
            'target_accuracy': target_accuracy
        }
    }
    
    # Save model and results if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / f'{model_type}_mod{dataset.p}.pt'
        trainer.save_model(model_path)
        
        # Save results
        results_path = save_dir / f'{model_type}_mod{dataset.p}_results.json'
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = convert_tensors_for_json(value)
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        if verbose:
            print(f"Saved model to: {model_path}")
            print(f"Saved results to: {results_path}")
    
    return results, model, trainer


def convert_tensors_for_json(obj):
    """Recursively convert tensors to lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_for_json(item) for item in obj]
    else:
        return obj


def compare_models(dataset: ModularArithmeticDataset,
                  device: str = 'cpu',
                  save_dir: Optional[Path] = None,
                  **train_kwargs) -> Dict[str, Any]:
    """
    Train and compare transformer vs Mamba models.
    
    Args:
        dataset: ModularArithmeticDataset instance
        device: Device to train on
        save_dir: Directory to save results
        **train_kwargs: Additional training arguments
        
    Returns:
        Dictionary comparing both models
    """
    print(f"\n🔬 Comparing Transformer vs Mamba on mod {dataset.p} arithmetic")
    
    # Train both models
    transformer_results, transformer_model, transformer_trainer = train_model(
        'transformer', dataset, device=device, save_dir=save_dir, **train_kwargs
    )
    
    mamba_results, mamba_model, mamba_trainer = train_model(
        'mamba', dataset, device=device, save_dir=save_dir, **train_kwargs
    )
    
    # Compare results
    comparison = {
        'transformer': transformer_results,
        'mamba': mamba_results,
        'comparison': {
            'accuracy_difference': (
                transformer_results['final_accuracy'] - mamba_results['final_accuracy']
            ),
            'training_time_ratio': (
                transformer_results['training_time'] / mamba_results['training_time']
            ),
            'parameter_count_ratio': (
                transformer_results['num_parameters'] / mamba_results['num_parameters']
            ),
            'structure_score_difference': (
                transformer_results['structure_validation']['overall_assessment']['overall_score'] -
                mamba_results['structure_validation']['overall_assessment']['overall_score']
            )
        }
    }
    
    print(f"\n📈 Model Comparison Summary:")
    print(f"Transformer accuracy: {transformer_results['final_accuracy']:.3f}")
    print(f"Mamba accuracy: {mamba_results['final_accuracy']:.3f}")
    print(f"Accuracy difference: {comparison['comparison']['accuracy_difference']:.3f}")
    print(f"Training time ratio (T/M): {comparison['comparison']['training_time_ratio']:.2f}")
    print(f"Parameter ratio (T/M): {comparison['comparison']['parameter_count_ratio']:.2f}")
    
    # Save comparison if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        comparison_path = save_dir / f'model_comparison_mod{dataset.p}.json'
        with open(comparison_path, 'w') as f:
            json.dump(convert_tensors_for_json(comparison), f, indent=2)
        print(f"Saved comparison to: {comparison_path}")
    
    return comparison, (transformer_model, mamba_model), (transformer_trainer, mamba_trainer)


def main():
    """Main training script with command line interface."""
    parser = argparse.ArgumentParser(description='Train models on modular arithmetic')
    parser.add_argument('--model', choices=['transformer', 'mamba', 'both'], 
                       default='both', help='Model type to train')
    parser.add_argument('--p', type=int, default=17, help='Modulus for arithmetic')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', default='cpu', help='Device to train on')
    parser.add_argument('--save_dir', type=str, help='Directory to save results')
    parser.add_argument('--target_acc', type=float, default=0.99, help='Target accuracy')
    
    args = parser.parse_args()
    
    # Create dataset
    print(f"Creating mod {args.p} arithmetic dataset...")
    dataset = ModularArithmeticDataset(p=args.p, representation='embedding')
    print(f"Dataset created with {dataset.data['num_examples']} examples")
    
    # Set up save directory
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    if args.model == 'both':
        comparison, models, trainers = compare_models(
            dataset=dataset,
            device=args.device,
            save_dir=save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            target_accuracy=args.target_acc
        )
        
        print(f"\n✅ Training comparison complete!")
        
    else:
        results, model, trainer = train_model(
            model_type=args.model,
            dataset=dataset,
            device=args.device,
            save_dir=save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            target_accuracy=args.target_acc
        )
        
        print(f"\n✅ {args.model.capitalize()} training complete!")


if __name__ == "__main__":
    main()