"""
Model utilities for training, evaluation, and analysis.

This module provides utilities for training neural networks on modular arithmetic
tasks and extracting representations for concept analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm


class ModelTrainer:
    """Trainer for neural network models on modular arithmetic tasks."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_times': []
        }
    
    def create_data_loaders(self, 
                          train_inputs: torch.Tensor,
                          train_targets: torch.Tensor,
                          val_inputs: Optional[torch.Tensor] = None,
                          val_targets: Optional[torch.Tensor] = None,
                          batch_size: int = 32) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create data loaders for training and validation."""
        
        # Create training data loader
        train_dataset = TensorDataset(train_inputs, train_targets)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=False
        )
        
        # Create validation data loader if validation data provided
        val_loader = None
        if val_inputs is not None and val_targets is not None:
            val_dataset = TensorDataset(val_inputs, val_targets)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * batch_x.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_x.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation/test data."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_x.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              early_stopping_patience: int = 10,
              target_accuracy: float = 0.99,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait for improvement
            target_accuracy: Stop training when this accuracy is reached
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print(f"Starting training for up to {num_epochs} epochs...")
            print(f"Target accuracy: {target_accuracy:.1%}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
            else:
                val_loss, val_acc = train_loss, train_acc
            
            epoch_time = time.time() - epoch_start_time
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            if verbose and (epoch % 10 == 0 or epoch < 10):
                print(f"Epoch {epoch:3d}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                      f"time={epoch_time:.2f}s")
            
            # Check target accuracy
            if val_acc >= target_accuracy:
                if verbose:
                    print(f"\n✅ Reached target accuracy {target_accuracy:.1%} at epoch {epoch}")
                break
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n⏹️ Early stopping at epoch {epoch} (patience: {early_stopping_patience})")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if verbose:
            final_acc = self.history['val_acc'][-1]
            print(f"Training completed. Final accuracy: {final_acc:.3f}")
        
        return self.history
    
    def save_model(self, filepath: Path, include_history: bool = True):
        """Save model and training history."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': getattr(self.model, 'vocab_size', None),
                'd_model': getattr(self.model, 'd_model', None),
                'nhead': getattr(self.model, 'nhead', None),
                'num_layers': getattr(self.model, 'num_layers', None)
            }
        }
        
        if include_history:
            save_dict['training_history'] = self.history
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: Path):
        """Load model and training history."""
        save_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        if 'training_history' in save_dict:
            self.history = save_dict['training_history']
        
        return save_dict.get('model_config', {})


class ModelEvaluator:
    """Evaluator for analyzing trained models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
    
    def evaluate_on_dataset(self, dataset) -> Dict[str, float]:
        """Evaluate model on a complete dataset."""
        inputs = dataset.data['inputs'].to(self.device)
        targets = dataset.data['targets'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs)
            predictions = torch.argmax(logits, dim=1)
            
            accuracy = (predictions == targets).float().mean().item()
            
            # Per-class accuracy
            per_class_acc = {}
            for class_id in range(dataset.p):
                class_mask = targets == class_id
                if class_mask.sum() > 0:
                    class_acc = (predictions[class_mask] == targets[class_mask]).float().mean().item()
                    per_class_acc[f'class_{class_id}'] = class_acc
        
        return {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'num_correct': (predictions == targets).sum().item(),
            'num_total': len(targets)
        }
    
    def extract_activations(self, 
                          inputs: torch.Tensor,
                          layer_names: List[str] = None) -> Dict[str, torch.Tensor]:
        """Extract activations from specified layers."""
        if layer_names is None:
            layer_names = ['embeddings', 'aggregated']
        
        # Register hooks
        for layer_name in layer_names:
            self.model.register_activation_hook(layer_name)
        
        # Forward pass
        with torch.no_grad():
            inputs = inputs.to(self.device)
            _ = self.model(inputs)
            activations = self.model.get_activations()
        
        # Clean up hooks
        for layer_name in layer_names:
            self.model.remove_activation_hook(layer_name)
        
        return activations
    
    def get_number_representations(self, layer_name: str = 'embeddings') -> torch.Tensor:
        """Get representations for all numbers 0 through p-1."""
        vocab_size = self.model.vocab_size
        
        if layer_name == 'embeddings':
            return self.model.get_number_embeddings()
        else:
            # Create inputs for all numbers paired with 0
            inputs = torch.stack([
                torch.arange(vocab_size),
                torch.zeros(vocab_size, dtype=torch.long)
            ], dim=1)
            
            activations = self.extract_activations(inputs, [layer_name])
            return activations[layer_name]
    
    def analyze_predictions(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Analyze model predictions in detail."""
        with torch.no_grad():
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            correct_mask = predictions == targets
            
            return {
                'predictions': predictions.cpu(),
                'probabilities': probabilities.cpu(),
                'correct_mask': correct_mask.cpu(),
                'confidence': probabilities.max(dim=1)[0].cpu(),
                'accuracy': correct_mask.float().mean().item()
            }


def create_training_splits(dataset, train_ratio: float = 0.8) -> Dict[str, torch.Tensor]:
    """Create training and validation splits from dataset."""
    total_size = dataset.data['num_examples']
    train_size = int(total_size * train_ratio)
    
    # Random permutation for splitting
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


# Example usage
if __name__ == "__main__":
    from transformer import create_model
    
    # Create model
    model = create_model(vocab_size=17)
    trainer = ModelTrainer(model)
    
    # Create dummy data for testing
    batch_size = 32
    num_samples = 100
    inputs = torch.randint(0, 17, (num_samples, 2))
    targets = (inputs[:, 0] + inputs[:, 1]) % 17
    
    # Create data loaders
    train_loader, _ = trainer.create_data_loaders(inputs, targets, batch_size=batch_size)
    
    # Test one epoch of training
    train_loss, train_acc = trainer.train_epoch(train_loader)
    print(f"Test epoch - Loss: {train_loss:.4f}, Accuracy: {train_acc:.3f}")
    
    # Test evaluation
    evaluator = ModelEvaluator(model)
    analysis = evaluator.analyze_predictions(inputs, targets)
    print(f"Test evaluation - Accuracy: {analysis['accuracy']:.3f}")
    
    print("✅ Model utilities implementation complete!")