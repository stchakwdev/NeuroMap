"""
Training utilities and helper functions.

This module provides utility functions for setting up training,
creating data loaders, and managing training configurations.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_data_loaders(dataset, 
                       batch_size: int = 32,
                       train_ratio: float = 0.8,
                       shuffle_train: bool = True,
                       random_seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders from dataset.
    
    Args:
        dataset: ModularArithmeticDataset instance
        batch_size: Batch size for data loaders
        train_ratio: Fraction of data to use for training
        shuffle_train: Whether to shuffle training data
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Create tensor dataset
    inputs = dataset.data['inputs']
    targets = dataset.data['targets']
    full_dataset = TensorDataset(inputs, targets)
    
    # Split into train and validation
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader


def setup_optimizer(model: torch.nn.Module,
                   optimizer_type: str = 'adamw',
                   learning_rate: float = 1e-3,
                   weight_decay: float = 1e-4,
                   **kwargs) -> torch.optim.Optimizer:
    """
    Set up optimizer for model training.
    
    Args:
        model: Neural network model
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def setup_scheduler(optimizer: torch.optim.Optimizer,
                   scheduler_type: str = 'cosine',
                   num_epochs: int = 100,
                   **kwargs):
    """
    Set up learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'step', 'exponential', 'none')
        num_epochs: Total number of training epochs
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type.lower() == 'none':
        return None
    elif scheduler_type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            **kwargs
        )
    elif scheduler_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
            **{k: v for k, v in kwargs.items() if k not in ['step_size', 'gamma']}
        )
    elif scheduler_type.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
            **{k: v for k, v in kwargs.items() if k != 'gamma'}
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Path] = None,
                         show_plot: bool = True):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
        show_plot: Whether to show the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], label='Train Acc', marker='o', markersize=3)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], label='Val Acc', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_learning_curves(histories: Dict[str, Dict[str, List[float]]],
                          save_path: Optional[Path] = None,
                          show_plot: bool = True):
    """
    Analyze and compare learning curves from multiple models.
    
    Args:
        histories: Dictionary mapping model names to training histories
        save_path: Path to save plot (optional)
        show_plot: Whether to show the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = sns.color_palette("husl", len(histories))
    
    for i, (model_name, history) in enumerate(histories.items()):
        color = colors[i]
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training loss
        axes[0, 0].plot(epochs, history['train_loss'], 
                       label=f'{model_name} (train)', color=color, linestyle='-')
        
        # Validation loss
        if 'val_loss' in history:
            axes[0, 1].plot(epochs, history['val_loss'], 
                           label=f'{model_name} (val)', color=color, linestyle='-')
        
        # Training accuracy
        axes[1, 0].plot(epochs, history['train_acc'], 
                       label=f'{model_name} (train)', color=color, linestyle='-')
        
        # Validation accuracy  
        if 'val_acc' in history:
            axes[1, 1].plot(epochs, history['val_acc'], 
                           label=f'{model_name} (val)', color=color, linestyle='-')
    
    # Configure subplots
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def compute_model_statistics(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Compute statistics about model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze parameter distribution by layer
    layer_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_stats[name] = {
                'shape': list(param.shape),
                'num_params': param.numel(),
                'param_percentage': param.numel() / trainable_params * 100
            }
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'layer_statistics': layer_stats,
        'memory_mb': total_params * 4 / 1024 / 1024  # Assuming float32
    }


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_model_convergence(history: Dict[str, List[float]], 
                          window: int = 10,
                          threshold: float = 1e-4) -> Dict[str, Any]:
    """
    Check if model training has converged.
    
    Args:
        history: Training history
        window: Window size for checking convergence
        threshold: Threshold for convergence detection
        
    Returns:
        Dictionary with convergence analysis
    """
    convergence_info = {}
    
    if len(history['train_loss']) < window:
        return {'converged': False, 'reason': 'Insufficient training epochs'}
    
    # Check loss convergence
    recent_losses = history['train_loss'][-window:]
    loss_std = np.std(recent_losses)
    loss_converged = loss_std < threshold
    
    # Check accuracy convergence  
    accuracy_converged = False
    if 'train_acc' in history:
        recent_accs = history['train_acc'][-window:]
        acc_std = np.std(recent_accs)
        accuracy_converged = acc_std < threshold
    
    # Overall convergence
    overall_converged = loss_converged and accuracy_converged
    
    return {
        'converged': overall_converged,
        'loss_converged': loss_converged,
        'accuracy_converged': accuracy_converged,
        'loss_std': loss_std,
        'accuracy_std': acc_std if 'train_acc' in history else None,
        'window_size': window,
        'threshold': threshold
    }


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from Dataset.dataset import ModularArithmeticDataset
    
    # Test utilities
    dataset = ModularArithmeticDataset(p=5)
    
    # Test data loader creation
    train_loader, val_loader = create_data_loaders(dataset, batch_size=16)
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test batch
    for batch_x, batch_y in train_loader:
        print(f"Batch shape: {batch_x.shape}, {batch_y.shape}")
        break
    
    print("✅ Training utilities implementation complete!")