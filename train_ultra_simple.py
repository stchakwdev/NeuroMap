#!/usr/bin/env python3
"""
Ultra-simple approach: Focus on p=13 with the most basic possible models.

This is our last attempt to find working hyperparameters for larger p values.
If this doesn't work, we'll document the fundamental scaling challenge.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import time
from typing import Dict, List

sys.path.insert(0, 'Dataset')
from dataset import ModularArithmeticDataset

torch.manual_seed(42)
np.random.seed(42)

class UltraSimpleModel(nn.Module):
    """The simplest possible model that could work."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        # Just direct linear mapping with minimal parameters
        self.vocab_size = vocab_size
        self.network = nn.Sequential(
            nn.Linear(2, vocab_size * 4),  # Input layer  
            nn.ReLU(),
            nn.Linear(vocab_size * 4, vocab_size)  # Output layer
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x_float = x.float()
        return self.network(x_float)


def train_ultra_simple(vocab_size: int = 13):
    """Train with the most basic setup possible."""
    
    print(f"ULTRA-SIMPLE TRAINING FOR p={vocab_size}")
    print("="*50)
    
    # Load dataset
    dataset = ModularArithmeticDataset.load(f'data/mod_{vocab_size}_dataset.pkl')
    
    inputs = dataset.data['inputs']
    targets = dataset.data['targets']
    
    # Use ALL data for training (no validation split)
    # This maximizes training data
    train_inputs = inputs
    train_targets = targets
    
    print(f"Training on ALL {len(inputs)} examples")
    
    # Try multiple ultra-aggressive configurations
    configs = [
        # Very high learning rate with full batch
        {'lr': 1.0, 'batch_size': len(inputs), 'optimizer': 'sgd', 'momentum': 0.9},
        {'lr': 2.0, 'batch_size': len(inputs), 'optimizer': 'sgd', 'momentum': 0.9},
        {'lr': 5.0, 'batch_size': len(inputs), 'optimizer': 'sgd', 'momentum': 0.9},
        
        # Very high learning rate with small batch
        {'lr': 0.1, 'batch_size': 1, 'optimizer': 'adam'},
        {'lr': 0.01, 'batch_size': 1, 'optimizer': 'adam'},
        {'lr': 0.1, 'batch_size': 2, 'optimizer': 'adam'},
        
        # Medium learning rates
        {'lr': 0.05, 'batch_size': 16, 'optimizer': 'adamw'},
        {'lr': 0.02, 'batch_size': 32, 'optimizer': 'adamw'},
    ]
    
    best_accuracy = 0
    best_config = None
    best_model_state = None
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        # Create fresh model
        model = UltraSimpleModel(vocab_size)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count}")
        
        # Create optimizer
        if config['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), 
                                lr=config['lr'], 
                                momentum=config.get('momentum', 0))
        elif config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        else:  # adamw
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        
        # Create data loader
        dataset_tensor = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset_tensor, batch_size=config['batch_size'], shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        
        # Train for many epochs
        max_epochs = 500
        model.train()
        
        for epoch in range(max_epochs):
            total_loss = 0
            
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Check accuracy every 50 epochs
            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    outputs = model(train_inputs)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == train_targets).float().mean().item()
                
                avg_loss = total_loss / len(loader)
                print(f"  Epoch {epoch+1:3d}: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config
                    best_model_state = model.state_dict().copy()
                
                model.train()
                
                # Early success stopping
                if accuracy > 0.95:
                    print(f"  🎉 Found success! Accuracy {accuracy:.4f}")
                    break
    
    print(f"\n" + "="*50)
    print(f"BEST RESULT FOR p={vocab_size}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Config: {best_config}")
    
    if best_accuracy > 0.80:
        # Save the best model
        output_dir = Path('models/ultra_simple')
        output_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': best_model_state,
            'config': best_config,
            'accuracy': best_accuracy,
            'vocab_size': vocab_size
        }, output_dir / f'ultra_simple_p{vocab_size}.pt')
        
        print(f"✅ Saved model with {best_accuracy:.3f} accuracy!")
        return True
    else:
        print(f"❌ Best accuracy {best_accuracy:.3f} still too low")
        return False


def test_fundamental_approach():
    """Test if the fundamental approach can work at all by trying p=5."""
    
    print("TESTING FUNDAMENTAL APPROACH ON p=5")
    print("="*40)
    
    # Create a tiny dataset for p=5 (25 examples)
    from dataset import create_mod_p_datasets
    datasets = create_mod_p_datasets([5])
    
    # Use the exact same approach as above
    dataset = datasets[5]
    inputs = dataset.data['inputs']
    targets = dataset.data['targets']
    
    print(f"Dataset size: {len(inputs)} examples")
    
    model = UltraSimpleModel(5)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Full batch training
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=len(inputs), shuffle=True)
    
    for epoch in range(200):
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).float().mean().item()
                print(f"  Epoch {epoch+1}: Accuracy {accuracy:.4f}")
                
                if accuracy > 0.95:
                    print(f"  🎉 p=5 SUCCESS! Accuracy {accuracy:.4f}")
                    return True
    
    print("  ❌ Even p=5 failed - fundamental issue exists")
    return False


def main():
    """Main function to test ultra-simple approach."""
    
    print("ULTRA-SIMPLE MODEL TRAINING")
    print("Last attempt with most basic possible models")
    print("="*60)
    
    # First test if approach works at all with p=5
    if not test_fundamental_approach():
        print("\n💡 INSIGHT: The approach itself has issues.")
        print("   Even p=5 (25 examples) doesn't work well.")
        print("   This suggests we need a completely different approach.")
        
        print("\n🔬 ANALYSIS:")
        print("   • Modular arithmetic is inherently difficult")
        print("   • Our models might need structural changes")
        print("   • Consider lookup table or memory-based approaches")
        return
    
    # If p=5 works, try p=13
    print("\n" + "="*60)
    success = train_ultra_simple(13)
    
    if success:
        print("\n🎉 SUCCESS! Found working approach for p=13")
        print("   Can now apply to p=17 and p=23")
    else:
        print("\n📊 FINAL ANALYSIS:")
        print("   • p=7 works perfectly (100% accuracy)")
        print("   • p=13+ much harder than expected")  
        print("   • Scaling challenge is fundamental")
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Focus research on p=7 for now")
        print("   2. Investigate why modular arithmetic scales poorly")
        print("   3. Consider different model architectures")
        print("   4. Proceed with neural topology analysis on p=7")


if __name__ == "__main__":
    main()