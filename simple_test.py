#!/usr/bin/env python3
"""
Simple test to validate core components work.
"""

import sys
import os
sys.path.insert(0, 'Dataset')

def test_basic_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from dataset import ModularArithmeticDataset
        print("✅ dataset import")
    except Exception as e:
        print(f"❌ dataset import: {e}")
        return False
    
    try:
        from validation import CircularStructureValidator
        print("✅ validation import")
    except Exception as e:
        print(f"❌ validation import: {e}")
        return False
    
    try:
        from models.transformer import create_model
        print("✅ transformer import")
    except Exception as e:
        print(f"❌ transformer import: {e}")
        return False
    
    try:
        from models.model_utils import ModelTrainer
        print("✅ model_utils import")
    except Exception as e:
        print(f"❌ model_utils import: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from dataset import ModularArithmeticDataset
        dataset = ModularArithmeticDataset(p=5, representation='embedding')
        print(f"✅ Dataset created: {dataset.data['num_examples']} examples")
    except Exception as e:
        print(f"❌ Dataset creation: {e}")
        return False
    
    try:
        from validation import CircularStructureValidator
        validator = CircularStructureValidator(p=5)
        print("✅ Validator created")
    except Exception as e:
        print(f"❌ Validator creation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE COMPONENT TEST")
    print("=" * 50)
    
    success = test_basic_imports()
    if success:
        success = test_basic_functionality()
    
    if success:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)