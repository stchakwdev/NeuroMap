#!/usr/bin/env python3
"""
Create datasets for all p values: 7, 13, 17, 23
"""

import sys
sys.path.insert(0, 'Dataset')

from dataset import create_mod_p_datasets

if __name__ == "__main__":
    print("Creating datasets for p=13, 17, 23...")
    print("(p=7 already exists)")
    
    # Create datasets for larger p values
    p_values = [13, 17, 23]
    datasets = create_mod_p_datasets(p_values)
    
    print("\n✅ All datasets created successfully!")
    
    # Print summary
    for p in p_values:
        dataset = datasets[p]
        print(f"p={p}: {dataset.data['num_examples']} examples ({p}×{p})")