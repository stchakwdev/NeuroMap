#!/usr/bin/env python3
"""
Complete extraction script for all models.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from topology_viz.backend.topology_extractor import TopologyExtractor

def main():
    print("🚀 Extracting topology data from all trained models...")
    
    extractor = TopologyExtractor()
    
    print("Available models:")
    for name, info in extractor.model_inventory.items():
        print(f"  {name}: {info['architecture']} (p={info['p']}, {info['type']})")
    
    # Extract all models
    raw_data = extractor.extract_all_models()
    
    # Save raw extraction data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    raw_data_path = data_dir / "raw_topology_data.json"
    print(f"\n💾 Saving raw extraction data to: {raw_data_path}")
    
    with open(raw_data_path, 'w') as f:
        json.dump(raw_data, f, indent=2, default=str)
    
    print("✅ Extraction complete!")

if __name__ == "__main__":
    main()