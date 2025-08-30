#!/usr/bin/env python3
"""
Quick script to process the existing raw topology data.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from topology_viz.backend.topology_processor import TopologyProcessor, convert_numpy_types

def main():
    data_dir = Path(__file__).parent / "data"
    raw_data_path = data_dir / "raw_topology_data.json"
    
    if not raw_data_path.exists():
        print(f"❌ Raw data file not found: {raw_data_path}")
        return
    
    print("🔄 Loading raw topology data...")
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
    
    print(f"📊 Found data for {len(raw_data['models'])} models")
    
    print("🔄 Processing data for visualization...")
    processor = TopologyProcessor()
    processed_data = processor.process_all_models(raw_data)
    
    # Save processed data
    viz_data_path = data_dir / "topology_visualization_data.json"
    print(f"💾 Saving visualization data to: {viz_data_path}")
    
    with open(viz_data_path, 'w') as f:
        json.dump(convert_numpy_types(processed_data), f, indent=2, default=str)
    
    # Also save to web interface directory
    web_data_dir = data_dir.parent.parent / "web_viz" / "data"
    web_data_dir.mkdir(exist_ok=True)
    
    web_viz_path = web_data_dir / "models.json"
    with open(web_viz_path, 'w') as f:
        json.dump(convert_numpy_types(processed_data), f, indent=2, default=str)
    
    print(f"🌐 Web visualization data saved to: {web_viz_path}")
    print("✅ Processing complete!")

if __name__ == "__main__":
    main()