"""
Generate Visualization Data - Main script for topology visualization pipeline.

This script orchestrates the complete pipeline from model loading to final
visualization data generation for the interactive web interface.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import argparse

from topology_extractor import TopologyExtractor
from topology_processor import TopologyProcessor, convert_numpy_types


def generate_complete_visualization_data(output_dir: str = None, 
                                       models_to_process: list = None) -> None:
    """
    Generate complete visualization data for all models.
    
    Args:
        output_dir: Directory to save output files
        models_to_process: Specific models to process (None = all models)
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Neural Topology Visualization Data Generation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Step 1: Extract topology data from all models
    print("\n📊 STEP 1: Extracting topology data from trained models")
    print("-" * 40)
    
    extractor = TopologyExtractor()
    
    if models_to_process:
        print(f"Processing specific models: {models_to_process}")
        # Extract only specified models
        raw_data = {
            "metadata": {
                "extraction_info": "Neural topology data extracted from selected models",
                "device": str(extractor.device),
                "total_models": len(models_to_process)
            },
            "models": {}
        }
        
        for model_name in models_to_process:
            if model_name in extractor.model_inventory:
                try:
                    model_data = extractor.extract_model_data(model_name)
                    raw_data["models"][model_name] = model_data
                except Exception as e:
                    print(f"  ❌ Failed to extract {model_name}: {str(e)}")
                    raw_data["models"][model_name] = {
                        "error": str(e),
                        "model_name": model_name
                    }
            else:
                print(f"  ⚠️ Unknown model: {model_name}")
    else:
        # Extract all models
        raw_data = extractor.extract_all_models()
    
    # Save raw extraction data
    raw_data_path = output_dir / "raw_topology_data.json"
    print(f"\n💾 Saving raw extraction data to: {raw_data_path}")
    
    with open(raw_data_path, 'w') as f:
        json.dump(convert_numpy_types(raw_data), f, indent=2, default=str)
    
    extraction_time = time.time() - start_time
    print(f"⏱️  Extraction completed in {extraction_time:.2f} seconds")
    
    # Step 2: Process data for visualization
    print(f"\n🔄 STEP 2: Processing data for interactive visualization")
    print("-" * 50)
    
    processor = TopologyProcessor()
    processed_data = processor.process_all_models(raw_data)
    
    # Save processed visualization data
    viz_data_path = output_dir / "topology_visualization_data.json"
    print(f"\n💾 Saving visualization data to: {viz_data_path}")
    
    with open(viz_data_path, 'w') as f:
        json.dump(convert_numpy_types(processed_data), f, indent=2, default=str)
    
    # Step 3: Generate summary and individual model files
    print(f"\n📋 STEP 3: Generating additional output files")
    print("-" * 40)
    
    # Create summary report
    summary = create_summary_report(processed_data)
    summary_path = output_dir / "topology_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"📄 Summary report saved to: {summary_path}")
    
    # Create individual model files for web interface
    web_data_dir = output_dir.parent.parent / "web_viz" / "data"
    web_data_dir.mkdir(exist_ok=True)
    
    # Main visualization data for web interface
    web_viz_path = web_data_dir / "models.json"
    with open(web_viz_path, 'w') as f:
        json.dump(convert_numpy_types(processed_data), f, indent=2, default=str)
    print(f"🌐 Web visualization data saved to: {web_viz_path}")
    
    # Create model index for web interface
    model_index = create_model_index(processed_data)
    index_path = web_data_dir / "model_index.json"
    with open(index_path, 'w') as f:
        json.dump(model_index, f, indent=2, default=str)
    print(f"📇 Model index saved to: {index_path}")
    
    # Step 4: Final report
    total_time = time.time() - start_time
    print(f"\n🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Models processed: {len([m for m in processed_data['models'].values() if 'error' not in m])}")
    print(f"📁 Files generated:")
    print(f"   - Raw data: {raw_data_path}")
    print(f"   - Visualization data: {viz_data_path}")
    print(f"   - Summary: {summary_path}")
    print(f"   - Web data: {web_viz_path}")
    print(f"   - Model index: {index_path}")
    
    print(f"\n🌐 Ready for interactive visualization!")
    print(f"Open topology_viz/web_viz/index.html to explore the models.")


def create_summary_report(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary report of the topology analysis."""
    
    summary = {
        "generation_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(processed_data["models"])
        },
        "model_summary": {},
        "comparative_insights": {},
        "visualization_ready": []
    }
    
    successful_models = []
    accuracy_by_p = {}
    architecture_performance = {}
    
    for model_name, model_data in processed_data["models"].items():
        if "error" in model_data:
            continue
        
        model_info = model_data["model_info"]
        topology_metrics = model_data["topology_metrics"]
        
        # Individual model summary
        model_summary = {
            "architecture": model_info["architecture"],
            "p": model_info["p"],
            "type": model_info["type"],
            "accuracy": model_info["accuracy"],
            "circular_structure_score": topology_metrics.get("circular_structure_score", 0),
            "silhouette_score": topology_metrics.get("silhouette_score", 0),
            "visualizations_available": len(model_data.get("visualizations", {}))
        }
        
        summary["model_summary"][model_name] = model_summary
        successful_models.append(model_name)
        
        if model_summary["visualizations_available"] > 0:
            summary["visualization_ready"].append(model_name)
        
        # Group by p value
        p = model_info["p"]
        if p not in accuracy_by_p:
            accuracy_by_p[p] = []
        accuracy_by_p[p].append(model_info["accuracy"])
        
        # Group by architecture
        arch = model_info["architecture"]
        if arch not in architecture_performance:
            architecture_performance[arch] = []
        architecture_performance[arch].append(topology_metrics.get("circular_structure_score", 0))
    
    # Comparative insights
    summary["comparative_insights"] = {
        "successful_models": len(successful_models),
        "visualization_ready_models": len(summary["visualization_ready"]),
        "average_accuracy_by_p": {p: sum(accs)/len(accs) for p, accs in accuracy_by_p.items()},
        "average_circular_score_by_architecture": {
            arch: sum(scores)/len(scores) for arch, scores in architecture_performance.items()
        },
        "best_circular_structure": None,
        "highest_accuracy_models": []
    }
    
    # Find best models
    if summary["model_summary"]:
        best_circular = max(summary["model_summary"].items(), 
                          key=lambda x: x[1].get("circular_structure_score", 0))
        summary["comparative_insights"]["best_circular_structure"] = {
            "model": best_circular[0],
            "score": best_circular[1]["circular_structure_score"]
        }
        
        highest_acc_models = [name for name, data in summary["model_summary"].items() 
                             if data["accuracy"] >= 1.0]
        summary["comparative_insights"]["highest_accuracy_models"] = highest_acc_models
    
    return summary


def create_model_index(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a model index for the web interface."""
    
    index = {
        "models": [],
        "architectures": set(),
        "moduli": set(),
        "types": set()
    }
    
    for model_name, model_data in processed_data["models"].items():
        if "error" in model_data:
            continue
        
        model_info = model_data["model_info"]
        topology_metrics = model_data["topology_metrics"]
        
        model_entry = {
            "id": model_name,
            "name": model_name,
            "architecture": model_info["architecture"],
            "p": model_info["p"],
            "type": model_info["type"],
            "accuracy": model_info["accuracy"],
            "circular_structure_score": topology_metrics.get("circular_structure_score", 0),
            "has_visualizations": len(model_data.get("visualizations", {})) > 0,
            "available_visualizations": list(model_data.get("visualizations", {}).keys())
        }
        
        index["models"].append(model_entry)
        index["architectures"].add(model_info["architecture"])
        index["moduli"].add(model_info["p"])
        index["types"].add(model_info["type"])
    
    # Convert sets to lists for JSON serialization
    index["architectures"] = sorted(list(index["architectures"]))
    index["moduli"] = sorted(list(index["moduli"]))
    index["types"] = sorted(list(index["types"]))
    
    # Sort models by p value then by name
    index["models"].sort(key=lambda x: (x["p"], x["name"]))
    
    return index


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate neural topology visualization data")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for generated files")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                       help="Specific models to process (default: all models)")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        extractor = TopologyExtractor()
        for name, info in extractor.model_inventory.items():
            print(f"  {name}: {info['architecture']} (p={info['p']}, {info['type']}, acc={info['accuracy']})")
        return
    
    # Generate visualization data
    generate_complete_visualization_data(
        output_dir=args.output_dir,
        models_to_process=args.models
    )


if __name__ == "__main__":
    main()