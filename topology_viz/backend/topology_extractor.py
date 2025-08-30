"""
Neural Topology Extractor - Extract representations from trained models.

This module loads all trained models and extracts their learned embeddings
and internal representations for visualization of concept topology.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
import os

# Add parent directory to path to import model modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.optimized_models import TinyTransformer, LinearModel, TinyMamba, MLP
from models.alternative_approaches.memory_models import DirectLookupModel, HybridMemoryModel, MemoryAugmentedModel
from Dataset.validation import CircularStructureValidator


class TopologyExtractor:
    """
    Extract neural topology data from trained models.
    
    This class handles loading different model architectures and extracting
    their learned representations for visualization.
    """
    
    def __init__(self, models_base_path: str = None):
        if models_base_path is None:
            self.models_base_path = Path(__file__).parent.parent.parent / "models"
        else:
            self.models_base_path = Path(models_base_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define our model inventory
        self.model_inventory = self._build_model_inventory()
    
    def _build_model_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Build inventory of all available trained models."""
        inventory = {
            # p=7 optimized models
            "Linear_SGD_FullBatch_p7": {
                "file_path": self.models_base_path / "optimized" / "Linear_SGD_FullBatch_p7.pt",
                "architecture": "LinearModel",
                "p": 7,
                "type": "traditional",
                "accuracy": 1.0
            },
            "Linear_Adam_SmallBatch_p7": {
                "file_path": self.models_base_path / "optimized" / "Linear_Adam_SmallBatch_p7.pt",
                "architecture": "LinearModel", 
                "p": 7,
                "type": "traditional",
                "accuracy": 1.0
            },
            "TinyTransformer_AdamW_p7": {
                "file_path": self.models_base_path / "optimized" / "TinyTransformer_AdamW_p7.pt",
                "architecture": "TinyTransformer",
                "p": 7,
                "type": "traditional",
                "accuracy": 1.0
            },
            
            # p=13 memory models
            "DirectLookup_Adam_p13": {
                "file_path": self.models_base_path / "alternative" / "DirectLookup_Adam_p13.pt",
                "architecture": "DirectLookupModel",
                "p": 13,
                "type": "memory",
                "accuracy": 1.0
            },
            "HybridMemory_AdamW_p13": {
                "file_path": self.models_base_path / "alternative" / "HybridMemory_AdamW_p13.pt",
                "architecture": "HybridMemoryModel",
                "p": 13,
                "type": "memory", 
                "accuracy": 1.0
            },
            
            # p=17 memory models
            "DirectLookup_Adam_p17": {
                "file_path": self.models_base_path / "successful" / "DirectLookup_Adam_p17.pt",
                "architecture": "DirectLookupModel",
                "p": 17,
                "type": "memory",
                "accuracy": 1.0
            },
            "HybridMemory_AdamW_p17": {
                "file_path": self.models_base_path / "successful" / "HybridMemory_AdamW_p17.pt",
                "architecture": "HybridMemoryModel",
                "p": 17,
                "type": "memory",
                "accuracy": 1.0
            },
            
            # p=23 memory models  
            "DirectLookup_Adam_p23": {
                "file_path": self.models_base_path / "successful" / "DirectLookup_Adam_p23.pt",
                "architecture": "DirectLookupModel",
                "p": 23,
                "type": "memory",
                "accuracy": 1.0
            },
            "HybridMemory_AdamW_p23": {
                "file_path": self.models_base_path / "successful" / "HybridMemory_AdamW_p23.pt",
                "architecture": "HybridMemoryModel",
                "p": 23,
                "type": "memory",
                "accuracy": 0.998
            }
        }
        
        return inventory
    
    def load_model(self, model_name: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a specific model by name.
        
        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.model_inventory:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = self.model_inventory[model_name]
        file_path = model_info["file_path"]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        print(f"Loading {model_name}...")
        
        # Load the saved model data
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # Extract configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            vocab_size = model_info['p']
        else:
            vocab_size = model_info['p']
            config = {'vocab_size': vocab_size}
        
        # Create model instance
        architecture = model_info["architecture"]
        
        if architecture == "LinearModel":
            model = LinearModel(vocab_size=vocab_size)
        elif architecture == "TinyTransformer":
            model = TinyTransformer(vocab_size=vocab_size)
        elif architecture == "TinyMamba":
            model = TinyMamba(vocab_size=vocab_size)
        elif architecture == "DirectLookupModel":
            model = DirectLookupModel(vocab_size=vocab_size)
        elif architecture == "HybridMemoryModel":
            model = HybridMemoryModel(vocab_size=vocab_size)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # Add config to model_info for metadata
        model_info = model_info.copy()
        model_info.update(config)
        
        print(f"  ✅ Loaded {architecture} for p={vocab_size}")
        return model, model_info
    
    def extract_embeddings(self, model: nn.Module, model_info: Dict[str, Any]) -> np.ndarray:
        """
        Extract learned embeddings from a model.
        
        For different architectures, we extract different representations:
        - Traditional models: embedding layer weights
        - Memory models: learned lookup table or memory representations
        """
        architecture = model_info["architecture"]
        p = model_info["p"]
        
        print(f"  Extracting embeddings from {architecture}...")
        
        if architecture in ["LinearModel", "TinyTransformer", "TinyMamba"]:
            # Extract embedding weights - these represent how the model 
            # internally represents each number 0 through p-1
            if hasattr(model, 'embedding'):
                embeddings = model.embedding.weight.detach().cpu().numpy()
            else:
                print(f"    Warning: No embedding layer found, using random embeddings")
                embeddings = np.random.randn(p, 32)
                
        elif architecture == "DirectLookupModel":
            # For direct lookup, extract the learned lookup table
            # We need to get the representation for each number 0 through p-1
            # by looking at what the model learned for inputs involving each number
            
            embeddings = []
            with torch.no_grad():
                for i in range(p):
                    # Get the lookup table entries where this number appears
                    # as the first operand in different additions
                    lookup_indices = [i * p + j for j in range(p)]  # (i, 0), (i, 1), ..., (i, p-1)
                    lookup_values = model.lookup_table.weight[lookup_indices].cpu().numpy()
                    
                    # Average the representations to get a single embedding for number i
                    avg_embedding = np.mean(lookup_values, axis=0)
                    embeddings.append(avg_embedding)
            
            embeddings = np.array(embeddings)
            
        elif architecture == "HybridMemoryModel":
            # For hybrid model, extract from the memory component
            memory_model = model.memory
            embeddings = []
            
            with torch.no_grad():
                for i in range(p):
                    lookup_indices = [i * p + j for j in range(p)]
                    lookup_values = memory_model.lookup_table.weight[lookup_indices].cpu().numpy()
                    avg_embedding = np.mean(lookup_values, axis=0)
                    embeddings.append(avg_embedding)
            
            embeddings = np.array(embeddings)
            
        else:
            raise ValueError(f"Unknown architecture for embedding extraction: {architecture}")
        
        print(f"    Extracted embeddings shape: {embeddings.shape}")
        return embeddings
    
    def validate_circular_structure(self, embeddings: np.ndarray, p: int) -> Dict[str, Any]:
        """Validate circular structure in extracted embeddings."""
        validator = CircularStructureValidator(p)
        
        # Convert to tensor for validation
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        # Run validation
        results = validator.validate_embeddings(embeddings_tensor, visualize=False)
        
        return results
    
    def extract_activation_patterns(self, model: nn.Module, model_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract activation patterns by running all possible inputs through the model.
        This shows how the model processes different arithmetic combinations.
        """
        p = model_info["p"]
        architecture = model_info["architecture"]
        
        print(f"  Extracting activation patterns for all {p}² inputs...")
        
        # Create all possible (a,b) pairs
        all_inputs = []
        for a in range(p):
            for b in range(p):
                all_inputs.append([a, b])
        
        inputs_tensor = torch.tensor(all_inputs, dtype=torch.long).to(self.device)
        
        patterns = {}
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(inputs_tensor)
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            
            patterns["predictions"] = predictions
            patterns["inputs"] = np.array(all_inputs)
            
            # Extract intermediate activations if possible
            if hasattr(model, 'embedding') and architecture != "DirectLookupModel":
                embeddings = model.embedding(inputs_tensor)
                if len(embeddings.shape) == 3:  # (batch, seq, dim)
                    # For sequence models, average over sequence
                    embeddings = embeddings.mean(dim=1)
                patterns["intermediate_embeddings"] = embeddings.cpu().numpy()
        
        return patterns
    
    def compute_topology_metrics(self, embeddings: np.ndarray, p: int) -> Dict[str, float]:
        """Compute various topology quality metrics."""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        metrics = {}
        
        # 1. Circular structure validation
        validation_results = self.validate_circular_structure(embeddings, p)
        overall_score = validation_results.get('overall_assessment', {}).get('overall_score', 0)
        metrics["circular_structure_score"] = float(overall_score)
        
        # 2. Clustering quality (how well separated are the number representations?)
        if len(embeddings) > 1:
            labels = list(range(len(embeddings)))
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                try:
                    sil_score = silhouette_score(embeddings, labels)
                    metrics["silhouette_score"] = float(sil_score)
                except:
                    metrics["silhouette_score"] = 0.0
            else:
                metrics["silhouette_score"] = 0.0
        
        # 3. Embedding spread (how well distributed are the embeddings?)
        if len(embeddings) > 1:
            center = np.mean(embeddings, axis=0)
            distances_from_center = np.linalg.norm(embeddings - center, axis=1)
            metrics["embedding_spread"] = float(np.std(distances_from_center))
            metrics["mean_distance_from_center"] = float(np.mean(distances_from_center))
        
        # 4. Nearest neighbor consistency (are adjacent numbers close?)
        adjacency_distances = []
        for i in range(p):
            next_i = (i + 1) % p
            dist = np.linalg.norm(embeddings[i] - embeddings[next_i])
            adjacency_distances.append(dist)
        
        metrics["mean_adjacency_distance"] = float(np.mean(adjacency_distances))
        metrics["adjacency_distance_std"] = float(np.std(adjacency_distances))
        
        return metrics
    
    def extract_model_data(self, model_name: str) -> Dict[str, Any]:
        """Extract complete topology data for a single model."""
        print(f"\n{'='*50}")
        print(f"Extracting topology data for: {model_name}")
        print(f"{'='*50}")
        
        # Load model
        model, model_info = self.load_model(model_name)
        p = model_info["p"]
        
        # Extract embeddings
        embeddings = self.extract_embeddings(model, model_info)
        
        # Validate structure
        validation_results = self.validate_circular_structure(embeddings, p)
        
        # Extract activation patterns
        activation_patterns = self.extract_activation_patterns(model, model_info)
        
        # Compute metrics
        topology_metrics = self.compute_topology_metrics(embeddings, p)
        
        # Compile all data
        model_data = {
            "model_name": model_name,
            "model_info": model_info,
            "embeddings": embeddings.tolist(),  # Convert to JSON-serializable format
            "embeddings_shape": list(embeddings.shape),
            "validation_results": validation_results,
            "topology_metrics": topology_metrics,
            "activation_patterns": {
                "predictions": activation_patterns["predictions"].tolist(),
                "inputs": activation_patterns["inputs"].tolist(),
            }
        }
        
        # Add intermediate embeddings if available
        if "intermediate_embeddings" in activation_patterns:
            model_data["activation_patterns"]["intermediate_embeddings"] = \
                activation_patterns["intermediate_embeddings"].tolist()
        
        print(f"  ✅ Extraction complete!")
        print(f"  📊 Circular structure score: {topology_metrics.get('circular_structure_score', 0):.3f}")
        print(f"  🎯 Silhouette score: {topology_metrics.get('silhouette_score', 0):.3f}")
        
        return model_data
    
    def extract_all_models(self) -> Dict[str, Any]:
        """Extract topology data for all available models."""
        print("🚀 Starting topology extraction for all models...")
        
        all_data = {
            "metadata": {
                "extraction_info": "Neural topology data extracted from trained models",
                "device": str(self.device),
                "total_models": len(self.model_inventory)
            },
            "models": {}
        }
        
        successful_extractions = 0
        
        for model_name in self.model_inventory.keys():
            try:
                model_data = self.extract_model_data(model_name)
                all_data["models"][model_name] = model_data
                successful_extractions += 1
                
            except Exception as e:
                print(f"  ❌ Failed to extract {model_name}: {str(e)}")
                all_data["models"][model_name] = {
                    "error": str(e),
                    "model_name": model_name
                }
        
        all_data["metadata"]["successful_extractions"] = successful_extractions
        all_data["metadata"]["failed_extractions"] = len(self.model_inventory) - successful_extractions
        
        print(f"\n🎉 Extraction complete!")
        print(f"✅ Successfully extracted: {successful_extractions}/{len(self.model_inventory)} models")
        
        return all_data


def tensor_to_serializable(obj):
    """Convert torch tensors to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(item) for item in obj]
    else:
        return obj


def main():
    """Main function for testing the extractor."""
    extractor = TopologyExtractor()
    
    print("Available models:")
    for name, info in extractor.model_inventory.items():
        print(f"  {name}: {info['architecture']} (p={info['p']}, {info['type']})")
    
    # Test extraction for one model first
    test_model = "Linear_SGD_FullBatch_p7"
    print(f"\nTesting extraction for: {test_model}")
    
    try:
        model_data = extractor.extract_model_data(test_model)
        print("✅ Test extraction successful!")
        
        # Save test data
        test_output_path = Path(__file__).parent / "data" / "test_topology_data.json"
        test_output_path.parent.mkdir(exist_ok=True)
        
        with open(test_output_path, 'w') as f:
            json.dump({"test_model": model_data}, f, indent=2, default=str)
        
        print(f"📁 Test data saved to: {test_output_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Topology Extractor ready for full extraction!")
    print("Run generate_viz_data.py to extract all models.")


if __name__ == "__main__":
    main()