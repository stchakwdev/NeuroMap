"""
Topology Processor - Process extracted representations for visualization.

This module takes raw extracted model data and processes it for interactive visualization,
including dimensionality reduction, layout computation, and graph construction.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


class TopologyProcessor:
    """
    Process topology data for interactive visualization.
    
    Handles dimensionality reduction, graph construction, layout computation,
    and data formatting for web visualization.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Visualization parameters
        self.layout_configs = {
            "force_directed": {"k": 1, "iterations": 50},
            "circular": {"scale": 2.0},
            "spectral": {},
            "spring": {"k": 1, "iterations": 50}
        }
    
    def reduce_dimensionality(self, embeddings: np.ndarray, method: str = "pca", 
                            target_dim: int = 2) -> np.ndarray:
        """
        Reduce embedding dimensionality for visualization.
        
        Args:
            embeddings: High-dimensional embeddings (n_points, n_dims)
            method: Reduction method ('pca', 'tsne', 'umap')
            target_dim: Target dimensionality (2 or 3)
        
        Returns:
            Reduced embeddings (n_points, target_dim)
        """
        if embeddings.shape[1] <= target_dim:
            return embeddings
        
        print(f"    Reducing {embeddings.shape} to {target_dim}D using {method}")
        
        # Standardize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        if method == "pca":
            reducer = PCA(n_components=target_dim, random_state=self.random_state)
        elif method == "tsne":
            # t-SNE parameters optimized for small datasets
            perplexity = min(30, max(5, len(embeddings) // 4))
            reducer = TSNE(n_components=target_dim, perplexity=perplexity,
                          random_state=self.random_state, max_iter=1000)
        elif method == "umap":
            if not HAS_UMAP:
                raise ValueError("UMAP not available. Install with: pip install umap-learn")
            # UMAP parameters for small datasets
            n_neighbors = min(15, max(2, len(embeddings) - 1))
            reducer = umap.UMAP(n_components=target_dim, n_neighbors=n_neighbors,
                              random_state=self.random_state)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced = reducer.fit_transform(embeddings_scaled)
        
        # Normalize to reasonable range for visualization
        reduced = (reduced - reduced.mean(axis=0)) / reduced.std(axis=0)
        reduced = reduced * 2.0  # Scale for better visualization
        
        return reduced
    
    def construct_graph(self, embeddings: np.ndarray, p: int, 
                       connection_method: str = "modular") -> nx.Graph:
        """
        Construct graph from embeddings showing relationships between numbers.
        
        Args:
            embeddings: 2D/3D embeddings for visualization
            p: Modulus (number of nodes will be p)
            connection_method: How to connect nodes ('modular', 'similarity', 'hybrid')
        
        Returns:
            NetworkX graph with positions and edge weights
        """
        G = nx.Graph()
        
        # Add nodes (representing numbers 0 through p-1)
        for i in range(p):
            G.add_node(i, 
                      label=str(i),
                      position=embeddings[i].tolist(),
                      embedding=embeddings[i])
        
        if connection_method == "modular":
            # Connect based on modular arithmetic relationships
            self._add_modular_edges(G, p)
        elif connection_method == "similarity":
            # Connect based on embedding similarity
            self._add_similarity_edges(G, embeddings, p)
        elif connection_method == "hybrid":
            # Combine both approaches
            self._add_modular_edges(G, p)
            self._add_similarity_edges(G, embeddings, p, weight_factor=0.5)
        else:
            raise ValueError(f"Unknown connection method: {connection_method}")
        
        return G
    
    def _add_modular_edges(self, G: nx.Graph, p: int) -> None:
        """Add edges based on modular arithmetic structure."""
        # Adjacent pairs in modular arithmetic (circular)
        for i in range(p):
            next_i = (i + 1) % p
            G.add_edge(i, next_i, 
                      edge_type="adjacent",
                      weight=1.0,
                      color="blue")
        
        # Addition relationships: if i + j = k (mod p), connect i and k, j and k
        for i in range(p):
            for j in range(i + 1, p):  # Avoid duplicates
                k = (i + j) % p
                if not G.has_edge(i, k):
                    G.add_edge(i, k, 
                              edge_type="addition",
                              weight=0.5,
                              color="green")
                if not G.has_edge(j, k):
                    G.add_edge(j, k,
                              edge_type="addition", 
                              weight=0.5,
                              color="green")
    
    def _add_similarity_edges(self, G: nx.Graph, embeddings: np.ndarray, 
                            p: int, threshold: float = None, weight_factor: float = 1.0) -> None:
        """Add edges based on embedding similarity."""
        # Compute pairwise distances
        distances = cdist(embeddings, embeddings, metric='euclidean')
        
        if threshold is None:
            # Use adaptive threshold based on distance distribution
            threshold = np.percentile(distances[distances > 0], 25)
        
        # Add edges for close pairs
        for i in range(p):
            for j in range(i + 1, p):
                distance = distances[i, j]
                if distance < threshold:
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    weight = similarity * weight_factor
                    
                    if G.has_edge(i, j):
                        # Update existing edge weight
                        G[i][j]['weight'] += weight
                        G[i][j]['edge_type'] = "hybrid"
                        G[i][j]['color'] = "purple"
                    else:
                        G.add_edge(i, j,
                                  edge_type="similarity",
                                  weight=weight,
                                  color="red")
    
    def compute_layouts(self, G: nx.Graph, embeddings_2d: np.ndarray, 
                       embeddings_3d: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Compute multiple layout algorithms for the graph.
        
        Returns layouts in both 2D and 3D when possible.
        """
        layouts = {}
        
        # 1. Learned layout (from actual embeddings)
        pos_learned_2d = {i: embeddings_2d[i] for i in G.nodes()}
        layouts["learned_2d"] = {
            "positions": pos_learned_2d,
            "name": "Learned Representation",
            "description": "Positions from model's learned embeddings"
        }
        
        if embeddings_3d is not None:
            pos_learned_3d = {i: embeddings_3d[i] for i in G.nodes()}
            layouts["learned_3d"] = {
                "positions": pos_learned_3d,
                "name": "Learned Representation 3D",
                "description": "3D positions from model's learned embeddings"
            }
        
        # 2. Circular layout (expected structure)
        angles = np.linspace(0, 2*np.pi, len(G.nodes()), endpoint=False)
        pos_circular = {
            i: [self.layout_configs["circular"]["scale"] * np.cos(angles[i]),
                self.layout_configs["circular"]["scale"] * np.sin(angles[i])]
            for i in G.nodes()
        }
        layouts["circular_2d"] = {
            "positions": pos_circular,
            "name": "Expected Circular",
            "description": "Expected circular arrangement for modular arithmetic"
        }
        
        # 3. Force-directed layout
        try:
            pos_spring = nx.spring_layout(G, **self.layout_configs["spring"])
            layouts["force_2d"] = {
                "positions": pos_spring,
                "name": "Force-Directed",
                "description": "Graph-based force-directed layout"
            }
        except:
            print("    Warning: Could not compute force-directed layout")
        
        # 4. Spectral layout
        try:
            pos_spectral = nx.spectral_layout(G)
            layouts["spectral_2d"] = {
                "positions": pos_spectral,
                "name": "Spectral", 
                "description": "Spectral embedding layout"
            }
        except:
            print("    Warning: Could not compute spectral layout")
        
        return layouts
    
    def compute_graph_metrics(self, G: nx.Graph, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute graph-level topology metrics."""
        metrics = {}
        
        # Basic graph properties
        metrics["num_nodes"] = G.number_of_nodes()
        metrics["num_edges"] = G.number_of_edges()
        metrics["density"] = nx.density(G)
        
        # Connectivity
        if G.number_of_edges() > 0:
            metrics["is_connected"] = nx.is_connected(G)
            metrics["num_components"] = nx.number_connected_components(G)
            
            # Average path length
            if nx.is_connected(G):
                metrics["avg_path_length"] = nx.average_shortest_path_length(G)
                metrics["diameter"] = nx.diameter(G)
            else:
                # Compute for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
                metrics["diameter"] = nx.diameter(subgraph)
            
            # Clustering
            metrics["avg_clustering"] = nx.average_clustering(G)
            
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            metrics["max_degree_centrality"] = max(degree_centrality.values())
            metrics["avg_degree_centrality"] = np.mean(list(degree_centrality.values()))
        
        return metrics
    
    def create_visualization_data(self, graph_data: Dict[str, Any], 
                                 layouts: Dict[str, Dict]) -> Dict[str, Any]:
        """Create final visualization data structure for web interface."""
        
        viz_data = {
            "nodes": [],
            "edges": [],
            "layouts": {},
            "metadata": graph_data.copy()
        }
        
        G = nx.node_link_graph(graph_data)
        
        # Process nodes
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            viz_node = {
                "id": int(node_id),
                "label": node_data.get("label", str(node_id)),
                "group": "number",
                "value": int(node_id)  # For coloring/sizing
            }
            viz_data["nodes"].append(viz_node)
        
        # Process edges
        for source, target, edge_data in G.edges(data=True):
            viz_edge = {
                "source": int(source),
                "target": int(target),
                "weight": float(edge_data.get("weight", 1.0)),
                "type": edge_data.get("edge_type", "default"),
                "color": edge_data.get("color", "gray")
            }
            viz_data["edges"].append(viz_edge)
        
        # Process layouts
        for layout_name, layout_data in layouts.items():
            positions = layout_data["positions"]
            layout_viz = {
                "name": layout_data["name"],
                "description": layout_data["description"],
                "positions": {}
            }
            
            # Convert positions to proper format
            for node_id, pos in positions.items():
                if isinstance(pos, np.ndarray):
                    pos = pos.tolist()
                layout_viz["positions"][str(node_id)] = pos
            
            viz_data["layouts"][layout_name] = layout_viz
        
        return viz_data
    
    def process_model_data(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single model's topology data for visualization.
        
        Args:
            model_data: Raw model data from TopologyExtractor
            
        Returns:
            Processed visualization data
        """
        model_name = model_data["model_name"]
        p = model_data["model_info"]["p"]
        embeddings = np.array(model_data["embeddings"])
        
        print(f"  Processing {model_name} (p={p})...")
        
        processed_data = {
            "model_name": model_name,
            "model_info": model_data["model_info"],
            "validation_results": model_data["validation_results"],
            "topology_metrics": model_data["topology_metrics"],
            "visualizations": {}
        }
        
        # Generate multiple visualizations with different reduction methods
        reduction_methods = ["pca", "tsne"]
        if embeddings.shape[0] >= 4 and HAS_UMAP:  # UMAP needs at least 4 points and must be available
            reduction_methods.append("umap")
        
        for method in reduction_methods:
            print(f"    Processing with {method.upper()}...")
            
            try:
                # Reduce to 2D and 3D
                embeddings_2d = self.reduce_dimensionality(embeddings, method, 2)
                embeddings_3d = None
                if embeddings.shape[0] >= 4:  # Need enough points for 3D
                    embeddings_3d = self.reduce_dimensionality(embeddings, method, 3)
                
                # Construct graphs with different connection methods
                connection_methods = ["modular", "similarity", "hybrid"]
                
                for conn_method in connection_methods:
                    viz_key = f"{method}_{conn_method}"
                    
                    # Build graph
                    G = self.construct_graph(embeddings_2d, p, conn_method)
                    
                    # Compute layouts
                    layouts = self.compute_layouts(G, embeddings_2d, embeddings_3d)
                    
                    # Compute graph metrics
                    graph_metrics = self.compute_graph_metrics(G, embeddings_2d)
                    
                    # Convert graph to JSON-serializable format
                    graph_data = nx.node_link_data(G)
                    
                    # Create visualization data
                    viz_data = self.create_visualization_data(graph_data, layouts)
                    viz_data["graph_metrics"] = graph_metrics
                    viz_data["reduction_method"] = method
                    viz_data["connection_method"] = conn_method
                    
                    processed_data["visualizations"][viz_key] = viz_data
                    
                    print(f"      ✅ {viz_key}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    
            except Exception as e:
                print(f"      ❌ Failed {method}: {str(e)}")
                continue
        
        return processed_data
    
    def process_all_models(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all models' topology data for visualization."""
        
        print("🔄 Processing topology data for visualization...")
        
        processed_data = {
            "metadata": raw_data["metadata"].copy(),
            "models": {},
            "comparative_analysis": {}
        }
        
        processed_data["metadata"]["processing_info"] = {
            "dimensionality_reduction_methods": ["pca", "tsne", "umap"],
            "connection_methods": ["modular", "similarity", "hybrid"],
            "layout_algorithms": ["learned", "circular", "force_directed", "spectral"]
        }
        
        successful_processing = 0
        
        for model_name, model_data in raw_data["models"].items():
            if "error" in model_data:
                print(f"  ⏭️  Skipping {model_name} (extraction failed)")
                processed_data["models"][model_name] = model_data
                continue
            
            try:
                processed_model = self.process_model_data(model_data)
                processed_data["models"][model_name] = processed_model
                successful_processing += 1
                
            except Exception as e:
                print(f"  ❌ Failed to process {model_name}: {str(e)}")
                processed_data["models"][model_name] = {
                    "error": f"Processing failed: {str(e)}",
                    "model_name": model_name
                }
        
        # Add comparative analysis
        processed_data["comparative_analysis"] = self._create_comparative_analysis(processed_data["models"])
        
        processed_data["metadata"]["successful_processing"] = successful_processing
        
        print(f"🎉 Processing complete!")
        print(f"✅ Successfully processed: {successful_processing} models")
        
        return processed_data
    
    def _create_comparative_analysis(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparative analysis across all models."""
        
        analysis = {
            "by_architecture": {},
            "by_modulus": {},
            "by_type": {},
            "summary_statistics": {}
        }
        
        # Group models by various criteria
        for model_name, model_data in models.items():
            if "error" in model_data:
                continue
            
            model_info = model_data["model_info"]
            architecture = model_info["architecture"]
            p = model_info["p"]
            model_type = model_info["type"]
            
            # Group by architecture
            if architecture not in analysis["by_architecture"]:
                analysis["by_architecture"][architecture] = []
            analysis["by_architecture"][architecture].append(model_name)
            
            # Group by modulus
            if p not in analysis["by_modulus"]:
                analysis["by_modulus"][p] = []
            analysis["by_modulus"][p].append(model_name)
            
            # Group by type
            if model_type not in analysis["by_type"]:
                analysis["by_type"][model_type] = []
            analysis["by_type"][model_type].append(model_name)
        
        # Summary statistics
        analysis["summary_statistics"] = {
            "total_architectures": len(analysis["by_architecture"]),
            "total_moduli": len(analysis["by_modulus"]),
            "total_types": len(analysis["by_type"]),
            "moduli_range": [min(analysis["by_modulus"].keys()), max(analysis["by_modulus"].keys())]
        }
        
        return analysis


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def main():
    """Test the topology processor."""
    print("Testing Topology Processor...")
    
    # Test with dummy data
    processor = TopologyProcessor()
    
    # Create test embeddings (circular pattern)
    p = 7
    angles = np.linspace(0, 2*np.pi, p, endpoint=False)
    test_embeddings = np.array([[np.cos(a), np.sin(a)] for a in angles])
    test_embeddings += np.random.randn(*test_embeddings.shape) * 0.1  # Add noise
    
    # Test dimensionality reduction
    print("Testing dimensionality reduction...")
    test_high_dim = np.random.randn(p, 10)  # High-dimensional test data
    reduced = processor.reduce_dimensionality(test_high_dim, "pca", 2)
    print(f"  Reduced from {test_high_dim.shape} to {reduced.shape}")
    
    # Test graph construction
    print("Testing graph construction...")
    G = processor.construct_graph(test_embeddings, p, "modular")
    print(f"  Created graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test layout computation
    print("Testing layout computation...")
    layouts = processor.compute_layouts(G, test_embeddings)
    print(f"  Computed {len(layouts)} layouts: {list(layouts.keys())}")
    
    print("✅ Topology Processor test complete!")


if __name__ == "__main__":
    main()