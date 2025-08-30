# Neural Topology Visualization Framework

An interactive web-based visualization tool for exploring how neural networks organize modular arithmetic concepts in their learned representations.

## 🎯 Overview

This framework extracts learned embeddings from trained neural network models and visualizes them as interactive 3D/2D graphs, revealing the topology of concept spaces learned by different architectures.

## 🏗️ Architecture

```
topology_viz/
├── backend/                    # Python data processing pipeline
│   ├── topology_extractor.py  # Extract representations from trained models
│   ├── topology_processor.py  # Process data for visualization
│   ├── generate_viz_data.py   # Main data generation script
│   └── data/                  # Generated visualization data
├── web_viz/                   # Interactive web interface
│   ├── index.html            # Main visualization interface
│   ├── css/style.css         # Custom styling
│   ├── js/                   # JavaScript components
│   │   ├── utils.js          # Utility functions
│   │   ├── topology_viewer.js # 3D/2D visualization engine
│   │   ├── comparison_mode.js # Model comparison tools
│   │   └── metrics_dashboard.js # Real-time metrics display
│   └── data/                 # Web-ready data files
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Generate Visualization Data

First, extract and process data from all trained models:

```bash
cd topology_viz/backend
python generate_viz_data.py
```

This will:
- Load all trained models from `../../models/`
- Extract learned embeddings and representations
- Apply dimensionality reduction (PCA, t-SNE, UMAP)
- Compute graph layouts and topology metrics
- Generate web-ready JSON files

### 2. Launch Web Interface

Open `topology_viz/web_viz/index.html` in a modern web browser.

**Note**: Due to CORS restrictions, you may need to serve the files through a local server:

```bash
cd topology_viz/web_viz
python -m http.server 8000
# Then open http://localhost:8000
```

### 3. Explore Models

- Select models from the dropdown to visualize their learned concept topology
- Choose different dimensionality reduction methods (PCA, t-SNE, UMAP)
- Switch between layout algorithms (Learned, Circular, Force-Directed, Spectral)
- Toggle graph connections and node labels
- View real-time topology quality metrics

## 📊 Features

### Interactive Visualization

- **3D/2D Graph Rendering**: Powered by Three.js for smooth, hardware-accelerated visualization
- **Multiple Layout Algorithms**: 
  - Learned embeddings from models
  - Expected circular structure
  - Force-directed graph layouts
  - Spectral embedding layouts
- **Real-time Interaction**: Orbit, zoom, pan, and select nodes for detailed information

### Model Analysis

- **Multi-Architecture Support**: LinearModel, TinyTransformer, DirectLookup, HybridMemory
- **Cross-Scale Comparison**: Models trained on p=7, 13, 17, 23
- **Quality Metrics**: Circular structure scores, silhouette analysis, graph statistics
- **Topology Validation**: Quantitative assessment of learned structure quality

### Dimensionality Reduction

- **PCA**: Linear dimensionality reduction preserving variance
- **t-SNE**: Nonlinear reduction preserving local neighborhoods  
- **UMAP**: Uniform manifold approximation with global structure preservation

### Graph Construction

- **Modular Connections**: Based on modular arithmetic relationships
- **Similarity Connections**: Based on embedding distance
- **Hybrid Approach**: Combination of structural and similarity-based connections

## 🎛️ Interface Guide

### Left Panel - Controls
- **Model Selection**: Choose models and apply filters
- **Visualization Settings**: Control reduction methods, layouts, and display options
- **Animation Controls**: Animate transitions and reset camera

### Center Panel - Visualization
- **Interactive 3D/2D Canvas**: Main visualization area
- **Node Information**: Click nodes to view detailed information
- **Screenshot Tool**: Capture current visualization

### Right Panel - Metrics Dashboard
- **Model Information**: Architecture, parameters, accuracy
- **Topology Metrics**: Circular structure score, silhouette score
- **Graph Statistics**: Nodes, edges, connectivity, density
- **Quality Assessment**: Overall quality rating with recommendations

## 📈 Understanding the Visualizations

### What You're Seeing

Each visualization shows how a neural network internally represents the numbers 0 through p-1:

- **Nodes**: Represent individual numbers (0, 1, 2, ..., p-1)
- **Node Colors**: Unique color for each number for easy identification
- **Node Positions**: Determined by the model's learned embeddings
- **Edges**: Show relationships between numbers (adjacency, arithmetic operations, similarity)

### Quality Indicators

- **Circular Structure Score**: How well the learned representation matches the expected circular arrangement
- **Silhouette Score**: How well-separated the number representations are
- **Graph Density**: Connectivity level of the learned structure

### Expected Patterns

For modular arithmetic, high-quality models should show:
- Numbers arranged in a circular pattern
- Adjacent numbers (n, n+1 mod p) close together
- Arithmetic relationships reflected in spatial proximity

## 🔬 Research Applications

### Architecture Comparison
- Compare how different architectures (Linear, Transformer, Memory-based) organize concepts
- Identify which architectures learn more interpretable representations

### Scaling Analysis
- Observe how concept organization changes with problem scale (p=7 vs p=23)
- Study the transition from traditional to memory-based learning approaches

### Interpretability Research
- Validate whether neural networks learn meaningful mathematical structure
- Explore the relationship between model accuracy and representation quality

## 📋 Model Coverage

The framework currently supports models from our neural topology research:

### p=7 Models (Traditional)
- `Linear_SGD_FullBatch_p7`: 100% accuracy, 2.5K parameters
- `Linear_Adam_SmallBatch_p7`: 100% accuracy, 2.5K parameters  
- `TinyTransformer_AdamW_p7`: 100% accuracy, 9K parameters

### p=13 Models (Memory-Based)
- `DirectLookup_Adam_p13`: 100% accuracy, 2.2K parameters
- `HybridMemory_AdamW_p13`: 100% accuracy, 3.9K parameters

### p=17 Models (Memory-Based)
- `DirectLookup_Adam_p17`: 100% accuracy, 4.9K parameters
- `HybridMemory_AdamW_p17`: 100% accuracy, 6.8K parameters

### p=23 Models (Memory-Based)
- `DirectLookup_Adam_p23`: 100% accuracy, 12.2K parameters
- `HybridMemory_AdamW_p23`: 99.8% accuracy, 14.2K parameters

## 🛠️ Technical Requirements

### Backend Dependencies
```python
torch>=1.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
networkx>=2.6
umap-learn>=0.5.0
```

### Frontend Dependencies
- Modern web browser with WebGL support
- Three.js (loaded via CDN)
- D3.js (loaded via CDN)
- Bootstrap 5 (loaded via CDN)

## 🤝 Usage Examples

### Comparing Architectures
1. Select a Linear model for p=7
2. Note the topology metrics and visual structure
3. Switch to the TinyTransformer model for p=7
4. Compare how different architectures organize the same concepts

### Analyzing Scaling
1. Start with DirectLookup_Adam_p13
2. Switch to DirectLookup_Adam_p17
3. Then to DirectLookup_Adam_p23
4. Observe how concept organization scales with problem size

### Method Comparison
1. Select any model
2. Switch between PCA, t-SNE, and UMAP reduction methods
3. Compare how different reduction methods reveal structure
4. Try different layout algorithms to see various perspectives

## 🔧 Customization

### Adding New Models
1. Place trained model files in the appropriate `models/` subdirectory
2. Update `topology_extractor.py` model inventory
3. Regenerate visualization data: `python generate_viz_data.py`

### Modifying Visualizations
- Edit `topology_processor.py` to add new reduction methods or layouts
- Modify `web_viz/js/topology_viewer.js` for rendering customizations
- Update `web_viz/css/style.css` for visual styling changes

## 📚 Related Research

This visualization framework supports the research findings documented in:
- `STATUS.md`: Complete project timeline and achievements
- `breakthrough_analysis_report.md`: Memory-based neural network breakthrough
- `scaling_analysis_report.md`: Neural network scaling limitations analysis

## 🎉 Key Insights Revealed

The interactive visualizations have revealed several key insights:

1. **Traditional vs Memory-Based Learning**: Memory-based models show perfect circular organization while traditional models exhibit more complex learned structures

2. **Architecture-Specific Patterns**: Different architectures organize concepts differently, even when achieving similar accuracy

3. **Scale-Dependent Organization**: Concept organization quality varies with problem scale, with memory-based approaches scaling linearly

4. **Validation of Learning**: High-accuracy models consistently show better circular structure scores, validating the relationship between mathematical learning and representation quality

## 🤖 Future Enhancements

Potential improvements to the framework:

- **Real-time Training Visualization**: Show how representations evolve during training
- **Interactive Graph Editing**: Allow manual adjustment of layouts for exploration
- **Advanced Comparison Tools**: Statistical comparison between multiple models
- **Export Capabilities**: Save visualizations and analysis results
- **Mobile Optimization**: Enhanced mobile interface for broader accessibility

---

**Neural Topology Visualization Framework** - Exploring the hidden geometry of neural network concept spaces through interactive visualization.

*This framework represents the culmination of our neural topology research, providing an intuitive way to explore how neural networks organize mathematical knowledge in their learned representations.*