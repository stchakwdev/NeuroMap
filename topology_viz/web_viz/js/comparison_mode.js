/**
 * ComparisonMode - Model comparison functionality for neural topology analysis
 */

class ComparisonMode {
    constructor() {
        this.isActive = false;
        this.selectedModels = [];
        this.maxComparisons = 4; // Maximum models to compare simultaneously
        
        console.log('✅ ComparisonMode initialized');
    }
    
    activate() {
        this.isActive = true;
        
        // Show comparison interface elements
        this.showComparisonControls();
        
        console.log('🔄 Comparison mode activated');
    }
    
    deactivate() {
        this.isActive = false;
        this.selectedModels = [];
        
        // Hide comparison interface elements
        this.hideComparisonControls();
        
        console.log('❌ Comparison mode deactivated');
    }
    
    addModelToComparison(modelName, modelData) {
        if (this.selectedModels.length >= this.maxComparisons) {
            console.warn('Maximum comparison limit reached');
            return false;
        }
        
        // Check if model already selected
        if (this.selectedModels.find(m => m.name === modelName)) {
            console.warn('Model already in comparison');
            return false;
        }
        
        this.selectedModels.push({
            name: modelName,
            data: modelData
        });
        
        this.updateComparisonView();
        
        console.log(`➕ Added ${modelName} to comparison (${this.selectedModels.length}/${this.maxComparisons})`);
        return true;
    }
    
    removeModelFromComparison(modelName) {
        const index = this.selectedModels.findIndex(m => m.name === modelName);
        if (index === -1) {
            return false;
        }
        
        this.selectedModels.splice(index, 1);
        this.updateComparisonView();
        
        console.log(`➖ Removed ${modelName} from comparison`);
        return true;
    }
    
    updateComparisonView() {
        if (!this.isActive || this.selectedModels.length === 0) {
            return;
        }
        
        // Create side-by-side visualization
        this.createComparisonVisualization();
        
        // Update comparison metrics
        this.updateComparisonMetrics();
    }
    
    createComparisonVisualization() {
        // This would create multiple visualization panels
        // For now, we'll update the single panel with comparison data
        const container = document.getElementById('main-visualization');
        
        if (this.selectedModels.length === 1) {
            // Single model view
            const model = this.selectedModels[0];
            // Use the existing TopologyViewer to show single model
            
        } else if (this.selectedModels.length > 1) {
            // Multi-model comparison view
            this.createMultiModelView(container);
        }
    }
    
    createMultiModelView(container) {
        // Clear existing content
        container.innerHTML = '';
        
        // Create grid layout for multiple models
        const grid = document.createElement('div');
        grid.className = 'comparison-grid';
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = this.selectedModels.length <= 2 ? 
            'repeat(2, 1fr)' : 'repeat(2, 1fr)';
        grid.style.gap = '10px';
        grid.style.height = '100%';
        
        this.selectedModels.forEach((model, index) => {
            const panel = this.createModelPanel(model, index);
            grid.appendChild(panel);
        });
        
        container.appendChild(grid);
    }
    
    createModelPanel(model, index) {
        const panel = document.createElement('div');
        panel.className = 'model-comparison-panel';
        panel.style.border = '1px solid #dee2e6';
        panel.style.borderRadius = '8px';
        panel.style.padding = '10px';
        panel.style.position = 'relative';
        
        // Model header
        const header = document.createElement('div');
        header.className = 'panel-header';
        header.innerHTML = `
            <h6 class="mb-1">${model.name}</h6>
            <p class="small text-muted mb-2">
                ${model.data.model_info.architecture} | p=${model.data.model_info.p} | 
                Acc: ${(model.data.model_info.accuracy * 100).toFixed(1)}%
            </p>
        `;
        panel.appendChild(header);
        
        // Visualization container
        const vizContainer = document.createElement('div');
        vizContainer.id = `viz-panel-${index}`;
        vizContainer.style.height = 'calc(100% - 60px)';
        vizContainer.style.position = 'relative';
        panel.appendChild(vizContainer);
        
        // Create mini TopologyViewer for this panel
        setTimeout(() => {
            this.createMiniViewer(vizContainer.id, model.data);
        }, 100);
        
        return panel;
    }
    
    createMiniViewer(containerId, modelData) {
        try {
            const miniViewer = new TopologyViewer(containerId);
            
            // Load the default visualization
            const defaultViz = Object.values(modelData.visualizations)[0];
            if (defaultViz) {
                miniViewer.loadVisualization(defaultViz, {
                    layout: 'learned_2d',
                    showEdges: true,
                    showLabels: true
                });
            }
        } catch (error) {
            console.error('Failed to create mini viewer:', error);
        }
    }
    
    updateComparisonMetrics() {
        if (this.selectedModels.length < 2) {
            return;
        }
        
        // Create comparison table or chart
        const comparisonData = this.computeComparisonMetrics();
        this.displayComparisonResults(comparisonData);
    }
    
    computeComparisonMetrics() {
        const comparison = {
            models: [],
            rankings: {},
            correlations: {}
        };
        
        this.selectedModels.forEach(model => {
            const metrics = model.data.topology_metrics;
            const modelInfo = model.data.model_info;
            
            comparison.models.push({
                name: model.name,
                architecture: modelInfo.architecture,
                p: modelInfo.p,
                type: modelInfo.type,
                accuracy: modelInfo.accuracy,
                circular_score: metrics.circular_structure_score || 0,
                silhouette_score: metrics.silhouette_score || 0,
                adjacency_distance: metrics.mean_adjacency_distance || 0
            });
        });
        
        // Rank models by different criteria
        comparison.rankings.accuracy = [...comparison.models]
            .sort((a, b) => b.accuracy - a.accuracy)
            .map(m => m.name);
            
        comparison.rankings.circular_structure = [...comparison.models]
            .sort((a, b) => b.circular_score - a.circular_score)
            .map(m => m.name);
            
        comparison.rankings.silhouette = [...comparison.models]
            .sort((a, b) => b.silhouette_score - a.silhouette_score)
            .map(m => m.name);
        
        return comparison;
    }
    
    displayComparisonResults(comparisonData) {
        // This could display comparison results in the metrics panel
        // For now, just log the results
        console.log('📊 Comparison Results:', comparisonData);
    }
    
    showComparisonControls() {
        // Show additional UI elements for comparison mode
        const comparisonControls = document.getElementById('comparison-controls');
        if (comparisonControls) {
            comparisonControls.style.display = 'block';
        }
    }
    
    hideComparisonControls() {
        // Hide comparison-specific UI elements
        const comparisonControls = document.getElementById('comparison-controls');
        if (comparisonControls) {
            comparisonControls.style.display = 'none';
        }
    }
    
    exportComparison() {
        if (this.selectedModels.length === 0) {
            console.warn('No models selected for export');
            return;
        }
        
        const comparisonData = this.computeComparisonMetrics();
        
        const dataStr = JSON.stringify(comparisonData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'topology_comparison.json';
        link.click();
        
        console.log('📁 Comparison data exported');
    }
    
    reset() {
        this.selectedModels = [];
        this.updateComparisonView();
    }
}

// Global instance
window.ComparisonMode = ComparisonMode;