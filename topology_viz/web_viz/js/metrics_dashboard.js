/**
 * MetricsDashboard - Real-time metrics display for neural topology analysis
 */

class MetricsDashboard {
    constructor() {
        this.currentModel = null;
        this.currentVisualization = null;
        
        console.log('✅ MetricsDashboard initialized');
    }
    
    updateModelInfo(modelData) {
        this.currentModel = modelData;
        
        // Update model info section
        const modelInfo = modelData.model_info;
        
        document.getElementById('model-name').textContent = modelData.model_name;
        document.getElementById('model-architecture').textContent = modelInfo.architecture;
        document.getElementById('model-p-value').textContent = `p = ${modelInfo.p}`;
        document.getElementById('model-type').textContent = modelInfo.type;
        document.getElementById('model-parameters').textContent = this.formatParameters(modelInfo);
        
        // Update topology metrics
        this.updateTopologyMetrics(modelData.topology_metrics);
        
        console.log('📊 Updated model info for:', modelData.model_name);
    }
    
    updateVisualizationMetrics(visualizationData) {
        this.currentVisualization = visualizationData;
        
        // Update graph statistics
        const graphMetrics = visualizationData.graph_metrics;
        
        document.getElementById('num-nodes').textContent = graphMetrics.num_nodes || '-';
        document.getElementById('num-edges').textContent = graphMetrics.num_edges || '-';
        document.getElementById('graph-density').textContent = 
            graphMetrics.density ? graphMetrics.density.toFixed(3) : '-';
        document.getElementById('is-connected').textContent = 
            graphMetrics.is_connected ? 'Yes' : 'No';
        
        // Update quality assessment
        this.updateQualityAssessment();
        
        console.log('📈 Updated visualization metrics');
    }
    
    updateTopologyMetrics(topologyMetrics) {
        // Circular structure score
        const circularScore = topologyMetrics.circular_structure_score || 0;
        document.getElementById('circular-score').textContent = circularScore.toFixed(3);
        this.updateMetricBar('circular-score-bar', circularScore);
        
        // Silhouette score (normalized to 0-1)
        const silhouetteScore = topologyMetrics.silhouette_score || 0;
        const normalizedSilhouette = (silhouetteScore + 1) / 2; // Convert from [-1,1] to [0,1]
        document.getElementById('silhouette-score').textContent = silhouetteScore.toFixed(3);
        this.updateMetricBar('silhouette-score-bar', normalizedSilhouette);
        
        // Model accuracy
        if (this.currentModel) {
            const accuracy = this.currentModel.model_info.accuracy;
            document.getElementById('model-accuracy').textContent = 
                (accuracy * 100).toFixed(1) + '%';
            this.updateMetricBar('accuracy-bar', accuracy);
        }
    }
    
    updateMetricBar(barId, value) {
        const bar = document.getElementById(barId);
        const fill = bar.querySelector('.metric-bar-fill');
        
        const percentage = Math.max(0, Math.min(100, value * 100));
        fill.style.width = percentage + '%';
        
        // Color coding
        if (percentage >= 80) {
            fill.style.backgroundColor = '#2ecc71'; // Green
        } else if (percentage >= 60) {
            fill.style.backgroundColor = '#f39c12'; // Orange
        } else {
            fill.style.backgroundColor = '#e74c3c'; // Red
        }
    }
    
    updateQualityAssessment() {
        const qualityContainer = document.getElementById('quality-assessment');
        
        if (!this.currentModel || !this.currentVisualization) {
            qualityContainer.innerHTML = '<p class="text-muted">No model selected</p><p class="small">Select a model to see quality assessment</p>';
            return;
        }
        
        const circularScore = this.currentModel.topology_metrics.circular_structure_score || 0;
        const silhouetteScore = this.currentModel.topology_metrics.silhouette_score || 0;
        const accuracy = this.currentModel.model_info.accuracy;
        
        // Calculate overall quality
        const overallQuality = (circularScore + ((silhouetteScore + 1) / 2) + accuracy) / 3;
        
        let qualityText, qualityClass, recommendations;
        
        if (overallQuality >= 0.8) {
            qualityText = 'Excellent';
            qualityClass = 'text-success';
            recommendations = ['Model shows excellent concept organization', 'Representations are well-structured'];
        } else if (overallQuality >= 0.6) {
            qualityText = 'Good';
            qualityClass = 'text-warning';
            recommendations = ['Model shows good concept organization', 'Some structure improvements possible'];
        } else if (overallQuality >= 0.4) {
            qualityText = 'Fair';
            qualityClass = 'text-warning';
            recommendations = ['Model shows basic concept organization', 'Consider architecture changes'];
        } else {
            qualityText = 'Poor';
            qualityClass = 'text-danger';
            recommendations = ['Model shows poor concept organization', 'Significant improvements needed'];
        }
        
        qualityContainer.innerHTML = `
            <div class="quality-score ${qualityClass}">
                <h6>${qualityText}</h6>
                <p class="small mb-2">Overall Score: ${overallQuality.toFixed(3)}</p>
            </div>
            <div class="recommendations">
                <h6 class="small fw-bold">Recommendations:</h6>
                ${recommendations.map(rec => `<p class="small mb-1">• ${rec}</p>`).join('')}
            </div>
        `;
    }
    
    formatParameters(modelInfo) {
        // Try to estimate parameter count
        const p = modelInfo.p;
        const architecture = modelInfo.architecture;
        
        let estimate = 'Unknown';
        
        if (architecture === 'LinearModel') {
            estimate = `~${(p * 32 + p).toLocaleString()}`;
        } else if (architecture === 'TinyTransformer') {
            estimate = `~${(p * 32 * 4).toLocaleString()}`;
        } else if (architecture === 'DirectLookupModel') {
            estimate = `~${(p * p * 10).toLocaleString()}`;
        } else if (architecture === 'HybridMemoryModel') {
            estimate = `~${(p * p * 15).toLocaleString()}`;
        }
        
        return estimate;
    }
    
    reset() {
        this.currentModel = null;
        this.currentVisualization = null;
        
        // Reset all display elements
        ['model-name', 'model-architecture', 'model-p-value', 'model-type', 'model-parameters'].forEach(id => {
            document.getElementById(id).textContent = '-';
        });
        
        ['circular-score', 'silhouette-score', 'model-accuracy'].forEach(id => {
            document.getElementById(id).textContent = '-';
        });
        
        ['num-nodes', 'num-edges', 'graph-density', 'is-connected'].forEach(id => {
            document.getElementById(id).textContent = '-';
        });
        
        // Reset metric bars
        ['circular-score-bar', 'silhouette-score-bar', 'accuracy-bar'].forEach(id => {
            this.updateMetricBar(id, 0);
        });
        
        this.updateQualityAssessment();
    }
}

// Global instance
window.MetricsDashboard = MetricsDashboard;