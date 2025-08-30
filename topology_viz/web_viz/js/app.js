/**
 * Main Neural Topology Application
 */

class NeuralTopologyApp {
    constructor() {
        this.modelsData = null;
        this.currentModel = null;
        this.topologyViewer = null;
        this.metricsDashboard = null;
        this.comparisonMode = null;
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing Neural Topology Application...');
        
        // Initialize components
        this.topologyViewer = new TopologyViewer('main-visualization');
        this.metricsDashboard = new MetricsDashboard();
        this.comparisonMode = new ComparisonMode();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load models data
        await this.loadModelsData();
        
        // Hide loading indicator
        this.hideLoading();
        
        console.log('✅ Neural Topology Application initialized');
    }
    
    async loadModelsData() {
        try {
            console.log('📊 Loading models data...');
            
            const response = await fetch('data/models.json');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.modelsData = await response.json();
            
            console.log(`✅ Loaded data for ${Object.keys(this.modelsData.models).length} models`);
            
            // Populate model selector
            this.populateModelSelector();
            
        } catch (error) {
            console.error('❌ Failed to load models data:', error);
            this.showError('Failed to load visualization data. Please check that the data generation pipeline has been run.');
        }
    }
    
    populateModelSelector() {
        const selector = document.getElementById('model-select');
        if (!selector) {
            console.error('Model selector element not found');
            return;
        }
        
        // Clear existing options
        selector.innerHTML = '<option value="">Select a model...</option>';
        
        // Add models grouped by type and p value
        const models = this.modelsData.models;
        const groupedModels = this.groupModels(models);
        
        // Create optgroups
        Object.entries(groupedModels).forEach(([groupName, modelList]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = groupName;
            
            modelList.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                const modelData = models[modelName];
                if (modelData && !modelData.error) {
                    const info = modelData.model_info;
                    option.textContent = `${modelName} (${info.architecture}, p=${info.p})`;
                } else {
                    option.textContent = `${modelName} (ERROR)`;
                    option.disabled = true;
                }
                optgroup.appendChild(option);
            });
            
            selector.appendChild(optgroup);
        });
        
        console.log('📋 Model selector populated');
    }
    
    groupModels(models) {
        const groups = {};
        
        Object.entries(models).forEach(([modelName, modelData]) => {
            if (modelData.error) {
                if (!groups['❌ Failed Models']) {
                    groups['❌ Failed Models'] = [];
                }
                groups['❌ Failed Models'].push(modelName);
                return;
            }
            
            const info = modelData.model_info;
            const groupKey = `p=${info.p} - ${info.type.charAt(0).toUpperCase() + info.type.slice(1)}`;
            
            if (!groups[groupKey]) {
                groups[groupKey] = [];
            }
            groups[groupKey].push(modelName);
        });
        
        return groups;
    }
    
    setupEventListeners() {
        // Model selection
        const modelSelector = document.getElementById('model-select');
        if (modelSelector) {
            modelSelector.addEventListener('change', (e) => {
                this.selectModel(e.target.value);
            });
        }
        
        // Visualization controls
        ['reduction-method', 'connection-method', 'layout-method'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.updateVisualization());
            }
        });
        
        // Display toggles
        ['show-edges', 'show-labels'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.updateVisualization());
            }
        });
        
        // Animation controls
        const animateBtn = document.getElementById('animate-layout');
        if (animateBtn) {
            animateBtn.addEventListener('click', () => this.topologyViewer.animateLayoutTransition());
        }
        
        // Camera controls
        const resetCameraBtn = document.getElementById('reset-camera');
        if (resetCameraBtn) {
            resetCameraBtn.addEventListener('click', () => this.topologyViewer.resetCamera());
        }
        
        // Screenshot
        const screenshotBtn = document.getElementById('screenshot-btn');
        if (screenshotBtn) {
            screenshotBtn.addEventListener('click', () => this.topologyViewer.takeScreenshot());
        }
        
        // Mode switches
        const singleViewTab = document.getElementById('single-view-tab');
        const comparisonViewTab = document.getElementById('comparison-view-tab');
        
        if (singleViewTab) {
            singleViewTab.addEventListener('click', (e) => {
                e.preventDefault();
                this.activateSingleView();
            });
        }
        
        if (comparisonViewTab) {
            comparisonViewTab.addEventListener('click', (e) => {
                e.preventDefault();
                this.activateComparisonView();
            });
        }
        
        console.log('🎛️ Event listeners setup complete');
    }
    
    selectModel(modelName) {
        if (!modelName || !this.modelsData) {
            this.currentModel = null;
            this.metricsDashboard.reset();
            return;
        }
        
        const modelData = this.modelsData.models[modelName];
        if (!modelData || modelData.error) {
            console.error('Invalid model selected:', modelName);
            this.showError(`Model ${modelName} is not available or has errors`);
            return;
        }
        
        this.currentModel = modelData;
        
        console.log('🎯 Selected model:', modelName);
        
        // Update metrics dashboard
        this.metricsDashboard.updateModelInfo(modelData);
        
        // Update visualization
        this.updateVisualization();
    }
    
    updateVisualization() {
        if (!this.currentModel) {
            return;
        }
        
        const reductionMethod = document.getElementById('reduction-method')?.value || 'pca';
        const connectionMethod = document.getElementById('connection-method')?.value || 'modular';
        const layoutMethod = document.getElementById('layout-method')?.value || 'learned_2d';
        const showConnections = document.getElementById('show-edges')?.checked !== false;
        const showLabels = document.getElementById('show-labels')?.checked !== false;
        
        const vizKey = `${reductionMethod}_${connectionMethod}`;
        const visualization = this.currentModel.visualizations[vizKey];
        
        if (!visualization) {
            this.showError(`Visualization not available: ${vizKey}`);
            return;
        }
        
        // Update topology viewer
        this.topologyViewer.loadVisualization(visualization, {
            layout: layoutMethod,
            showEdges: showConnections,
            showLabels: showLabels
        });
        
        // Update metrics dashboard with visualization-specific metrics
        this.metricsDashboard.updateVisualizationMetrics(visualization);
        
        console.log(`🔄 Updated visualization: ${vizKey} with ${layoutMethod} layout`);
    }
    
    activateSingleView() {
        this.comparisonMode.deactivate();
        
        // Update UI
        document.getElementById('single-view-tab').classList.add('active');
        document.getElementById('comparison-view-tab').classList.remove('active');
        
        // Reset to single visualization
        const container = document.getElementById('main-visualization');
        container.innerHTML = '';
        
        // Recreate topology viewer
        this.topologyViewer = new TopologyViewer('main-visualization');
        
        // Reload current model if any
        if (this.currentModel) {
            this.updateVisualization();
        }
        
        console.log('👁️ Single view activated');
    }
    
    activateComparisonView() {
        this.comparisonMode.activate();
        
        // Update UI
        document.getElementById('single-view-tab').classList.remove('active');
        document.getElementById('comparison-view-tab').classList.add('active');
        
        console.log('🔄 Comparison view activated');
    }
    
    showLoading(show = true) {
        const loadingIndicator = document.getElementById('loading-indicator');
        const mainViz = document.getElementById('main-visualization');
        
        if (loadingIndicator) {
            loadingIndicator.style.display = show ? 'flex' : 'none';
        }
        if (mainViz) {
            mainViz.style.display = show ? 'none' : 'block';
        }
    }
    
    hideLoading() {
        this.showLoading(false);
    }
    
    showError(message) {
        console.error('❌', message);
        
        // Show error in main visualization area
        const container = document.getElementById('main-visualization');
        if (container) {
            container.innerHTML = `
                <div class="alert alert-danger m-4" role="alert">
                    <h5><i class="fas fa-exclamation-triangle"></i> Error</h5>
                    <p>${message}</p>
                </div>
            `;
        }
    }
    
    // Utility methods
    getAvailableVisualizations(modelData) {
        return Object.keys(modelData.visualizations || {});
    }
    
    getModelSummary(modelData) {
        const info = modelData.model_info;
        const metrics = modelData.topology_metrics;
        
        return {
            name: modelData.model_name,
            architecture: info.architecture,
            p: info.p,
            accuracy: info.accuracy,
            circular_score: metrics.circular_structure_score || 0,
            silhouette_score: metrics.silhouette_score || 0,
            type: info.type
        };
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🌐 DOM ready, initializing Neural Topology Application...');
    
    // Create global application instance
    window.NeuralTopologyApp = new NeuralTopologyApp();
});

console.log('📜 Neural Topology Application script loaded');