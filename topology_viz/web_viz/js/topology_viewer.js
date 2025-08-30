/**
 * TopologyViewer - 3D/2D interactive visualization engine for neural topology
 */

class TopologyViewer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.nodes = [];
        this.edges = [];
        this.currentVisualization = null;
        this.is3D = true;
        
        this.init();
    }
    
    init() {
        // Create Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf8f9fa);
        
        // Setup camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 5);
        
        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Add renderer to container
        this.container.appendChild(this.renderer.domElement);
        
        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Add lights
        this.setupLighting();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start animation loop
        this.animate();
        
        console.log('✅ TopologyViewer initialized');
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
    }
    
    clearVisualization() {
        // Remove existing nodes and edges
        this.nodes.forEach(node => this.scene.remove(node));
        this.edges.forEach(edge => this.scene.remove(edge));
        this.nodes = [];
        this.edges = [];
    }
    
    loadVisualization(visualizationData, options = {}) {
        console.log('Loading visualization:', visualizationData);
        
        this.clearVisualization();
        this.currentVisualization = visualizationData;
        
        const layout = options.layout || 'learned_2d';
        const showEdges = options.showEdges !== false;
        const showLabels = options.showLabels !== false;
        
        // Get layout positions
        const layoutData = visualizationData.layouts[layout];
        if (!layoutData) {
            console.error('Layout not found:', layout);
            return;
        }
        
        const positions = layoutData.positions;
        
        // Create nodes
        visualizationData.nodes.forEach(nodeData => {
            const nodeId = nodeData.id;
            const position = positions[nodeId.toString()];
            
            if (!position) {
                console.warn('No position for node:', nodeId);
                return;
            }
            
            const node = this.createNode(nodeData, position, showLabels);
            this.nodes.push(node);
            this.scene.add(node);
        });
        
        // Create edges
        if (showEdges) {
            visualizationData.edges.forEach(edgeData => {
                const sourcePos = positions[edgeData.source.toString()];
                const targetPos = positions[edgeData.target.toString()];
                
                if (!sourcePos || !targetPos) {
                    return;
                }
                
                const edge = this.createEdge(sourcePos, targetPos, edgeData);
                this.edges.push(edge);
                this.scene.add(edge);
            });
        }
        
        // Update camera to fit content
        this.fitCameraToContent();
        
        console.log(`✅ Loaded visualization with ${this.nodes.length} nodes, ${this.edges.length} edges`);
    }
    
    createNode(nodeData, position, showLabels = true) {
        const group = new THREE.Group();
        
        // Create sphere for node
        const geometry = new THREE.SphereGeometry(0.1, 16, 16);
        const color = this.getNodeColor(nodeData.id, nodeData.value);
        const material = new THREE.MeshLambertMaterial({ color: color });
        const sphere = new THREE.Mesh(geometry, material);
        
        // Position the node
        if (position.length === 3) {
            sphere.position.set(position[0], position[1], position[2]);
        } else {
            sphere.position.set(position[0], position[1], 0);
        }
        
        sphere.userData = nodeData;
        group.add(sphere);
        
        // Add label if requested
        if (showLabels) {
            const label = this.createNodeLabel(nodeData.label, sphere.position);
            group.add(label);
        }
        
        return group;
    }
    
    createEdge(sourcePos, targetPos, edgeData) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        
        // Handle 2D and 3D positions
        const source3D = sourcePos.length === 3 ? sourcePos : [sourcePos[0], sourcePos[1], 0];
        const target3D = targetPos.length === 3 ? targetPos : [targetPos[0], targetPos[1], 0];
        
        positions.push(...source3D, ...target3D);
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        
        const color = this.getEdgeColor(edgeData.type);
        const material = new THREE.LineBasicMaterial({ 
            color: color,
            opacity: 0.6,
            transparent: true
        });
        
        const line = new THREE.Line(geometry, material);
        line.userData = edgeData;
        
        return line;
    }
    
    createNodeLabel(text, position) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 32;
        
        context.fillStyle = '#000000';
        context.font = '16px Arial';
        context.textAlign = 'center';
        context.fillText(text, 32, 20);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        
        sprite.position.copy(position);
        sprite.position.y += 0.2;
        sprite.scale.set(0.5, 0.25, 1);
        
        return sprite;
    }
    
    getNodeColor(nodeId, value) {
        // Generate consistent colors for nodes based on their value
        const hue = (value * 360 / 23) % 360; // Distribute colors around color wheel
        return new THREE.Color().setHSL(hue / 360, 0.7, 0.5);
    }
    
    getEdgeColor(edgeType) {
        const colorMap = {
            'adjacent': 0x3498db,      // Blue
            'addition': 0x2ecc71,      // Green  
            'similarity': 0xe74c3c,    // Red
            'hybrid': 0x9b59b6,        // Purple
            'default': 0x95a5a6        // Gray
        };
        
        return colorMap[edgeType] || colorMap['default'];
    }
    
    fitCameraToContent() {
        if (this.nodes.length === 0) return;
        
        // Calculate bounding box
        const box = new THREE.Box3();
        this.nodes.forEach(node => {
            box.expandByObject(node);
        });
        
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        // Position camera to see all content
        const maxDim = Math.max(size.x, size.y, size.z);
        const distance = maxDim * 2;
        
        this.camera.position.set(center.x, center.y, center.z + distance);
        this.controls.target.copy(center);
        this.controls.update();
    }
    
    animateLayoutTransition() {
        // Simple animation - could be enhanced
        this.nodes.forEach((node, index) => {
            const delay = index * 50;
            setTimeout(() => {
                node.scale.set(1.2, 1.2, 1.2);
                setTimeout(() => {
                    node.scale.set(1, 1, 1);
                }, 200);
            }, delay);
        });
    }
    
    resetCamera() {
        this.camera.position.set(0, 0, 5);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
        this.fitCameraToContent();
    }
    
    takeScreenshot() {
        this.renderer.render(this.scene, this.camera);
        const dataURL = this.renderer.domElement.toDataURL('image/png');
        
        // Download the screenshot
        const link = document.createElement('a');
        link.download = 'neural_topology_screenshot.png';
        link.href = dataURL;
        link.click();
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Global instance
window.TopologyViewer = TopologyViewer;