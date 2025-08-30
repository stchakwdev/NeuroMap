/**
 * Utility functions for Neural Topology Visualization
 */

// === Color Utilities ===

/**
 * Generate colors for nodes based on their value
 */
function generateNodeColors(numNodes) {
    const colors = [];
    for (let i = 0; i < numNodes; i++) {
        const hue = (i / numNodes) * 360;
        colors.push(`hsl(${hue}, 70%, 60%)`);
    }
    return colors;
}

/**
 * Convert HSL to RGB
 */
function hslToRgb(h, s, l) {
    h /= 360;
    s /= 100;
    l /= 100;
    
    const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
    };
    
    let r, g, b;
    
    if (s === 0) {
        r = g = b = l; // achromatic
    } else {
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    
    return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
    };
}

/**
 * Generate color for edge based on its type
 */
function getEdgeColor(edgeType) {
    const colors = {
        'adjacent': '#0066cc',      // Blue for adjacent pairs
        'addition': '#28a745',      // Green for addition relationships
        'similarity': '#dc3545',    // Red for similarity-based
        'hybrid': '#6f42c1',        // Purple for hybrid
        'default': '#6c757d'        // Gray for default
    };
    return colors[edgeType] || colors.default;
}

// === Math Utilities ===

/**
 * Calculate distance between two points
 */
function distance(p1, p2) {
    const dx = p1[0] - p2[0];
    const dy = p1[1] - p2[1];
    const dz = p1.length > 2 ? (p1[2] - p2[2]) : 0;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Normalize vector
 */
function normalize(vector) {
    const length = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return length === 0 ? vector : vector.map(val => val / length);
}

/**
 * Linear interpolation between two values
 */
function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Linear interpolation between two vectors
 */
function lerpVector(v1, v2, t) {
    return v1.map((val, i) => lerp(val, v2[i], t));
}

/**
 * Clamp value between min and max
 */
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

// === Data Utilities ===

/**
 * Deep clone an object
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== "object") return obj;
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof Array) return obj.map(item => deepClone(item));
    if (typeof obj === "object") {
        const clonedObj = {};
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                clonedObj[key] = deepClone(obj[key]);
            }
        }
        return clonedObj;
    }
}

/**
 * Debounce function calls
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func.apply(this, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(this, args);
    };
}

/**
 * Throttle function calls
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// === DOM Utilities ===

/**
 * Create HTML element with attributes
 */
function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);
    
    // Set attributes
    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'innerHTML') {
            element.innerHTML = value;
        } else if (key.startsWith('on')) {
            element.addEventListener(key.substring(2).toLowerCase(), value);
        } else {
            element.setAttribute(key, value);
        }
    }
    
    // Add children
    children.forEach(child => {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else {
            element.appendChild(child);
        }
    });
    
    return element;
}

/**
 * Show/hide element with fade animation
 */
function fadeToggle(element, show, duration = 300) {
    if (show) {
        element.style.display = 'block';
        element.style.opacity = '0';
        element.offsetHeight; // Trigger reflow
        element.style.transition = `opacity ${duration}ms ease`;
        element.style.opacity = '1';
    } else {
        element.style.transition = `opacity ${duration}ms ease`;
        element.style.opacity = '0';
        setTimeout(() => {
            element.style.display = 'none';
        }, duration);
    }
}

// === Animation Utilities ===

/**
 * Easing functions
 */
const Easing = {
    linear: t => t,
    easeInQuad: t => t * t,
    easeOutQuad: t => t * (2 - t),
    easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeInCubic: t => t * t * t,
    easeOutCubic: t => (--t) * t * t + 1,
    easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
};

/**
 * Animate value over time
 */
function animate(from, to, duration, easing = Easing.easeInOutQuad, callback) {
    const start = performance.now();
    
    function step(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = easing(progress);
        const currentValue = from + (to - from) * easedProgress;
        
        callback(currentValue, progress);
        
        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }
    
    requestAnimationFrame(step);
}

// === Validation Utilities ===

/**
 * Check if value is a number
 */
function isNumber(value) {
    return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

/**
 * Check if object has required properties
 */
function hasRequiredProperties(obj, requiredProps) {
    return requiredProps.every(prop => obj.hasOwnProperty(prop));
}

/**
 * Validate visualization data structure
 */
function validateVisualizationData(data) {
    const errors = [];
    
    if (!data) {
        errors.push('Data is null or undefined');
        return errors;
    }
    
    if (!data.nodes || !Array.isArray(data.nodes)) {
        errors.push('Missing or invalid nodes array');
    }
    
    if (!data.edges || !Array.isArray(data.edges)) {
        errors.push('Missing or invalid edges array');
    }
    
    if (!data.layouts || typeof data.layouts !== 'object') {
        errors.push('Missing or invalid layouts object');
    }
    
    // Validate nodes
    if (data.nodes) {
        data.nodes.forEach((node, index) => {
            if (!hasRequiredProperties(node, ['id', 'label'])) {
                errors.push(`Node ${index} missing required properties (id, label)`);
            }
        });
    }
    
    // Validate edges
    if (data.edges) {
        data.edges.forEach((edge, index) => {
            if (!hasRequiredProperties(edge, ['source', 'target'])) {
                errors.push(`Edge ${index} missing required properties (source, target)`);
            }
        });
    }
    
    return errors;
}

// === Format Utilities ===

/**
 * Format number with specified decimal places
 */
function formatNumber(num, decimals = 2) {
    if (!isNumber(num)) return 'N/A';
    return num.toFixed(decimals);
}

/**
 * Format percentage
 */
function formatPercentage(num, decimals = 1) {
    if (!isNumber(num)) return 'N/A';
    return `${(num * 100).toFixed(decimals)}%`;
}

/**
 * Format large numbers with SI suffixes
 */
function formatLargeNumber(num) {
    if (!isNumber(num)) return 'N/A';
    
    const suffixes = ['', 'K', 'M', 'B', 'T'];
    let suffixIndex = 0;
    
    while (num >= 1000 && suffixIndex < suffixes.length - 1) {
        num /= 1000;
        suffixIndex++;
    }
    
    return `${num.toFixed(1)}${suffixes[suffixIndex]}`;
}

// === Export Utilities ===

/**
 * Download text as file
 */
function downloadText(text, filename, mimeType = 'text/plain') {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

/**
 * Download JSON data
 */
function downloadJSON(data, filename) {
    const jsonString = JSON.stringify(data, null, 2);
    downloadText(jsonString, filename, 'application/json');
}

/**
 * Download canvas as image
 */
function downloadCanvas(canvas, filename = 'visualization.png') {
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

// === Browser Utilities ===

/**
 * Check if browser supports WebGL
 */
function supportsWebGL() {
    try {
        const canvas = document.createElement('canvas');
        return !!(window.WebGLRenderingContext && 
                 (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch (e) {
        return false;
    }
}

/**
 * Get device pixel ratio
 */
function getPixelRatio() {
    return window.devicePixelRatio || 1;
}

/**
 * Check if device is mobile
 */
function isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

// === Performance Utilities ===

/**
 * Simple performance timer
 */
class PerformanceTimer {
    constructor(name) {
        this.name = name;
        this.startTime = null;
    }
    
    start() {
        this.startTime = performance.now();
        console.time(this.name);
    }
    
    end() {
        if (this.startTime) {
            const duration = performance.now() - this.startTime;
            console.timeEnd(this.name);
            return duration;
        }
        return 0;
    }
}

// === Event Utilities ===

/**
 * Custom event emitter
 */
class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.events[event]) return;
        this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
    
    emit(event, data) {
        if (!this.events[event]) return;
        this.events[event].forEach(callback => callback(data));
    }
    
    once(event, callback) {
        const onceCallback = (data) => {
            callback(data);
            this.off(event, onceCallback);
        };
        this.on(event, onceCallback);
    }
}

// Global utility object
window.NeuralTopologyUtils = {
    // Color utilities
    generateNodeColors,
    hslToRgb,
    getEdgeColor,
    
    // Math utilities
    distance,
    normalize,
    lerp,
    lerpVector,
    clamp,
    
    // Data utilities
    deepClone,
    debounce,
    throttle,
    
    // DOM utilities
    createElement,
    fadeToggle,
    
    // Animation utilities
    Easing,
    animate,
    
    // Validation utilities
    isNumber,
    hasRequiredProperties,
    validateVisualizationData,
    
    // Format utilities
    formatNumber,
    formatPercentage,
    formatLargeNumber,
    
    // Export utilities
    downloadText,
    downloadJSON,
    downloadCanvas,
    
    // Browser utilities
    supportsWebGL,
    getPixelRatio,
    isMobile,
    
    // Performance utilities
    PerformanceTimer,
    
    // Event utilities
    EventEmitter
};