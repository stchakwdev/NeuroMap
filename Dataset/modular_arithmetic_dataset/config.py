"""Configuration constants for modular arithmetic dataset."""

# Dataset parameters
DEFAULT_P = 17  # Prime modulus - chosen for optimal toy model size
REPRESENTATION_TYPES = ['one_hot', 'embedding', 'integer']
DEFAULT_REPRESENTATION = 'embedding'

# Model parameters (for reference)
MODEL_CONFIGS = {
    'transformer': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'vocab_size': DEFAULT_P
    },
    'mamba': {
        'd_model': 64,
        'n_layer': 2, 
        'vocab_size': DEFAULT_P
    }
}

# Validation thresholds
CIRCULAR_STRUCTURE_THRESHOLD = 0.1
ADJACENCY_DISTANCE_THRESHOLD = 0.2
ACCURACY_THRESHOLD = 0.99  # Models should achieve near-perfect accuracy

# File paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

