"""
Dataset module for modular arithmetic neural network interpretability research.
"""

from .dataset import ModularArithmeticDataset, create_mod_p_datasets
from .validation import CircularStructureValidator, create_validation_test_suite
from .config import DEFAULT_P, REPRESENTATION_TYPES, ACCURACY_THRESHOLD

__all__ = [
    'ModularArithmeticDataset',
    'create_mod_p_datasets', 
    'CircularStructureValidator',
    'create_validation_test_suite',
    'DEFAULT_P',
    'REPRESENTATION_TYPES',
    'ACCURACY_THRESHOLD'
]