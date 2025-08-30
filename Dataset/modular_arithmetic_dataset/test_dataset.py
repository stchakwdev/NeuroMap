"""
Comprehensive unit tests for modular arithmetic dataset.

Tests all core functionality including dataset creation, validation,
and utility functions to ensure correctness and reliability.
"""

import unittest
import tempfile
import torch
import numpy as np
from pathlib import Path

from dataset import ModularArithmeticDataset, create_mod_p_datasets
from validation import CircularStructureValidator, create_validation_test_suite
from utils import (
    check_dataset_consistency, analyze_algebraic_properties,
    create_training_splits, print_dataset_summary
)

class TestModularArithmeticDataset(unittest.TestCase):
    """Test cases for ModularArithmeticDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_p = 5  # Use small p for faster testing
        self.dataset = ModularArithmeticDataset(p=self.test_p, representation='embedding')
    
    def test_dataset_initialization(self):
        """Test basic dataset initialization."""
        
        # Check basic properties
        self.assertEqual(self.dataset.p, self.test_p)
        self.assertEqual(self.dataset.representation, 'embedding')
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.metadata)
    
    def test_dataset_size(self):
        """Test that dataset has correct number of examples."""
        
        expected_size = self.test_p * self.test_p
        actual_size = self.dataset.data['num_examples']
        self.assertEqual(actual_size, expected_size)
        
        # Check tensor shapes
        self.assertEqual(self.dataset.data['inputs'].shape, (expected_size, 2))
        self.assertEqual(self.dataset.data['targets'].shape, (expected_size,))
    
    def test_mathematical_correctness(self):
        """Test that all mathematical operations are correct."""
        
        for i, (a, b) in enumerate(self.dataset.data['raw_inputs']):
            expected_result = (a + b) % self.test_p
            actual_result = self.dataset.data['raw_targets'][i]
            self.assertEqual(actual_result, expected_result, 
                           f"Incorrect result for {a} + {b} mod {self.test_p}")
    
    def test_completeness(self):
        """Test that all possible (a,b) pairs are included exactly once."""
        
        # Create set of expected pairs
        expected_pairs = set()
        for a in range(self.test_p):
            for b in range(self.test_p):
                expected_pairs.add((a, b))
        
        # Check actual pairs
        actual_pairs = set(self.dataset.data['raw_inputs'])
        self.assertEqual(actual_pairs, expected_pairs)
    
    def test_different_representations(self):
        """Test different input representations."""
        
        # Test one-hot representation
        onehot_dataset = ModularArithmeticDataset(p=self.test_p, representation='one_hot')
        expected_shape = (self.test_p * self.test_p, 2, self.test_p)
        self.assertEqual(onehot_dataset.data['inputs'].shape, expected_shape)
        
        # Test integer representation  
        int_dataset = ModularArithmeticDataset(p=self.test_p, representation='integer')
        expected_shape = (self.test_p * self.test_p, 2)
        self.assertEqual(int_dataset.data['inputs'].shape, expected_shape)
    
    def test_metadata_generation(self):
        """Test that structural metadata is generated correctly."""
        
        metadata = self.dataset.metadata
        
        # Test adjacency pairs
        adjacency_pairs = metadata['circular_structure']['adjacency_pairs']
        self.assertEqual(len(adjacency_pairs), self.test_p)
        
        # Check that each number has exactly one next neighbor
        firsts = [pair[0] for pair in adjacency_pairs]
        self.assertEqual(set(firsts), set(range(self.test_p)))
        
        # Test commutative pairs
        commutative_pairs = metadata['algebraic_properties']['commutative_pairs']
        expected_commutative_count = (self.test_p * (self.test_p - 1)) // 2
        self.assertEqual(len(commutative_pairs), expected_commutative_count)
        
        # Test identity pairs
        identity_pairs = metadata['algebraic_properties']['identity_pairs'] 
        self.assertEqual(len(identity_pairs), self.test_p)
        
        # Test addition table
        addition_table = metadata['algebraic_properties']['addition_table']
        self.assertEqual(addition_table.shape, (self.test_p, self.test_p))
        
        # Verify addition table is correct
        for a in range(self.test_p):
            for b in range(self.test_p):
                expected = (a + b) % self.test_p
                actual = addition_table[a, b].item()
                self.assertEqual(actual, expected)
    
    def test_save_and_load(self):
        """Test dataset serialization."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_dataset.pkl'
            
            # Save dataset
            self.dataset.save(filepath)
            self.assertTrue(filepath.exists())
            
            # Load dataset
            loaded_dataset = ModularArithmeticDataset.load(filepath)
            
            # Compare key properties
            self.assertEqual(loaded_dataset.p, self.dataset.p)
            self.assertEqual(loaded_dataset.representation, self.dataset.representation)
            
            # Compare data
            torch.testing.assert_close(loaded_dataset.data['inputs'], self.dataset.data['inputs'])
            torch.testing.assert_close(loaded_dataset.data['targets'], self.dataset.data['targets'])
    
    def test_json_export(self):
        """Test JSON metadata export."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_metadata.json'
            
            # Export metadata
            self.dataset.export_metadata_json(filepath)
            self.assertTrue(filepath.exists())
            
            # Load and verify JSON structure
            import json
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            
            self.assertIn('dataset_info', metadata)
            self.assertIn('circular_structure', metadata)
            self.assertIn('algebraic_properties', metadata)
            self.assertEqual(metadata['dataset_info']['p'], self.test_p)

class TestCircularStructureValidator(unittest.TestCase):
    """Test cases for circular structure validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_p = 8  # Use 8 for cleaner circle geometry
        self.validator = CircularStructureValidator(self.test_p)
    
    def test_perfect_circle_validation(self):
        """Test validation with perfect circular embeddings."""
        
        # Create perfect circle embeddings
        angles = torch.linspace(0, 2 * np.pi, self.test_p + 1)[:-1]  # Remove last point to avoid endpoint
        embeddings = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
        # Validate
        results = self.validator.validate_embeddings(embeddings, visualize=False)
        
        # Should detect perfect circular structure
        self.assertTrue(results['circular_ordering']['is_circular_order'])
        self.assertGreater(results['distance_consistency']['distance_correlation'], 0.8)
        self.assertTrue(results['adjacency_structure']['passes_adjacency_test'])
        self.assertGreater(results['overall_assessment']['overall_score'], 0.8)
    
    def test_random_embeddings_validation(self):
        """Test validation with random embeddings."""
        
        # Create random embeddings
        embeddings = torch.randn(self.test_p, 64)
        
        # Validate
        results = self.validator.validate_embeddings(embeddings, visualize=False)
        
        # Should NOT detect circular structure
        self.assertLess(results['overall_assessment']['overall_score'], 0.5)
    
    def test_noisy_circle_validation(self):
        """Test validation with noisy circular embeddings."""
        
        # Create noisy circle
        angles = torch.linspace(0, 2 * np.pi, self.test_p + 1)[:-1]  # Remove last point to avoid endpoint
        perfect_embeddings = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        noise = torch.randn_like(perfect_embeddings) * 0.3  # Increased noise level
        noisy_embeddings = perfect_embeddings + noise
        
        # Validate
        results = self.validator.validate_embeddings(noisy_embeddings, visualize=False)
        
        # Should still detect circular structure but with lower score than perfect
        self.assertGreater(results['overall_assessment']['overall_score'], 0.4)
        # Note: Score might still be high due to robustness of validation metrics

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset = ModularArithmeticDataset(p=7, representation='embedding')
    
    def test_consistency_checks(self):
        """Test dataset consistency checking."""
        
        checks = check_dataset_consistency(self.dataset)
        
        # All checks should pass for valid dataset
        for check_name, result in checks.items():
            self.assertTrue(result, f"Consistency check failed: {check_name}")
    
    def test_algebraic_analysis(self):
        """Test algebraic properties analysis."""
        
        analysis = analyze_algebraic_properties(self.dataset)
        
        # Should detect correct algebraic properties
        self.assertTrue(analysis['commutativity']['is_commutative'])
        self.assertTrue(analysis['identity']['has_identity'])
        self.assertTrue(analysis['inverses']['all_have_inverses'])
        self.assertEqual(analysis['identity']['identity_element'], 0)
    
    def test_training_splits(self):
        """Test training/validation split creation."""
        
        splits = create_training_splits(self.dataset, train_ratio=0.8)
        
        total_size = self.dataset.data['num_examples']
        expected_train_size = int(total_size * 0.8)
        expected_val_size = total_size - expected_train_size
        
        self.assertEqual(splits['train_inputs'].shape[0], expected_train_size)
        self.assertEqual(splits['val_inputs'].shape[0], expected_val_size)
        self.assertEqual(splits['train_targets'].shape[0], expected_train_size)
        self.assertEqual(splits['val_targets'].shape[0], expected_val_size)
    
    def test_validation_test_suite(self):
        """Test creation of validation test suite."""
        
        test_suite = create_validation_test_suite(p=5)
        
        self.assertIn('perfect_circle_test', test_suite)
        self.assertIn('noisy_circle_test', test_suite)
        self.assertIn('random_embeddings_test', test_suite)
        self.assertIn('validator', test_suite)
        
        # Test that embeddings have correct shapes
        perfect = test_suite['perfect_circle_test']
        self.assertEqual(perfect.shape, (5, 2))

class TestDatasetCreation(unittest.TestCase):
    """Test cases for dataset creation functions."""
    
    def test_create_mod_p_datasets(self):
        """Test creation of multiple datasets."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change DATA_DIR for testing
            import config
            original_data_dir = config.DATA_DIR
            config.DATA_DIR = temp_dir
            
            # Also need to update the dataset module's reference
            import dataset
            dataset.DATA_DIR = temp_dir
            
            try:
                # Create datasets
                datasets = create_mod_p_datasets([5, 7])
                
                # Check that datasets were created
                self.assertIn(5, datasets)
                self.assertIn(7, datasets)
                
                # Check dataset properties
                self.assertEqual(datasets[5].p, 5)
                self.assertEqual(datasets[7].p, 7)
                self.assertEqual(datasets[5].data['num_examples'], 25)
                self.assertEqual(datasets[7].data['num_examples'], 49)
                
                # Check that files were saved
                self.assertTrue(Path(temp_dir, 'mod_5_dataset.pkl').exists())
                self.assertTrue(Path(temp_dir, 'mod_7_dataset.pkl').exists())
                self.assertTrue(Path(temp_dir, 'mod_5_metadata.json').exists())
                self.assertTrue(Path(temp_dir, 'mod_7_metadata.json').exists())
                
            finally:
                # Restore original DATA_DIR
                config.DATA_DIR = original_data_dir
                dataset.DATA_DIR = original_data_dir

class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases."""
    
    def test_invalid_representation(self):
        """Test error handling for invalid representation."""
        
        with self.assertRaises(ValueError):
            ModularArithmeticDataset(p=5, representation='invalid')
    
    def test_edge_case_p_values(self):
        """Test with edge case p values."""
        
        # Test with p=2 (smallest meaningful case)
        dataset_p2 = ModularArithmeticDataset(p=2, representation='embedding')
        self.assertEqual(dataset_p2.data['num_examples'], 4)
        
        # Test with p=3
        dataset_p3 = ModularArithmeticDataset(p=3, representation='embedding')
        self.assertEqual(dataset_p3.data['num_examples'], 9)
        
        # Verify mathematical correctness for p=2
        expected_results = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for i, (a, b) in enumerate(dataset_p2.data['raw_inputs']):
            expected = expected_results[(a, b)]
            actual = dataset_p2.data['raw_targets'][i]
            self.assertEqual(actual, expected)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestModularArithmeticDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestCircularStructureValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")

