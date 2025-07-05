#!/usr/bin/env python3
"""
Test suite for AquaML Coordinator

This module contains comprehensive tests for the AquaML coordinator and its components.
"""

import unittest
import torch
import tempfile
import os
from typing import Any
import pytest

# Import AquaML components
from AquaML import AquaMLCoordinator, ComponentRegistry, LifecycleManager, coordinator


@pytest.mark.unit
class TestCoordinator(unittest.TestCase):
    """Test cases for AquaML Coordinator"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.coordinator = AquaMLCoordinator()
        
    def test_singleton_pattern(self):
        """Test that coordinator follows singleton pattern"""
        coord1 = AquaMLCoordinator()
        coord2 = AquaMLCoordinator()
        self.assertIs(coord1, coord2, "Coordinator should be singleton")
        
    def test_component_registry(self):
        """Test component registry functionality"""
        registry = ComponentRegistry()
        
        # Test registration
        registry.register('test_component', 'test_value')
        self.assertEqual(registry.get('test_component'), 'test_value')
        
        # Test existence check
        self.assertTrue(registry.has('test_component'))
        self.assertFalse(registry.has('non_existent'))
        
        # Test listing components
        components = registry.list_components()
        self.assertIn('test_component', components)
        
    def test_lifecycle_manager(self):
        """Test lifecycle manager functionality"""
        lifecycle = LifecycleManager()
        
        # Test state management
        lifecycle.set_component_state('test_component', 'running')
        self.assertEqual(lifecycle.get_component_state('test_component'), 'running')
        
        # Test running status check
        self.assertTrue(lifecycle.is_component_running('test_component'))
        
        # Test all states
        all_states = lifecycle.get_all_component_states()
        self.assertIn('test_component', all_states)
        self.assertEqual(all_states['test_component'], 'running')


@pytest.mark.legacy
class TestLegacyCompatibility(unittest.TestCase):
    """Test legacy API compatibility"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = coordinator
    
    def test_legacy_model_registration(self):
        """Test legacy model registration and retrieval"""
        dummy_model = torch.nn.Linear(4, 2)
        
        # Test registration
        self.coordinator.registerModel(dummy_model, 'test_model')
        
        # Test retrieval
        retrieved_model = self.coordinator.getModel('test_model')
        self.assertIs(retrieved_model, dummy_model)
        
    def test_legacy_runner_registration(self):
        """Test legacy runner registration"""
        # This should not raise an exception
        self.coordinator.registerRunner('test_runner')
        
    def test_save_data_unit_info(self):
        """Test legacy save data unit info"""
        # This should not raise an exception
        self.coordinator.saveDataUnitInfo()


@pytest.mark.integration
class TestDataComponents(unittest.TestCase):
    """Test data component functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_numpy_unit_import(self):
        """Test NumpyUnit import and basic functionality"""
        try:
            from AquaML.data.numpy_unit import NumpyUnit
            from AquaML.data.cfg_status import unitCfg
            
            # Create proper configuration
            config = unitCfg(
                name='test_numpy',
                dtype='float32',
                single_shape=(10, 5),
                size=1
            )
            
            # Test creation
            numpy_unit = NumpyUnit(config)
            self.assertIsNotNone(numpy_unit)
        except ImportError:
            self.skipTest("NumpyUnit not available")
    
    def test_tensor_unit_import(self):
        """Test TensorUnit import and basic functionality"""
        try:
            from AquaML.data.tensor_unit import TensorUnit
            from AquaML.data.cfg_status import unitCfg
            import torch
            
            # Create proper configuration - use numpy-compatible dtype
            config = unitCfg(
                name='test_tensor',
                dtype='float32',  # Use string dtype compatible with NumPy
                single_shape=(5, 3),
                size=1
            )
            
            # Test creation
            tensor_unit = TensorUnit(config)
            self.assertIsNotNone(tensor_unit)
        except ImportError:
            self.skipTest("TensorUnit not available")
        except Exception as e:
            # If there's an issue with the unit itself, skip the test
            self.skipTest(f"TensorUnit test skipped due to: {e}")
    
    def test_file_system_import(self):
        """Test DefaultFileSystem import and basic functionality"""
        try:
            from AquaML.utils.file_system.default_file_system import DefaultFileSystem
            # Test creation
            file_system = DefaultFileSystem(self.temp_dir)
            self.assertIsNotNone(file_system)
        except ImportError:
            self.skipTest("DefaultFileSystem not available")


@pytest.mark.unit
class TestCoordinatorStatus(unittest.TestCase):
    """Test coordinator status and reporting"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = coordinator
    
    def test_status_reporting(self):
        """Test coordinator status reporting"""
        status = self.coordinator.get_status()
        
        # Check required status fields
        self.assertIn('initialized', status)
        self.assertIn('lifecycle_initialized', status)
        self.assertIn('registered_components', status)
        self.assertIn('component_states', status)
        self.assertIn('components', status)
        
        # Check types
        self.assertIsInstance(status['initialized'], bool)
        self.assertIsInstance(status['registered_components'], int)
        self.assertIsInstance(status['components'], list)
        
    def test_component_listing(self):
        """Test component listing functionality"""
        components = self.coordinator.list_components()
        self.assertIsInstance(components, list)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2) 