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
from AquaML.core.coordinator import AquaMLCoordinator, get_coordinator


class MockEnvironment:
    """Mock environment for testing"""
    def __init__(self, name="TestEnv"):
        self.name = name

class MockAgent:
    """Mock agent for testing"""
    def __init__(self, name="TestAgent"):
        self.name = name

class MockDataUnit:
    """Mock data unit for testing"""
    def __init__(self, name="TestDataUnit"):
        self.name = name
    
    def getUnitStatusDict(self):
        return {"status": "active", "size": 100}

class MockFileSystem:
    """Mock file system for testing"""
    def __init__(self):
        pass
    
    def configRunner(self, runner_name):
        pass
    
    def saveDataUnit(self, runner_name, data_unit_status):
        pass

class MockCommunicator:
    """Mock communicator for testing"""
    def __init__(self, name="TestCommunicator"):
        self.name = name


@pytest.mark.unit
class TestCoordinator(unittest.TestCase):
    """Test cases for AquaML Coordinator"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.coordinator = AquaMLCoordinator()
        # Clear any existing components for clean testing
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_singleton_pattern(self):
        """Test that coordinator follows singleton pattern"""
        coord1 = AquaMLCoordinator()
        coord2 = AquaMLCoordinator()
        self.assertIs(coord1, coord2, "Coordinator should be singleton")
        
    def test_get_coordinator_function(self):
        """Test the get_coordinator function"""
        coord = get_coordinator()
        self.assertIsInstance(coord, AquaMLCoordinator)
        self.assertIs(coord, self.coordinator)


@pytest.mark.unit
class TestModelRegistration(unittest.TestCase):
    """Test model registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()  # Clear existing state
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_model_registration(self):
        """Test model registration and retrieval"""
        dummy_model = torch.nn.Linear(4, 2)
        
        # Test registration
        self.coordinator.registerModel(dummy_model, 'test_model')
        
        # Test retrieval
        retrieved_model_dict = self.coordinator.getModel('test_model')
        self.assertIn('model', retrieved_model_dict)
        self.assertIn('status', retrieved_model_dict)
        self.assertIs(retrieved_model_dict['model'], dummy_model)
        
    def test_duplicate_model_registration(self):
        """Test that duplicate model registration raises error"""
        dummy_model = torch.nn.Linear(4, 2)
        
        # Register model
        self.coordinator.registerModel(dummy_model, 'test_model')
        
        # Try to register again - should raise error
        with self.assertRaises(ValueError):
            self.coordinator.registerModel(dummy_model, 'test_model')
            
    def test_get_nonexistent_model(self):
        """Test getting non-existent model raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getModel('nonexistent_model')


@pytest.mark.unit
class TestEnvironmentRegistration(unittest.TestCase):
    """Test environment registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_environment_registration(self):
        """Test environment registration and retrieval"""
        @self.coordinator.registerEnv
        class TestEnv(MockEnvironment):
            pass
        
        # Create environment instance
        env = TestEnv("CustomTestEnv")
        
        # Test retrieval
        retrieved_env = self.coordinator.getEnv()
        self.assertIs(retrieved_env, env)
        self.assertEqual(retrieved_env.name, "CustomTestEnv")
        
    def test_get_nonexistent_environment(self):
        """Test getting non-existent environment raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getEnv()


@pytest.mark.unit
class TestAgentRegistration(unittest.TestCase):
    """Test agent registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_agent_registration(self):
        """Test agent registration and retrieval"""
        @self.coordinator.registerAgent
        class TestAgent(MockAgent):
            pass
        
        # Create agent instance
        agent = TestAgent("CustomTestAgent")
        
        # Test retrieval
        retrieved_agent = self.coordinator.getAgent()
        self.assertIs(retrieved_agent, agent)
        self.assertEqual(retrieved_agent.name, "CustomTestAgent")
        
    def test_duplicate_agent_registration(self):
        """Test that duplicate agent registration raises error"""
        @self.coordinator.registerAgent
        class TestAgent1(MockAgent):
            pass
        
        agent1 = TestAgent1("Agent1")
        
        # Try to register another agent - should raise error
        with self.assertRaises(ValueError):
            @self.coordinator.registerAgent
            class TestAgent2(MockAgent):
                pass
            TestAgent2("Agent2")
            
    def test_get_nonexistent_agent(self):
        """Test getting non-existent agent raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getAgent()


@pytest.mark.unit
class TestDataUnitRegistration(unittest.TestCase):
    """Test data unit registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_data_unit_registration(self):
        """Test data unit registration and retrieval"""
        @self.coordinator.registerDataUnit
        class TestDataUnit(MockDataUnit):
            pass
        
        # Create data unit instance
        data_unit = TestDataUnit("CustomDataUnit")
        
        # Test retrieval
        retrieved_data_unit = self.coordinator.getDataUnit("CustomDataUnit")
        self.assertIs(retrieved_data_unit, data_unit)
        self.assertEqual(retrieved_data_unit.name, "CustomDataUnit")
        
    def test_get_nonexistent_data_unit(self):
        """Test getting non-existent data unit raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getDataUnit("nonexistent_unit")


@pytest.mark.unit
class TestFileSystemRegistration(unittest.TestCase):
    """Test file system registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_file_system_registration(self):
        """Test file system registration and retrieval"""
        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass
        
        # Create file system instance
        fs = TestFileSystem()
        
        # Test retrieval
        retrieved_fs = self.coordinator.getFileSystem()
        self.assertIs(retrieved_fs, fs)
        
    def test_duplicate_file_system_registration(self):
        """Test that duplicate file system registration raises error"""
        @self.coordinator.registerFileSystem
        class TestFileSystem1(MockFileSystem):
            pass
        
        fs1 = TestFileSystem1()
        
        # Try to register another file system - should raise error
        with self.assertRaises(ValueError):
            @self.coordinator.registerFileSystem
            class TestFileSystem2(MockFileSystem):
                pass
            TestFileSystem2()
            
    def test_get_nonexistent_file_system(self):
        """Test getting non-existent file system raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getFileSystem()


@pytest.mark.unit
class TestCommunicatorRegistration(unittest.TestCase):
    """Test communicator registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_communicator_registration(self):
        """Test communicator registration and retrieval"""
        @self.coordinator.registerCommunicator
        class TestCommunicator(MockCommunicator):
            pass
        
        # Create communicator instance
        comm = TestCommunicator("CustomCommunicator")
        
        # Test retrieval
        retrieved_comm = self.coordinator.getCommunicator()
        self.assertIs(retrieved_comm, comm)
        self.assertEqual(retrieved_comm.name, "CustomCommunicator")
        
    def test_duplicate_communicator_registration(self):
        """Test that duplicate communicator registration raises error"""
        @self.coordinator.registerCommunicator
        class TestCommunicator1(MockCommunicator):
            pass
        
        comm1 = TestCommunicator1("Comm1")
        
        # Try to register another communicator - should raise error
        with self.assertRaises(ValueError):
            @self.coordinator.registerCommunicator
            class TestCommunicator2(MockCommunicator):
                pass
            TestCommunicator2("Comm2")
            
    def test_get_nonexistent_communicator(self):
        """Test getting non-existent communicator raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getCommunicator()


@pytest.mark.unit
class TestRunnerRegistration(unittest.TestCase):
    """Test runner registration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_runner_registration(self):
        """Test runner registration and retrieval"""
        # First register a file system
        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass
        fs = TestFileSystem()
        
        # Register runner
        self.coordinator.registerRunner("test_runner")
        
        # Test retrieval
        retrieved_runner = self.coordinator.getRunner()
        self.assertEqual(retrieved_runner, "test_runner")
        
    def test_runner_registration_without_file_system(self):
        """Test runner registration without file system (should warn but work)"""
        # Register runner without file system
        self.coordinator.registerRunner("test_runner")
        
        # Should still work
        retrieved_runner = self.coordinator.getRunner()
        self.assertEqual(retrieved_runner, "test_runner")
        
    def test_get_nonexistent_runner(self):
        """Test getting non-existent runner raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.getRunner()


@pytest.mark.unit
class TestDataManagement(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_save_data_unit_info(self):
        """Test saving data unit info"""
        # Register components needed for save operation
        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass
        fs = TestFileSystem()
        
        @self.coordinator.registerDataUnit
        class TestDataUnit(MockDataUnit):
            pass
        data_unit = TestDataUnit("test_unit")
        
        self.coordinator.registerRunner("test_runner")
        
        # This should not raise an exception
        self.coordinator.saveDataUnitInfo()
        
    def test_save_data_unit_info_without_runner(self):
        """Test saving data unit info without runner raises error"""
        with self.assertRaises(ValueError):
            self.coordinator.saveDataUnitInfo()


@pytest.mark.unit
class TestCoordinatorStatus(unittest.TestCase):
    """Test coordinator status and reporting"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        self.coordinator.shutdown()
        
    def tearDown(self):
        """Clean up after each test"""
        self.coordinator.shutdown()
        
    def test_status_reporting(self):
        """Test coordinator status reporting"""
        status = self.coordinator.get_status()
        
        # Check required status fields
        self.assertIn('initialized', status)
        self.assertIn('components', status)
        self.assertIn('device_info', status)
        self.assertIn('runner_name', status)
        
        # Check types
        self.assertIsInstance(status['initialized'], bool)
        self.assertIsInstance(status['components'], dict)
        self.assertIsInstance(status['device_info'], dict)
        
    def test_component_listing(self):
        """Test component listing functionality"""
        components = self.coordinator.list_components()
        self.assertIsInstance(components, dict)
        
        # Check expected component categories
        expected_categories = [
            'models', 'data_units', 'environment', 'agent', 
            'file_system', 'communicator', 'data_manager', 'runner'
        ]
        for category in expected_categories:
            self.assertIn(category, components)
            self.assertIsInstance(components[category], int)
        
    def test_component_counting(self):
        """Test that component counting works correctly"""
        # Register some components
        dummy_model = torch.nn.Linear(2, 1)
        self.coordinator.registerModel(dummy_model, 'test_model')
        
        @self.coordinator.registerDataUnit
        class TestDataUnit(MockDataUnit):
            pass
        data_unit = TestDataUnit("test_unit")
        
        # Check counts
        components = self.coordinator.list_components()
        self.assertEqual(components['models'], 1)
        self.assertEqual(components['data_units'], 1)
        self.assertEqual(components['environment'], 0)
        self.assertEqual(components['agent'], 0)


@pytest.mark.integration
class TestDeviceManagement(unittest.TestCase):
    """Test device management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = get_coordinator()
        
    def test_device_management(self):
        """Test basic device management"""
        # Test getting device
        device = self.coordinator.get_device()
        self.assertIsInstance(device, str)
        
        # Test torch device
        torch_device = self.coordinator.get_torch_device()
        self.assertIsInstance(torch_device, torch.device)
        
        # Test available devices
        available = self.coordinator.get_available_devices()
        self.assertIsInstance(available, list)
        self.assertIn('cpu', available)
        
        # Test device validation
        self.assertTrue(self.coordinator.validate_device('cpu'))
        
        # Test GPU availability check
        gpu_available = self.coordinator.is_gpu_available()
        self.assertIsInstance(gpu_available, bool)
        
        # Test device info
        device_info = self.coordinator.get_device_info()
        self.assertIsInstance(device_info, dict)
        self.assertIn('current_device', device_info)
        self.assertIn('available_devices', device_info)
        self.assertIn('gpu_available', device_info)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2) 