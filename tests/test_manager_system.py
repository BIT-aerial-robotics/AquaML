#!/usr/bin/env python3
"""
Test suite for AquaML Manager System

This module contains comprehensive tests for the new AquaML manager system architecture.
Tests cover all specialized managers and their interactions with the coordinator.
"""

import unittest
import torch
import torch.nn as nn
from typing import Any
import pytest

# Import AquaML components
from AquaML.core.coordinator import AquaMLCoordinator, get_coordinator


class MockModel(nn.Module):
    """Mock PyTorch model for testing"""

    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class MockEnvironment:
    """Mock environment for testing"""

    def __init__(self, name="TestEnv"):
        self.name = name
        self.state = "initialized"

    def reset(self):
        self.state = "reset"
        return [0.0, 0.0, 0.0]


class MockAgent:
    """Mock agent for testing"""

    def __init__(self, name="TestAgent"):
        self.name = name
        self.actions = []

    def act(self, state):
        action = "test_action"
        self.actions.append(action)
        return action


class MockDataUnit:
    """Mock data unit for testing"""

    def __init__(self, name="TestDataUnit"):
        self.name = name
        self.data = []

    def add_data(self, item):
        self.data.append(item)

    def getUnitStatusDict(self):
        return {"status": "active", "size": len(self.data)}


class MockFileSystem:
    """Mock file system for testing"""

    def __init__(self):
        self.configured_runners = []
        self.saved_data = {}

    def configRunner(self, runner_name):
        self.configured_runners.append(runner_name)

    def saveDataUnit(self, runner_name, data_unit_status):
        self.saved_data[runner_name] = data_unit_status


class MockCommunicator:
    """Mock communicator for testing"""

    def __init__(self, name="TestCommunicator"):
        self.name = name
        self.messages = []

    def send_message(self, message):
        self.messages.append(message)


class MockDataManager:
    """Mock data manager for testing"""

    def __init__(self):
        self.datasets = {}

    def load_dataset(self, name):
        return self.datasets.get(name, [])


@pytest.mark.unit
class TestModelManager(unittest.TestCase):
    """Test model manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()
        self.model_manager = self.coordinator.get_model_manager()

    def tearDown(self):
        """Clean up after tests"""
        self.model_manager.clear_models()

    def test_model_registration(self):
        """Test model registration functionality"""
        model = MockModel()
        self.coordinator.registerModel(model, "test_model")

        # Test model exists
        self.assertTrue(self.model_manager.model_exists("test_model"))

        # Test model retrieval
        retrieved_model = self.model_manager.get_model("test_model")
        self.assertIn("model", retrieved_model)
        self.assertIn("status", retrieved_model)
        self.assertEqual(retrieved_model["model"], model)

    def test_model_manager_status(self):
        """Test model manager status reporting"""
        model1 = MockModel()
        model2 = MockModel(input_size=20, output_size=5)

        self.coordinator.registerModel(model1, "model1")
        self.coordinator.registerModel(model2, "model2")

        status = self.model_manager.get_status()
        self.assertEqual(status["total_models"], 2)
        self.assertIn("model1", status["model_names"])
        self.assertIn("model2", status["model_names"])

    def test_model_removal(self):
        """Test model removal functionality"""
        model = MockModel()
        self.coordinator.registerModel(model, "test_model")

        # Verify model exists
        self.assertTrue(self.model_manager.model_exists("test_model"))

        # Remove model
        self.model_manager.remove_model("test_model")

        # Verify model no longer exists
        self.assertFalse(self.model_manager.model_exists("test_model"))

    def test_duplicate_model_registration(self):
        """Test error handling for duplicate model registration"""
        model1 = MockModel()
        model2 = MockModel()

        self.coordinator.registerModel(model1, "test_model")

        with self.assertRaises(ValueError):
            self.coordinator.registerModel(model2, "test_model")


@pytest.mark.unit
class TestEnvironmentManager(unittest.TestCase):
    """Test environment manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()
        self.env_manager = self.coordinator.get_environment_manager()

    def tearDown(self):
        """Clean up after tests"""
        self.env_manager.remove_env()

    def test_environment_registration(self):
        """Test environment registration functionality"""

        @self.coordinator.registerEnv
        class TestEnv(MockEnvironment):
            pass

        env = TestEnv()

        # Test environment exists
        self.assertTrue(self.env_manager.env_exists())

        # Test environment retrieval
        retrieved_env = self.coordinator.getEnv()
        self.assertEqual(retrieved_env.name, "TestEnv")

    def test_environment_manager_status(self):
        """Test environment manager status reporting"""

        @self.coordinator.registerEnv
        class TestEnv(MockEnvironment):
            pass

        env = TestEnv()

        status = self.env_manager.get_status()
        self.assertTrue(status["env_registered"])
        self.assertTrue(status["env_info"]["exists"])
        self.assertEqual(status["env_info"]["name"], "TestEnv")


@pytest.mark.unit
class TestAgentManager(unittest.TestCase):
    """Test agent manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()
        self.agent_manager = self.coordinator.get_agent_manager()

    def tearDown(self):
        """Clean up after tests"""
        self.agent_manager.remove_agent()

    def test_agent_registration(self):
        """Test agent registration functionality"""

        @self.coordinator.registerAgent
        class TestAgent(MockAgent):
            pass

        agent = TestAgent()

        # Test agent exists
        self.assertTrue(self.agent_manager.agent_exists())

        # Test agent retrieval
        retrieved_agent = self.coordinator.getAgent()
        self.assertEqual(retrieved_agent.name, "TestAgent")

    def test_agent_manager_status(self):
        """Test agent manager status reporting"""

        @self.coordinator.registerAgent
        class TestAgent(MockAgent):
            pass

        agent = TestAgent()

        status = self.agent_manager.get_status()
        self.assertTrue(status["agent_registered"])
        self.assertTrue(status["agent_info"]["exists"])
        self.assertEqual(status["agent_info"]["name"], "TestAgent")


@pytest.mark.unit
class TestDataUnitManager(unittest.TestCase):
    """Test data unit manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()
        self.data_unit_manager = self.coordinator.get_data_unit_manager()

    def tearDown(self):
        """Clean up after tests"""
        self.data_unit_manager.clear_data_units()

    def test_data_unit_registration(self):
        """Test data unit registration functionality"""

        @self.coordinator.registerDataUnit
        class TestDataUnit(MockDataUnit):
            pass

        data_unit = TestDataUnit()
        data_unit.add_data("test_data")

        # Test data unit exists
        self.assertTrue(self.data_unit_manager.data_unit_exists("TestDataUnit"))

        # Test data unit retrieval
        retrieved_unit = self.coordinator.getDataUnit("TestDataUnit")
        self.assertEqual(retrieved_unit.name, "TestDataUnit")
        self.assertIn("test_data", retrieved_unit.data)

    def test_data_unit_manager_status(self):
        """Test data unit manager status reporting"""

        @self.coordinator.registerDataUnit
        class TestDataUnit1(MockDataUnit):
            def __init__(self):
                super().__init__(name="TestDataUnit1")

        @self.coordinator.registerDataUnit
        class TestDataUnit2(MockDataUnit):
            def __init__(self):
                super().__init__(name="TestDataUnit2")

        data_unit1 = TestDataUnit1()
        data_unit2 = TestDataUnit2()

        status = self.data_unit_manager.get_status()
        self.assertEqual(status["total_data_units"], 2)
        self.assertIn("TestDataUnit1", status["data_unit_names"])
        self.assertIn("TestDataUnit2", status["data_unit_names"])


@pytest.mark.unit
class TestFileSystemManager(unittest.TestCase):
    """Test file system manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()
        self.fs_manager = self.coordinator.get_file_system_manager()

    def tearDown(self):
        """Clean up after tests"""
        self.fs_manager.remove_file_system()

    def test_file_system_registration(self):
        """Test file system registration functionality"""

        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass

        fs = TestFileSystem()

        # Test file system exists
        self.assertTrue(self.fs_manager.file_system_exists())

        # Test file system retrieval
        retrieved_fs = self.coordinator.getFileSystem()
        self.assertIsInstance(retrieved_fs, MockFileSystem)

    def test_runner_configuration(self):
        """Test runner configuration in file system"""

        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass

        fs = TestFileSystem()

        # Register runner
        self.coordinator.registerRunner("test_runner")

        # Check if runner was configured
        retrieved_fs = self.coordinator.getFileSystem()
        self.assertIn("test_runner", retrieved_fs.configured_runners)


@pytest.mark.unit
class TestRunnerManager(unittest.TestCase):
    """Test runner manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()
        self.runner_manager = self.coordinator.get_runner_manager()

    def tearDown(self):
        """Clean up after tests"""
        self.runner_manager.remove_runner()

    def test_runner_registration(self):
        """Test runner registration functionality"""
        self.coordinator.registerRunner("test_runner")

        # Test runner exists
        self.assertTrue(self.runner_manager.runner_exists())

        # Test runner retrieval
        runner_name = self.coordinator.getRunner()
        self.assertEqual(runner_name, "test_runner")

    def test_runner_manager_status(self):
        """Test runner manager status reporting"""
        self.coordinator.registerRunner("test_runner")

        status = self.runner_manager.get_status()
        self.assertTrue(status["runner_registered"])
        self.assertEqual(status["runner_info"]["name"], "test_runner")


@pytest.mark.integration
class TestManagerIntegration(unittest.TestCase):
    """Test integration between different managers"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()

    def tearDown(self):
        """Clean up after tests"""
        self.coordinator.shutdown()

    def test_full_system_integration(self):
        """Test full system integration with all managers"""
        # Register model
        model = MockModel()
        self.coordinator.registerModel(model, "policy_model")

        # Register environment
        @self.coordinator.registerEnv
        class TestEnv(MockEnvironment):
            pass

        env = TestEnv()

        # Register agent
        @self.coordinator.registerAgent
        class TestAgent(MockAgent):
            pass

        agent = TestAgent()

        # Register data unit
        @self.coordinator.registerDataUnit
        class TestDataUnit(MockDataUnit):
            pass

        data_unit = TestDataUnit()

        # Register file system
        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass

        fs = TestFileSystem()

        # Register runner
        self.coordinator.registerRunner("test_runner")

        # Test component listing
        components = self.coordinator.list_components()
        self.assertEqual(components["models"], 1)
        self.assertEqual(components["environment"], 1)
        self.assertEqual(components["agent"], 1)
        self.assertEqual(components["data_units"], 1)
        self.assertEqual(components["file_system"], 1)
        self.assertEqual(components["runner"], 1)

        # Test all managers status
        all_status = self.coordinator.get_all_managers_status()
        self.assertIn("model_manager", all_status)
        self.assertIn("environment_manager", all_status)
        self.assertIn("agent_manager", all_status)
        self.assertIn("data_unit_manager", all_status)
        self.assertIn("file_system_manager", all_status)
        self.assertIn("runner_manager", all_status)

    def test_data_unit_info_saving(self):
        """Test data unit information saving functionality"""

        # Register file system
        @self.coordinator.registerFileSystem
        class TestFileSystem(MockFileSystem):
            pass

        fs = TestFileSystem()

        # Register data unit
        @self.coordinator.registerDataUnit
        class TestDataUnit(MockDataUnit):
            pass

        data_unit = TestDataUnit()
        data_unit.add_data("sample_data")

        # Register runner
        self.coordinator.registerRunner("test_runner")

        # Save data unit info
        self.coordinator.saveDataUnitInfo()

        # Verify data was saved
        retrieved_fs = self.coordinator.getFileSystem()
        self.assertIn("test_runner", retrieved_fs.saved_data)
        self.assertIn("TestDataUnit", retrieved_fs.saved_data["test_runner"])


@pytest.mark.unit
class TestErrorHandling(unittest.TestCase):
    """Test error handling across all managers"""

    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = AquaMLCoordinator()

    def test_nonexistent_model_access(self):
        """Test error handling for accessing nonexistent models"""
        with self.assertRaises(ValueError):
            self.coordinator.getModel("nonexistent_model")

    def test_nonexistent_environment_access(self):
        """Test error handling for accessing nonexistent environment"""
        with self.assertRaises(ValueError):
            self.coordinator.getEnv()

    def test_nonexistent_agent_access(self):
        """Test error handling for accessing nonexistent agent"""
        with self.assertRaises(ValueError):
            self.coordinator.getAgent()

    def test_nonexistent_data_unit_access(self):
        """Test error handling for accessing nonexistent data unit"""
        with self.assertRaises(ValueError):
            self.coordinator.getDataUnit("nonexistent_unit")

    def test_nonexistent_runner_access(self):
        """Test error handling for accessing nonexistent runner"""
        with self.assertRaises(ValueError):
            self.coordinator.getRunner()


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
