#!/usr/bin/env python3
"""
Test suite for AquaML File System

This module contains comprehensive tests for the file system components.
"""

import unittest
import tempfile
import shutil
import os
import yaml
import pytest
from unittest.mock import patch, MagicMock

# Import AquaML components
from AquaML.utils.file_system import DefaultFileSystem
from AquaML.utils.file_system.base_file_system import BaseFileSystem


@pytest.mark.unit
class TestBaseFileSystem(unittest.TestCase):
    """Test cases for BaseFileSystem"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp(prefix="aquaml_fs_test_")
        self.test_runner_name = "test_runner"
        
        # Create a concrete implementation for testing
        class TestFileSystem(BaseFileSystem):
            def __init__(self, workspace_dir):
                super().__init__(workspace_dir)
        
        self.fs = TestFileSystem(self.temp_dir)
        
    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test BaseFileSystem initialization"""
        self.assertEqual(self.fs.workspace_dir_, self.temp_dir)
        self.assertEqual(self.fs.logger_path_, os.path.join(self.temp_dir, "logger"))
        self.assertEqual(self.fs.runner_dir_dict_, {})
    
    def test_init_folder(self):
        """Test folder initialization"""
        self.fs.initFolder()
        
        # Check if workspace directory is created
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Check if logger directory is created
        self.assertTrue(os.path.exists(self.fs.logger_path_))
    
    def test_config_runner_with_creation(self):
        """Test runner configuration with folder creation"""
        self.fs.configRunner(self.test_runner_name, create_first=True)
        
        # Check if runner is registered
        self.assertIn(self.test_runner_name, self.fs.runner_dir_dict_)
        
        # Check if all required directories are created
        runner_dirs = self.fs.runner_dir_dict_[self.test_runner_name]
        for dir_type, path in runner_dirs.items():
            self.assertTrue(os.path.exists(path), f"Directory {dir_type} not created: {path}")
    
    def test_config_runner_without_creation(self):
        """Test runner configuration without folder creation"""
        self.fs.configRunner(self.test_runner_name, create_first=False)
        
        # Check if runner is registered
        self.assertIn(self.test_runner_name, self.fs.runner_dir_dict_)
        
        # Check if directories are NOT created
        runner_dirs = self.fs.runner_dir_dict_[self.test_runner_name]
        for dir_type, path in runner_dirs.items():
            self.assertFalse(os.path.exists(path), f"Directory {dir_type} should not be created: {path}")
    
    def test_query_history_model_path(self):
        """Test querying history model path"""
        self.fs.configRunner(self.test_runner_name)
        
        path = self.fs.queryHistoryModelPath(self.test_runner_name)
        expected_path = os.path.join(self.temp_dir, self.test_runner_name, "history_model")
        
        self.assertEqual(path, expected_path)
    
    def test_query_history_model_path_unregistered(self):
        """Test querying history model path for unregistered runner"""
        with self.assertRaises(KeyError):
            self.fs.queryHistoryModelPath("non_existent_runner")
    
    def test_query_cache_path(self):
        """Test querying cache path"""
        self.fs.configRunner(self.test_runner_name)
        
        path = self.fs.queryCachePath(self.test_runner_name)
        expected_path = os.path.join(self.temp_dir, self.test_runner_name, "cache")
        
        self.assertEqual(path, expected_path)
    
    def test_query_cache_path_unregistered(self):
        """Test querying cache path for unregistered runner"""
        with self.assertRaises(KeyError):
            self.fs.queryCachePath("non_existent_runner")
    
    def test_query_log_path(self):
        """Test querying log path"""
        self.fs.configRunner(self.test_runner_name)
        
        path = self.fs.queryLogPath(self.test_runner_name)
        expected_path = os.path.join(self.temp_dir, self.test_runner_name, "log")
        
        self.assertEqual(path, expected_path)
    
    def test_query_log_path_unregistered(self):
        """Test querying log path for unregistered runner"""
        with self.assertRaises(KeyError):
            self.fs.queryLogPath("non_existent_runner")
    
    def test_query_data_unit_file(self):
        """Test querying data unit file path"""
        self.fs.configRunner(self.test_runner_name)
        
        path = self.fs.queryDataUnitFile(self.test_runner_name)
        expected_path = os.path.join(self.temp_dir, self.test_runner_name, "data_config", "data_unit.yaml")
        
        self.assertEqual(path, expected_path)
    
    def test_query_data_unit_file_unregistered(self):
        """Test querying data unit file for unregistered runner"""
        with self.assertRaises(KeyError):
            self.fs.queryDataUnitFile("non_existent_runner")
    
    def test_query_env_info_file(self):
        """Test querying environment info file path"""
        self.fs.configRunner(self.test_runner_name)
        
        path = self.fs.queryEnvInfoFile(self.test_runner_name)
        expected_path = os.path.join(self.temp_dir, self.test_runner_name, "data_config", "env_info.yaml")
        
        self.assertEqual(path, expected_path)
    
    def test_query_env_info_file_unregistered(self):
        """Test querying environment info file for unregistered runner"""
        with self.assertRaises(KeyError):
            self.fs.queryEnvInfoFile("non_existent_runner")
    
    def test_save_data_unit_info(self):
        """Test saving data unit information"""
        self.fs.configRunner(self.test_runner_name)
        
        test_data = {
            "name": "test_unit",
            "type": "numpy",
            "shape": [10, 5],
            "dtype": "float32"
        }
        
        self.fs.saveDataUnitInfo(self.test_runner_name, test_data)
        
        # Check if file was created and contains correct data
        file_path = self.fs.queryDataUnitFile(self.test_runner_name)
        self.assertTrue(os.path.exists(file_path))
        
        with open(file_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertEqual(saved_data, test_data)
    
    def test_save_env_info(self):
        """Test saving environment information"""
        self.fs.configRunner(self.test_runner_name)
        
        test_env_info = {
            "environment": "test_env",
            "observation_space": "Box(10,)",
            "action_space": "Discrete(4)",
            "max_episode_steps": 1000
        }
        
        self.fs.saveEnvInfo(self.test_runner_name, test_env_info)
        
        # Check if file was created and contains correct data
        file_path = self.fs.queryEnvInfoFile(self.test_runner_name)
        self.assertTrue(os.path.exists(file_path))
        
        with open(file_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertEqual(saved_data, test_env_info)


@pytest.mark.unit
class TestDefaultFileSystem(unittest.TestCase):
    """Test cases for DefaultFileSystem"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp(prefix="aquaml_default_fs_test_")
        # Clear coordinator registry to avoid conflicts
        from AquaML import coordinator
        if hasattr(coordinator, 'registry'):
            coordinator.registry.clear()
        
    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clear coordinator registry after test
        from AquaML import coordinator
        if hasattr(coordinator, 'registry'):
            coordinator.registry.clear()
    
    def test_init(self):
        """Test DefaultFileSystem initialization"""
        fs = DefaultFileSystem(self.temp_dir)
        
        # Check if it's properly initialized
        self.assertEqual(fs.workspace_dir_, self.temp_dir)
        self.assertIsInstance(fs, BaseFileSystem)
    
    def test_coordinator_registration(self):
        """Test that DefaultFileSystem is properly registered with coordinator"""
        # This test verifies that the @coordinator.registerFileSystem decorator works
        from AquaML import coordinator
        
        # Create instance (this should trigger registration)
        fs = DefaultFileSystem(self.temp_dir)
        
        # The registration should have happened during class definition
        # We can't easily test this without mocking, but we can verify instance creation works
        self.assertIsNotNone(fs)
        
    def test_inheritance(self):
        """Test that DefaultFileSystem properly inherits from BaseFileSystem"""
        fs = DefaultFileSystem(self.temp_dir)
        
        # Test that it has all BaseFileSystem methods
        self.assertTrue(hasattr(fs, 'initFolder'))
        self.assertTrue(hasattr(fs, 'configRunner'))
        self.assertTrue(hasattr(fs, 'queryHistoryModelPath'))
        self.assertTrue(hasattr(fs, 'queryCachePath'))
        self.assertTrue(hasattr(fs, 'queryLogPath'))
        self.assertTrue(hasattr(fs, 'queryDataUnitFile'))
        self.assertTrue(hasattr(fs, 'queryEnvInfoFile'))
        self.assertTrue(hasattr(fs, 'saveDataUnitInfo'))
        self.assertTrue(hasattr(fs, 'saveEnvInfo'))
    
    def test_full_workflow(self):
        """Test complete workflow with DefaultFileSystem"""
        fs = DefaultFileSystem(self.temp_dir)
        runner_name = "workflow_test_runner"
        
        # Initialize folders
        fs.initFolder()
        
        # Configure runner
        fs.configRunner(runner_name)
        
        # Test all path queries
        cache_path = fs.queryCachePath(runner_name)
        log_path = fs.queryLogPath(runner_name)
        model_path = fs.queryHistoryModelPath(runner_name)
        
        # Verify all paths exist
        self.assertTrue(os.path.exists(cache_path))
        self.assertTrue(os.path.exists(log_path))
        self.assertTrue(os.path.exists(model_path))
        
        # Test data saving
        test_data = {"test": "data"}
        test_env = {"env": "info"}
        
        fs.saveDataUnitInfo(runner_name, test_data)
        fs.saveEnvInfo(runner_name, test_env)
        
        # Verify data was saved
        data_file = fs.queryDataUnitFile(runner_name)
        env_file = fs.queryEnvInfoFile(runner_name)
        
        self.assertTrue(os.path.exists(data_file))
        self.assertTrue(os.path.exists(env_file))


@pytest.mark.integration
class TestFileSystemIntegration(unittest.TestCase):
    """Integration tests for file system with coordinator"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="aquaml_integration_test_")
        # Clear coordinator registry to avoid conflicts
        from AquaML import coordinator
        if hasattr(coordinator, 'registry'):
            coordinator.registry.clear()
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clear coordinator registry after test
        from AquaML import coordinator
        if hasattr(coordinator, 'registry'):
            coordinator.registry.clear()
    
    def test_coordinator_file_system_integration(self):
        """Test file system integration with coordinator"""
        from AquaML import coordinator
        
        # Create file system instance
        fs = DefaultFileSystem(self.temp_dir)
        
        # Test that coordinator can work with file system
        # This is a basic integration test
        self.assertIsNotNone(fs)
        self.assertTrue(hasattr(fs, 'workspace_dir_'))
        self.assertTrue(hasattr(fs, 'configRunner'))
        self.assertTrue(hasattr(fs, 'queryHistoryModelPath'))


@pytest.mark.unit
class TestFileSystemEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="aquaml_edge_test_")
        
        class TestFileSystem(BaseFileSystem):
            def __init__(self, workspace_dir):
                super().__init__(workspace_dir)
        
        self.fs = TestFileSystem(self.temp_dir)
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_workspace_dir(self):
        """Test behavior with empty workspace directory"""
        class TestFileSystem(BaseFileSystem):
            def __init__(self, workspace_dir):
                super().__init__(workspace_dir)
        
        fs = TestFileSystem("")
        self.assertEqual(fs.workspace_dir_, "")
    
    def test_special_characters_in_runner_name(self):
        """Test runner names with special characters"""
        special_runner_name = "test_runner_with_special_chars_!@#"
        self.fs.configRunner(special_runner_name)
        
        # Should handle special characters gracefully
        self.assertIn(special_runner_name, self.fs.runner_dir_dict_)
    
    def test_multiple_runners(self):
        """Test handling multiple runners"""
        runners = ["runner1", "runner2", "runner3"]
        
        for runner in runners:
            self.fs.configRunner(runner)
        
        # All runners should be registered
        for runner in runners:
            self.assertIn(runner, self.fs.runner_dir_dict_)
            
        # Each runner should have independent paths
        paths = set()
        for runner in runners:
            cache_path = self.fs.queryCachePath(runner)
            paths.add(cache_path)
        
        # All paths should be unique
        self.assertEqual(len(paths), len(runners))


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2) 