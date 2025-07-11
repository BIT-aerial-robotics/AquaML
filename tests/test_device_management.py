#!/usr/bin/env python3
"""
Test suite for AquaML Device Management

This module contains comprehensive tests for the AquaML device management functionality.
"""

import unittest
import torch
import pytest
from unittest.mock import patch, MagicMock

# Import AquaML components
from AquaML import coordinator
from AquaML.core.device_info import GPUInfo, detect_gpu_devices, get_optimal_device
from AquaML.core.coordinator import global_device, available_devices


@pytest.mark.unit
class TestGPUInfo(unittest.TestCase):
    """Test cases for GPUInfo class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.sample_gpu_info = GPUInfo(
            index=0,
            name="Test GPU",
            device_id="cuda:0",
            memory_total=8589934592,  # 8GB
            memory_allocated=1073741824,  # 1GB
            memory_reserved=2147483648,  # 2GB
            memory_free=6442450944,  # 6GB
            compute_capability=(8, 6),
            sm_count=68,
            max_threads_per_sm=1536,
            max_threads_per_block=1024,
            max_shared_memory_per_block=49152,
            warp_size=32,
            memory_clock_rate=19000,
            memory_bus_width=320,
            l2_cache_size=5242880,
            max_texture_1d=131072,
            max_texture_2d=(131072, 65536),
            max_texture_3d=(16384, 16384, 16384)
        )
    
    def test_gpu_info_creation(self):
        """Test GPUInfo object creation"""
        gpu = self.sample_gpu_info
        self.assertEqual(gpu.index, 0)
        self.assertEqual(gpu.name, "Test GPU")
        self.assertEqual(gpu.device_id, "cuda:0")
        self.assertEqual(gpu.memory_total, 8589934592)
    
    def test_memory_calculations(self):
        """Test memory calculation properties"""
        gpu = self.sample_gpu_info
        self.assertAlmostEqual(gpu.memory_total_gb, 8.0, places=1)
        # Note: memory_allocated and memory_free are updated in __post_init__
        # so we test the calculation logic rather than exact values
        self.assertGreaterEqual(gpu.memory_allocated_gb, 0.0)
        self.assertGreaterEqual(gpu.memory_free_gb, 0.0)
        self.assertGreaterEqual(gpu.memory_usage_percent, 0.0)
    
    def test_compute_capability_string(self):
        """Test compute capability string representation"""
        gpu = self.sample_gpu_info
        self.assertEqual(gpu.compute_capability_str, "8.6")
    
    def test_theoretical_performance(self):
        """Test theoretical performance calculation"""
        gpu = self.sample_gpu_info
        performance = gpu.theoretical_fp32_performance
        self.assertGreater(performance, 0)
        self.assertIsInstance(performance, float)
    
    def test_gpu_info_dict_conversion(self):
        """Test conversion to dictionary"""
        gpu = self.sample_gpu_info
        gpu_dict = gpu.to_dict()
        
        # Check required keys
        self.assertIn('index', gpu_dict)
        self.assertIn('name', gpu_dict)
        self.assertIn('device_id', gpu_dict)
        self.assertIn('memory', gpu_dict)
        self.assertIn('compute', gpu_dict)
        
        # Check nested structure
        self.assertIn('total_gb', gpu_dict['memory'])
        self.assertIn('capability', gpu_dict['compute'])
    
    def test_gpu_info_string_representation(self):
        """Test string representation"""
        gpu = self.sample_gpu_info
        str_repr = str(gpu)
        self.assertIn("GPU 0", str_repr)
        self.assertIn("Test GPU", str_repr)
        self.assertIn("8.00 GB", str_repr)


@pytest.mark.unit
class TestDeviceDetection(unittest.TestCase):
    """Test cases for device detection functionality"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_no_cuda_available(self, mock_device_count, mock_is_available):
        """Test device detection when CUDA is not available"""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        devices = detect_gpu_devices()
        self.assertEqual(len(devices), 0)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_cuda_device_detection(self, mock_memory_reserved, mock_memory_allocated,
                                  mock_get_properties, mock_device_count, mock_is_available):
        """Test CUDA device detection"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        mock_memory_allocated.return_value = 0
        mock_memory_reserved.return_value = 0
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = 8589934592
        mock_props.major = 8
        mock_props.minor = 6
        mock_props.multi_processor_count = 68
        mock_props.max_threads_per_multi_processor = 1536
        mock_props.shared_memory_per_block = 49152
        mock_props.warp_size = 32
        mock_props.L2_cache_size = 5242880
        mock_get_properties.return_value = mock_props
        
        devices = detect_gpu_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].name, "Test GPU")
    
    def test_get_optimal_device_empty_list(self):
        """Test optimal device selection with empty GPU list"""
        optimal = get_optimal_device([])
        self.assertEqual(optimal, "cpu")
    
    def test_get_optimal_device_selection(self):
        """Test optimal device selection logic"""
        gpu1 = GPUInfo(0, "GPU1", "cuda:0", 8000000000, 0, 0, 6000000000, 
                      (8, 6), 68, 1536, 1024, 49152, 32, 0, 0, 0, 0, (0, 0), (0, 0, 0))
        gpu2 = GPUInfo(1, "GPU2", "cuda:1", 8000000000, 0, 0, 7000000000, 
                      (8, 6), 68, 1536, 1024, 49152, 32, 0, 0, 0, 0, (0, 0), (0, 0, 0))
        
        # Set memory_free explicitly after creation to override __post_init__ updates
        gpu1.memory_free = 6000000000
        gpu2.memory_free = 7000000000
        
        optimal = get_optimal_device([gpu1, gpu2])
        self.assertEqual(optimal, "cuda:1")  # Should select GPU with more free memory


@pytest.mark.integration
class TestCoordinatorDeviceManagement(unittest.TestCase):
    """Test cases for coordinator device management integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = coordinator
    
    def test_device_initialization(self):
        """Test device management initialization"""
        # Initialize without specific device config
        self.coordinator.initialize()
        
        # Check that device is set
        current_device = self.coordinator.get_device()
        self.assertIsNotNone(current_device)
        self.assertIsInstance(current_device, str)
        
        # Check available devices
        available = self.coordinator.get_available_devices()
        self.assertIsInstance(available, list)
        self.assertIn('cpu', available)
        
        # Cleanup
        self.coordinator.shutdown()
    
    def test_device_configuration_with_cpu(self):
        """Test device initialization with CPU specified"""
        config = {'device': 'cpu'}
        self.coordinator.initialize(config)
        
        device = self.coordinator.get_device()
        self.assertEqual(device, 'cpu')
        
        # Cleanup
        self.coordinator.shutdown()
    
    def test_device_switching(self):
        """Test manual device switching"""
        self.coordinator.initialize()
        
        # Test CPU switch
        success = self.coordinator.set_device('cpu')
        self.assertTrue(success)
        self.assertEqual(self.coordinator.get_device(), 'cpu')
        
        # Test GPU switch if available
        if self.coordinator.is_gpu_available():
            available_devices = self.coordinator.get_available_devices()
            gpu_devices = [d for d in available_devices if d.startswith('cuda')]
            if gpu_devices:
                success = self.coordinator.set_device(gpu_devices[0])
                self.assertTrue(success)
                self.assertEqual(self.coordinator.get_device(), gpu_devices[0])
        
        # Test invalid device
        success = self.coordinator.set_device('invalid_device')
        self.assertFalse(success)
        
        # Cleanup
        self.coordinator.shutdown()
    
    def test_torch_device_integration(self):
        """Test PyTorch device integration"""
        self.coordinator.initialize()
        
        torch_device = self.coordinator.get_torch_device()
        self.assertIsInstance(torch_device, torch.device)
        
        # Test tensor operations
        tensor = torch.randn(2, 2)
        device_tensor = tensor.to(torch_device)
        self.assertEqual(device_tensor.device, torch_device)
        
        # Cleanup
        self.coordinator.shutdown()
    
    def test_device_info_reporting(self):
        """Test device information reporting"""
        self.coordinator.initialize()
        
        device_info = self.coordinator.get_device_info()
        
        # Check required fields
        self.assertIn('current_device', device_info)
        self.assertIn('available_devices', device_info)
        self.assertIn('gpu_available', device_info)
        self.assertIn('gpu_count', device_info)
        
        # Check types
        self.assertIsInstance(device_info['current_device'], str)
        self.assertIsInstance(device_info['available_devices'], list)
        self.assertIsInstance(device_info['gpu_available'], bool)
        self.assertIsInstance(device_info['gpu_count'], int)
        
        # Cleanup
        self.coordinator.shutdown()
    
    def test_gpu_availability_check(self):
        """Test GPU availability checking"""
        self.coordinator.initialize()
        
        gpu_available = self.coordinator.is_gpu_available()
        self.assertIsInstance(gpu_available, bool)
        
        # If GPU is available, check device list
        if gpu_available:
            available_devices = self.coordinator.get_available_devices()
            gpu_devices = [d for d in available_devices if d.startswith('cuda')]
            self.assertGreater(len(gpu_devices), 0)
        
        # Cleanup
        self.coordinator.shutdown()
    
    def test_coordinator_status_includes_device_info(self):
        """Test that coordinator status includes device information"""
        self.coordinator.initialize()
        
        status = self.coordinator.get_status()
        self.assertIn('device_info', status)
        
        device_info = status['device_info']
        self.assertIn('current_device', device_info)
        self.assertIn('gpu_available', device_info)
        
        # Cleanup
        self.coordinator.shutdown()


@pytest.mark.integration
class TestDeviceManagementEndToEnd(unittest.TestCase):
    """End-to-end tests for device management"""
    
    def test_full_device_workflow(self):
        """Test complete device management workflow"""
        coord = coordinator
        
        # Step 1: Initialize
        coord.initialize()
        
        # Step 2: Check initial state
        initial_device = coord.get_device()
        self.assertIsNotNone(initial_device)
        
        # Step 3: Get device info
        device_info = coord.get_device_info()
        self.assertIsNotNone(device_info)
        
        # Step 4: Test device switching
        success = coord.set_device('cpu')
        self.assertTrue(success)
        self.assertEqual(coord.get_device(), 'cpu')
        
        # Step 5: Test PyTorch integration
        torch_device = coord.get_torch_device()
        self.assertEqual(torch_device.type, 'cpu')
        
        # Step 6: Test with tensor
        tensor = torch.randn(3, 3)
        device_tensor = tensor.to(torch_device)
        self.assertEqual(device_tensor.device, torch_device)
        
        # Step 7: Test GPU if available
        if coord.is_gpu_available():
            available_devices = coord.get_available_devices()
            gpu_devices = [d for d in available_devices if d.startswith('cuda')]
            if gpu_devices:
                success = coord.set_device(gpu_devices[0])
                self.assertTrue(success)
                
                gpu_torch_device = coord.get_torch_device()
                self.assertEqual(gpu_torch_device.type, 'cuda')
        
        # Step 8: Cleanup
        coord.shutdown()


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2) 