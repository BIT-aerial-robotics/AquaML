#!/usr/bin/env python3
"""
Simplified test file for AquaML Core Data Units

This test file focuses on basic functionality testing
without complex type checking that causes linter errors.
"""

import sys
import os
import traceback
import numpy as np
import torch

# Add the current directory to the path so we can import AquaML modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AquaML.data.core_units import (
        DataMode, DataFormat, UnitConfig, BaseUnit, TensorUnit, NumpyUnit,
        DataUnitFactory
    )
    from AquaML.core.exceptions import AquaMLException
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class TestResult:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"✓ {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n=== Test Summary ===")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        if self.failed > 0:
            print(f"Success rate: {(self.passed/total)*100:.1f}%")
            print("\nFailed tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        else:
            print("All tests passed! 🎉")


def run_test(test_name, test_func, result):
    """Run a single test"""
    try:
        test_func()
        result.add_pass(test_name)
    except Exception as e:
        result.add_fail(test_name, str(e))


def test_data_enums():
    """Test data enums"""
    assert DataMode.NUMPY.value == "numpy"
    assert DataMode.TORCH.value == "torch"
    assert DataMode.AUTO.value == "auto"
    
    assert DataFormat.NUMPY.value == "numpy"
    assert DataFormat.TORCH.value == "torch"
    assert DataFormat.AUTO.value == "auto"


def test_unit_config_basic():
    """Test basic UnitConfig functionality"""
    config = UnitConfig(
        name="test_unit",
        dtype=torch.float32,
        single_shape=(10, 20),
        size=100,
        mode=DataMode.TORCH,
        device="cpu"
    )
    
    assert config.name == "test_unit"
    assert config.dtype == torch.float32
    assert config.single_shape == (10, 20)
    assert config.size == 100
    assert config.mode == DataMode.TORCH
    assert config.device == "cpu"
    assert config.shape == (100, 10, 20)
    assert config.bytes > 0


def test_unit_config_shape_calculation():
    """Test shape calculation"""
    config = UnitConfig(
        name="test",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10
    )
    assert config.shape == (10, 5, 5)


def test_unit_config_bytes_calculation():
    """Test bytes calculation"""
    config = UnitConfig(
        name="test",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10
    )
    # 10 * 5 * 5 * 4 bytes (float32)
    assert config.bytes == 1000


def test_tensor_unit_creation():
    """Test TensorUnit creation"""
    config = UnitConfig(
        name="tensor_unit",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10,
        mode=DataMode.TORCH,
        device="cpu"
    )
    unit = TensorUnit(config)
    
    assert unit.name == "tensor_unit"
    assert unit.mode == DataMode.TORCH
    assert not unit.is_initialized


def test_tensor_unit_data_creation():
    """Test TensorUnit data creation"""
    config = UnitConfig(
        name="tensor_unit",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10,
        mode=DataMode.TORCH,
        device="cpu"
    )
    unit = TensorUnit(config)
    data = unit.create_data()
    
    assert isinstance(data, torch.Tensor)
    assert data.shape == (10, 5, 5)
    assert data.dtype == torch.float32
    assert unit.is_initialized


def test_numpy_unit_creation():
    """Test NumpyUnit creation"""
    config = UnitConfig(
        name="numpy_unit",
        dtype=np.float32,
        single_shape=(5, 5),
        size=10,
        mode=DataMode.NUMPY
    )
    unit = NumpyUnit(config)
    
    assert unit.name == "numpy_unit"
    assert unit.mode == DataMode.NUMPY
    assert not unit.is_initialized


def test_numpy_unit_data_creation():
    """Test NumpyUnit data creation"""
    config = UnitConfig(
        name="numpy_unit",
        dtype=np.float32,
        single_shape=(5, 5),
        size=10,
        mode=DataMode.NUMPY
    )
    unit = NumpyUnit(config)
    data = unit.create_data()
    
    assert isinstance(data, np.ndarray)
    assert data.shape == (10, 5, 5)
    assert data.dtype == np.float32
    assert unit.is_initialized


def test_factory_create_tensor_unit():
    """Test DataUnitFactory tensor unit creation"""
    unit = DataUnitFactory.create_tensor_unit(
        name="test_tensor",
        shape=(10, 5, 5),
        dtype=torch.float32,
        device="cpu"
    )
    
    assert isinstance(unit, TensorUnit)
    assert unit.name == "test_tensor"
    assert unit.size == 10
    assert unit.single_shape == (5, 5)
    assert unit.dtype == torch.float32
    assert unit.device == "cpu"


def test_factory_create_numpy_unit():
    """Test DataUnitFactory numpy unit creation"""
    unit = DataUnitFactory.create_numpy_unit(
        name="test_numpy",
        shape=(10, 5, 5),
        dtype=np.float32
    )
    
    assert isinstance(unit, NumpyUnit)
    assert unit.name == "test_numpy"
    assert unit.size == 10
    assert unit.single_shape == (5, 5)
    assert unit.dtype == np.float32


def test_tensor_to_numpy_conversion():
    """Test tensor to numpy conversion"""
    tensor_unit = DataUnitFactory.create_tensor_unit(
        name="conversion_test",
        shape=(5, 3, 3),
        dtype=torch.float32
    )
    
    # Create data and fill with ones
    tensor_data = tensor_unit.create_data()
    tensor_data.fill_(1.0)
    
    # Convert to numpy unit
    numpy_unit = tensor_unit.to_numpy_unit()
    
    # Verify conversion
    assert isinstance(numpy_unit, NumpyUnit)
    assert numpy_unit.is_initialized
    assert np.allclose(numpy_unit.data_, 1.0)


def test_numpy_to_tensor_conversion():
    """Test numpy to tensor conversion"""
    numpy_unit = DataUnitFactory.create_numpy_unit(
        name="conversion_test",
        shape=(5, 3, 3),
        dtype=np.float32
    )
    
    # Create data and fill with ones
    numpy_data = numpy_unit.create_data()
    numpy_data.fill(1.0)
    
    # Convert to tensor unit
    tensor_unit = numpy_unit.to_tensor_unit()
    
    # Verify conversion
    assert isinstance(tensor_unit, TensorUnit)
    assert tensor_unit.is_initialized
    assert torch.allclose(tensor_unit.data_, torch.ones_like(tensor_unit.data_))


def test_unit_properties():
    """Test unit properties"""
    config = UnitConfig(
        name="test_unit",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10,
        enable_history=True
    )
    
    # Create concrete implementation for testing
    class TestUnit(BaseUnit):
        def create_data(self):
            self.data_ = np.zeros(self.unit_cfg_.shape)
            self.is_initialized_ = True
            return self.data_
        
        def compute_bytes(self):
            return self.unit_cfg_.bytes or 0
    
    unit = TestUnit(config)
    
    assert unit.name == "test_unit"
    assert unit.single_shape == (5, 5)
    assert unit.size == 10
    assert unit.device == "cpu"
    assert unit.bytes > 0


def test_unit_history():
    """Test unit history functionality"""
    config = UnitConfig(
        name="test_unit",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10,
        enable_history=True
    )
    
    class TestUnit(BaseUnit):
        def create_data(self):
            self.data_ = np.zeros(self.unit_cfg_.shape)
            self.is_initialized_ = True
            return self.data_
        
        def compute_bytes(self):
            return self.unit_cfg_.bytes or 0
    
    unit = TestUnit(config)
    
    # Initially no history
    assert unit.get_history() == []
    
    # Add some data to history
    unit.add_to_history([1, 2, 3])
    unit.add_to_history([4, 5, 6])
    
    history = unit.get_history()
    assert len(history) == 2
    assert history[0] == [1, 2, 3]
    assert history[1] == [4, 5, 6]
    
    # Clear history
    unit.clear_history()
    assert unit.get_history() == []


def test_unit_reset():
    """Test unit reset functionality"""
    config = UnitConfig(
        name="test_unit",
        dtype=torch.float32,
        single_shape=(5, 5),
        size=10,
        enable_history=True
    )
    
    class TestUnit(BaseUnit):
        def create_data(self):
            self.data_ = np.zeros(self.unit_cfg_.shape)
            self.is_initialized_ = True
            return self.data_
        
        def compute_bytes(self):
            return self.unit_cfg_.bytes or 0
    
    unit = TestUnit(config)
    unit.create_data()
    unit.add_to_history([1, 2, 3])
    
    assert unit.is_initialized
    assert len(unit.get_history()) == 1
    
    unit.reset()
    assert not unit.is_initialized
    assert unit.data_ is None
    assert len(unit.get_history()) == 0


def main():
    """Run all tests"""
    print("Running AquaML Core Units Tests...\n")
    
    result = TestResult()
    
    # Define all tests
    tests = [
        ("Data Enums", test_data_enums),
        ("UnitConfig Basic", test_unit_config_basic),
        ("UnitConfig Shape Calculation", test_unit_config_shape_calculation),
        ("UnitConfig Bytes Calculation", test_unit_config_bytes_calculation),
        ("TensorUnit Creation", test_tensor_unit_creation),
        ("TensorUnit Data Creation", test_tensor_unit_data_creation),
        ("NumpyUnit Creation", test_numpy_unit_creation),
        ("NumpyUnit Data Creation", test_numpy_unit_data_creation),
        ("Factory Create TensorUnit", test_factory_create_tensor_unit),
        ("Factory Create NumpyUnit", test_factory_create_numpy_unit),
        ("Tensor to Numpy Conversion", test_tensor_to_numpy_conversion),
        ("Numpy to Tensor Conversion", test_numpy_to_tensor_conversion),
        ("Unit Properties", test_unit_properties),
        ("Unit History", test_unit_history),
        ("Unit Reset", test_unit_reset),
    ]
    
    # Run all tests
    for test_name, test_func in tests:
        run_test(test_name, test_func, result)
    
    # Print summary
    result.summary()
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 