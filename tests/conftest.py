"""
Pytest configuration and shared fixtures for AquaML tests.
"""

import pytest
import tempfile
import shutil
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def temp_workspace():
    """Create a temporary workspace for tests"""
    temp_dir = tempfile.mkdtemp(prefix="aquaml_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_coordinator():
    """Provide a clean coordinator instance for each test"""
    from AquaML import coordinator
    # Clear any existing registrations
    coordinator.registry.clear()
    yield coordinator
    # Clean up after test
    coordinator.registry.clear()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment"""
    # Suppress unnecessary warnings during tests
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set up logging for tests
    import logging
    logging.getLogger("AquaML").setLevel(logging.WARNING)
    
    yield
    
    # Clean up after all tests
    pass 