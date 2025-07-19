# AquaML File Management System (FileSystem)

## Overview

The AquaML FileSystem module provides unified file and directory management functionality. It serves as a core component of the framework, responsible for managing all file operation-related path generation, directory creation, and file organization structure.

## Architecture Design

```
FileSystem Architecture
├── BaseFileSystem (Abstract Base Class)
│   ├── Basic Path Management
│   ├── Runner Configuration Management
│   └── Unified File Operation Interface
├── DefaultFileSystem (Default Implementation)
│   └── Standard FileSystem Implementation
└── FileSystemManager (Manager)
    ├── Register and Retrieve FileSystem Instances
    └── Integration with Coordinator
```

## Directory Structure

AquaML's standard directory structure:

```
workspace_dir/                    # Workspace Root Directory
├── logger/                       # Global Log Directory
│   └── YYYY-MM-DD-HH-MM-SS.log  # Time-stamped Log Files
├── runner_name_1/               # Runner-specific or data-time Directory
│   ├── cache/                   # Cache Directory
│   ├── history_model/           # Historical Model Storage
│   ├── log/                     # Runner Log Directory
│   │   ├── checkpoints/         # Checkpoint Storage
│   │   └── tensorboard/         # TensorBoard Logs
│   └── data_config/            # Data Configuration Storage
│       ├── data_unit.yaml      # Data Unit Configuration
│       └── env_info.yaml       # Environment Information
└── runner_name_2/               # Other Runner Directories
    └── ...
```

## Main Components

### 1. BaseFileSystem

Abstract base class defining the core FileSystem interface:

```python
class BaseFileSystem(ABC):
    def __init__(self, workspace_dir: str)
    def initFolder(self)                              # Initialize directory structure
    def configRunner(self, runner_name: str)          # Configure runner directories
    def ensureDir(self, dir_path: str) -> bool        # Ensure directory exists
    
    # Path query methods
    def queryHistoryModelPath(self, runner_name: str) -> str
    def queryCachePath(self, runner_name: str) -> str
    def queryLogPath(self, runner_name: str) -> str
    def queryDataUnitFile(self, runner_name: str) -> str
    def queryEnvInfoFile(self, runner_name: str) -> str
    
    # Unified path management interface
    def getCheckpointDir(self, runner_name: str) -> str
    def getCheckpointPath(self, runner_name: str, checkpoint_name: str) -> str
    def getModelPath(self, runner_name: str, model_name: str) -> str
    def getLogDir(self, runner_name: str) -> str
    def getTensorboardLogDir(self, runner_name: str) -> str
    def getExperimentDir(self, runner_name: str) -> str
    
    # Data saving methods
    def saveDataUnitInfo(self, runner_name: str, data_unit_status: dict)
    def saveEnvInfo(self, runner_name: str, env_info: dict)
```

### 2. DefaultFileSystem

Default FileSystem implementation, inheriting from BaseFileSystem:

```python
class DefaultFileSystem(BaseFileSystem):
    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
```

### 3. FileSystemManager

Responsible for registering and managing FileSystem instances:

```python
class FileSystemManager:
    def register_file_system(self, file_system_cls: type) -> Callable
    def get_file_system(self) -> Any
    def set_file_system(self, file_system_instance: Any) -> None
    def file_system_exists(self) -> bool
    def remove_file_system(self) -> None
    def config_runner(self, runner_name: str) -> None
```

## Usage

### 1. Basic Usage

FileSystem is automatically created when Coordinator initializes:

```python
from AquaML import coordinator

# FileSystem is automatically initialized
print(f"FileSystem exists: {coordinator.file_system_manager.file_system_exists()}")

# Get FileSystem instance
fs = coordinator.getFileSystem()
```

### 2. Runner Management

```python
# Register Runner (automatically configures directory structure)
coordinator.registerRunner('my_experiment')

# Get various paths
fs = coordinator.getFileSystem()
checkpoint_dir = fs.getCheckpointDir('my_experiment')
model_path = fs.getModelPath('my_experiment', 'final_model')
log_dir = fs.getLogDir('my_experiment')
```

### 3. Directory Management

```python
# Ensure directory exists
fs.ensureDir('/path/to/custom/directory')

# Get TensorBoard log directory
tb_dir = fs.getTensorboardLogDir('my_experiment')

# Get checkpoint path
checkpoint_path = fs.getCheckpointPath('my_experiment', 'epoch_100')
```

### 4. Data Saving

```python
# Save data unit information
data_unit_info = {'batch_size': 32, 'sequence_length': 100}
fs.saveDataUnitInfo('my_experiment', data_unit_info)

# Save environment information
env_info = {'env_name': 'CartPole-v1', 'max_steps': 500}
fs.saveEnvInfo('my_experiment', env_info)
```

## Integration with Other Modules

### 1. Integration with Coordinator

FileSystem is a required component of Coordinator:

```python
class AquaMLCoordinator:
    def __init__(self):
        # ...
        self.file_system_manager = FileSystemManager()
        self._initialize_default_file_system()  # Auto initialize
```

### 2. Integration with Logging System

Logging system uses FileSystem to manage log directories:

```python
def configure_loguru_logging(log_level, log_file, file_system_instance):
    if log_file and file_system_instance:
        log_dir = os.path.dirname(log_file)
        file_system_instance.ensureDir(log_dir)
```

### 3. Integration with Learning Modules

Model saving and checkpoint management use FileSystem:

```python
# In Agent
def save_checkpoint(self, timestep):
    file_system = coordinator.getFileSystem()
    runner_name = coordinator.getRunner()
    checkpoint_dir = file_system.getCheckpointDir(runner_name)
    # Save checkpoint...

# In Model
def save(self, path):
    file_system = coordinator.getFileSystem()
    file_system.ensureDir(os.path.dirname(path))
    torch.save(self.state_dict(), path)
```

## Configuration Options

### Default Workspace

```python
# Default workspace location
default_workspace = os.path.join(os.getcwd(), "aquaml_workspace")
```

### Custom FileSystem

```python
class CustomFileSystem(BaseFileSystem):
    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
        # Custom implementation...

# Register custom FileSystem
custom_fs = CustomFileSystem("/custom/workspace")
coordinator.file_system_manager.set_file_system(custom_fs)
```

## Best Practices

### 1. Path Management

- **Always use FileSystem to get paths**, avoid hardcoding
- **Use ensureDir() to ensure directories exist**, not direct os.makedirs calls
- **Organize files by Runner name**, maintain experiment isolation

### 2. Error Handling

```python
try:
    file_system = coordinator.getFileSystem()
    path = file_system.getModelPath(runner_name, model_name)
except Exception:
    # Fallback to direct path handling
    path = os.path.join("fallback_dir", f"{model_name}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
```

### 3. Performance Considerations

- **Directory creation is idempotent**, multiple ensureDir() calls are safe
- **Path query methods are lightweight**, can be called frequently
- **Use caching mechanisms** to avoid repeated filesystem operations

## Troubleshooting

### Common Issues

1. **FileSystem not initialized**
   ```
   Error: File system not exists!
   Solution: Ensure coordinator is properly initialized
   ```

2. **Runner not registered**
   ```
   Error: runner xxx not registered
   Solution: Use coordinator.registerRunner() to register Runner
   ```

3. **Permission issues**
   ```
   Error: Permission denied
   Solution: Ensure write permissions to workspace directory
   ```

### Debugging Tips

```python
# Check FileSystem status
status = coordinator.file_system_manager.get_status()
print(f"FileSystem Status: {status}")

# View directory structure
fs = coordinator.getFileSystem()
print(f"Workspace: {fs.workspace_dir_}")
print(f"Runner directories: {fs.runner_dir_dict_}")
```

## Extension Development

### Custom FileSystem Implementation

```python
class CloudFileSystem(BaseFileSystem):
    """Cloud storage FileSystem implementation"""
    
    def __init__(self, workspace_dir: str, cloud_config: dict):
        super().__init__(workspace_dir)
        self.cloud_config = cloud_config
    
    def ensureDir(self, dir_path: str) -> bool:
        # Cloud storage directory creation logic
        pass
    
    def saveModel(self, model_data, file_path: str):
        # Cloud storage model saving logic
        pass
```

### External Storage Integration

```python
class S3FileSystem(BaseFileSystem):
    """AWS S3 storage FileSystem"""
    
    def __init__(self, workspace_dir: str, s3_bucket: str):
        super().__init__(workspace_dir)
        self.s3_bucket = s3_bucket
        # S3 client initialization...
```

## Before and After Refactoring

### Before (Scattered Management)

```python
# Various modules directly create directories
os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.dirname(path), exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Hardcoded paths
checkpoint_path = os.path.join(self.experiment_dir, "checkpoints", f"{name}_{tag}.pt")
```

### After (Unified Management)

```python
# Unified management through FileSystem
file_system = coordinator.getFileSystem()
runner_name = coordinator.getRunner()

# Automatic directory creation and path generation
checkpoint_dir = file_system.getCheckpointDir(runner_name)
checkpoint_path = file_system.getCheckpointPath(runner_name, f"{name}_{tag}")
model_path = file_system.getModelPath(runner_name, model_name)
```

## Version History

- **v1.0**: Basic FileSystem implementation
- **v1.1**: Added unified path management interface
- **v1.2**: Integrated into Coordinator as required component
- **v1.3**: Added logging system integration
- **v1.4**: Completed framework-wide file operation unification refactoring

## Related Documentation

- [Architecture.md](./Architecture.md) - Overall architecture documentation
- [README.md](../README.md) - Project overview and usage tutorial