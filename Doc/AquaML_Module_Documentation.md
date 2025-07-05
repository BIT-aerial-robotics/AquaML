# AquaML Module Documentation

## Project Overview

AquaML is a comprehensive framework for evolutionary machine learning with a focus on reinforcement learning and real-world robotics applications. The framework provides a modular architecture that supports both online and offline learning, multi-process communication, and integration with various robot APIs.

## Project Structure

```
AquaML/
├── framework/          # Core framework implementation
├── param/             # Parameter definitions and data structures
├── policy/            # Policy implementations
├── core/              # Core utilities and base classes
├── communicator/      # Multi-process communication
├── tf/                # TensorFlow implementations
├── torch/             # PyTorch implementations (if available)
├── RobotAPI/          # Robot interface implementations
├── buffer/            # Data buffer implementations
├── algo/              # Algorithm base classes
├── algo_base/         # Algorithm base abstractions
├── worker/            # Worker processes for distributed training
├── recorder/          # Data recording and logging
├── config/            # Configuration files
└── Tool/              # Utility tools
```

## Module Detailed Documentation

### 1. Framework Module (`framework/`)

**Purpose**: Contains the core framework implementation for evolutionary machine learning.

**Key Components**:
- **`EvolutionML.py`**: Main framework class that orchestrates the entire system
  - Manages real-world policies, policy updaters, robot APIs, and policy selectors
  - Handles multi-process communication and coordination
  - Supports both online and offline learning modes
  - Manages data flow between different system components

- **`RL.py`**: Reinforcement learning framework base
- **`FrameWorkBase.py`**: Base class for all framework implementations
- **`RealUpdaterStarter.py`**: Manages real-world policy updates
- **`Offline2online.py`**: Handles transition from offline to online learning

**Main Features**:
- Unified interface for different learning algorithms
- Multi-process task coordination
- Real-world robot interaction management
- Policy candidate selection and evaluation

### 2. Parameter Module (`param/`)

**Purpose**: Defines parameter structures and data information for the entire system.

**Key Components**:
- **`DataInfo.py`**: Core data structure management
  - Defines environment information and data schemas
  - Manages RL states, actions, and rewards
  - Handles data element registration and validation

- **`AquaParam.py`**: General parameter definitions
- **`OfflineRL.py`**: Offline RL specific parameters
- **`PolicyCandidate.py`**: Policy candidate parameters
- **`Buffer.py`**: Buffer configuration parameters
- **`ParamBase.py`**: Base parameter class

**Main Features**:
- Type-safe parameter definitions
- Validation and schema management
- Hierarchical parameter organization

### 3. Policy Module (`policy/`)

**Purpose**: Implements various policy types for different use cases.

**Key Components**:
- **`RealWorldPolicy.py`**: Real-world interaction policies
  - `DeterminateRealWorldPolicy`: Deterministic policies for real-world deployment
  - Supports model switching and weight loading
  - Thread-safe operations for continuous interaction

- **`FixNNPolicy.py`**: Fixed neural network policies
- **`PolicyBase.py`**: Base class for all policies
- **`RealWorldPolicyBase.py`**: Base class for real-world policies

**Main Features**:
- Real-time policy execution
- Model switching capabilities
- Multi-threaded policy management
- Integration with file system for model updates

### 4. Core Module (`core/`)

**Purpose**: Provides core utilities and base classes used throughout the system.

**Key Components**:
- **`DataModule.py`**: Data management and processing
- **`DataUnit.py`**: Basic data unit definitions
- **`Tool.py`**: Utility functions and tools
- **`FileSystem.py`**: File system operations
- **`DataInfo.py`**: Data information management
- **`DataList.py`**: Data list operations
- **`Communicator.py`**: Communication protocols
- **`Recorder.py`**: Data recording utilities
- **`Protocol.py`**: System protocols
- **`TaskBase.py`**: Task base definitions

**Main Features**:
- Data structure management
- File I/O operations
- System utilities
- Protocol definitions

### 5. Communicator Module (`communicator/`)

**Purpose**: Handles multi-process communication and coordination.

**Key Components**:
- **`MPICommunicator.py`**: MPI-based communication
  - Uses MPI for distributed processing
  - Supports process synchronization and barriers
  - Configurable logging and debugging

- **`DebugCommunicator.py`**: Debug-only communication for testing
- **`CommunicatorBase.py`**: Base communication class
- **`ProcessSimulator.py`**: Process simulation utilities

**Main Features**:
- Distributed computing support
- Process synchronization
- Logging and debugging
- Scalable communication patterns

### 6. TensorFlow Module (`tf/`)

**Purpose**: TensorFlow-specific implementations of algorithms and models.

**Key Components**:

#### 6.1 Offline RL (`tf/OfflineRL/`)
- **`IQL.py`**: Implicit Q-Learning implementation
  - Actor-critic architecture with state-value function
  - Expectile regression for conservative learning
  - Support for continuous action spaces

- **`TD3BC.py`**: Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning
  - Combines TD3 with behavior cloning
  - Offline RL with conservative updates

#### 6.2 Policy Candidate (`tf/PolicyCandidate/`)
- **`PEX.py`**: Policy Expansion (PEX) implementation
  - Policy candidate selection algorithm
  - Supports both IQL and TD3BC backends
  - Temperature-based policy selection

#### 6.3 Common Components
- **`TFAlgoBase.py`**: Base class for TensorFlow algorithms
- **`Dataset.py`**: Dataset management for TensorFlow

**Main Features**:
- State-of-the-art offline RL algorithms
- Policy candidate evaluation
- TensorFlow integration
- GPU support

### 7. Robot API Module (`RobotAPI/`)

**Purpose**: Provides interfaces for different robot platforms and simulators.

**Key Components**:
- **`GymWrapper.py`**: OpenAI Gym environment wrapper
  - Standardized interface for Gym environments
  - Support for various observation and action spaces
  - Episode management and reset functionality

- **`GymWrapperPex.py`**: Gym wrapper with PEX integration
- **`ROSAPI.py`**: ROS (Robot Operating System) integration
- **`APIBase.py`**: Base class for all robot APIs
- **`ROSExample.py`**: Example ROS implementation

**Main Features**:
- Multi-platform robot support
- Standardized API interface
- Real-time robot control
- Simulation environment integration

### 8. Buffer Module (`buffer/`)

**Purpose**: Implements various data buffer types for experience replay and data management.

**Key Components**:
- **`BufferBase.py`**: Base buffer implementation
- **`MixtureBuffer.py`**: Mixed data buffer for different data types
- **`MixtureBufferBase.py`**: Base class for mixture buffers
- **`RealCollectBuffer.py`**: Real-world data collection buffer
- **`RealCollectBufferBase.py`**: Base for real collection buffers
- **`DynamicBufferBase.py`**: Dynamic buffer with variable size

**Main Features**:
- Efficient data storage and retrieval
- Multiple buffer types for different use cases
- Memory management
- Real-time data collection support

### 9. Algorithm Module (`algo/`)

**Purpose**: Base classes and interfaces for learning algorithms.

**Key Components**:
- **`AlgoBase.py`**: Base algorithm class
- **`RLAlgoBase.py`**: Reinforcement learning algorithm base
- **`ModelBase.py`**: Base model class

**Main Features**:
- Algorithm abstraction
- Consistent interface design
- Extensible architecture

### 10. Algorithm Base Module (`algo_base/`)

**Purpose**: Provides abstract base classes for different algorithm categories.

**Key Components**:
- **`OfflineRLBase.py`**: Base class for offline RL algorithms
- Other algorithm base classes

**Main Features**:
- Type-safe algorithm interfaces
- Common functionality abstraction
- Consistent API design

### 11. Worker Module (`worker/`)

**Purpose**: Implements worker processes for distributed training and data collection.

**Key Components**:
- **`RLWorker.py`**: Standard RL worker
- **`RLWorkerBase.py`**: Base RL worker class
- **`RLCollector.py`**: Data collection worker
- **`RLVectorEnv.py`**: Vectorized environment worker
- **`RLEnvBase.py`**: Environment base class
- **`RLIsaacGymWorker.py`**: Isaac Gym specific worker
- **`RLAerialGymWorker.py`**: Aerial Gym specific worker

**Main Features**:
- Distributed training support
- Parallel data collection
- Environment vectorization
- Platform-specific optimizations

### 12. Recorder Module (`recorder/`)

**Purpose**: Handles data recording, logging, and experiment tracking.

**Key Components**:
- **`RecorderBase.py`**: Base recorder class
- **`WandbRecorder.py`**: Weights & Biases integration
- **`BoardRecorder.py`**: TensorBoard integration

**Main Features**:
- Experiment tracking
- Metrics visualization
- Multiple backend support
- Real-time monitoring

### 13. Configuration Module (`config/`)

**Purpose**: Contains configuration files and settings for the framework.

**Main Features**:
- Process configuration
- Environment settings
- Algorithm parameters
- System configuration

### 14. Tool Module (`Tool/`)

**Purpose**: Utility tools and helper functions.

**Main Features**:
- Data processing utilities
- Visualization tools
- System utilities
- Development helpers

## Usage Example

The framework is designed to be used as shown in the `IQLBipedalWalker.py` example:

1. **Define Environment**: Set up the environment information using `DataInfo`
2. **Configure Communication**: Set up multi-process communication with `MPICommunicator`
3. **Define Networks**: Create neural network architectures for actors and critics
4. **Configure Algorithms**: Set up offline RL algorithms (e.g., IQL)
5. **Define Policies**: Configure real-world policies for robot interaction
6. **Set up Robot API**: Configure robot interfaces (e.g., Gym environments)
7. **Initialize Framework**: Create `EvolutionML` instance with all components
8. **Run Training**: Execute the training loop

## Key Design Principles

1. **Modularity**: Each component is designed to be independent and replaceable
2. **Extensibility**: Easy to add new algorithms, policies, and robot platforms
3. **Scalability**: Support for distributed training and multi-process execution
4. **Real-world Focus**: Designed specifically for real-world robotics applications
5. **Framework Agnostic**: Supports both TensorFlow and PyTorch (planned)

## Dependencies

- TensorFlow 2.x
- OpenAI Gym
- MPI4Py (for distributed computing)
- NumPy
- Weights & Biases (optional, for experiment tracking)
- ROS (optional, for robot integration)

## Notes

- The framework is actively under development with some modules marked as "old" indicating ongoing refactoring
- GPU support is available for TensorFlow implementations
- The system supports both simulation and real-world robot deployment
- Multi-process communication enables scalable training on distributed systems 