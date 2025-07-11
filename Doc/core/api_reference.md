# AquaML Core 模块 API 参考文档

## 模块导入

```python
from AquaML.core.coordinator import (
    AquaMLCoordinator,
    get_coordinator
)

from AquaML.core.exceptions import (
    AquaMLException
)

from AquaML.core.device_info import (
    GPUInfo,
    detect_gpu_devices,
    get_optimal_device
)
```

## AquaMLCoordinator

### 类描述
主要协调器类，采用单例模式，负责管理整个AquaML框架的组件注册、存储和设备管理。新版本采用简化架构，直接存储组件实例，避免复杂的注册系统。

### 构造函数
```python
AquaMLCoordinator()
```
使用单例模式，多次调用返回同一实例。

### 初始化方法

#### `initialize(config: Optional[Dict[str, Any]] = None) -> None`
初始化协调器

**参数:**
- `config`: 配置字典，可选。可包含设备设置和插件配置

**异常:**
- `AquaMLException`: 初始化失败时抛出

**示例:**
```python
coordinator = get_coordinator()
coordinator.initialize({
    "device": "cuda:0",
    "plugins": {"example": {"path": "plugins/example"}}
})
```

#### `shutdown() -> None`
关闭协调器，清理所有组件引用

**示例:**
```python
coordinator.shutdown()
```

### 模型管理方法

#### `registerModel(model: Module, model_name: str) -> None`
注册机器学习模型

**参数:**
- `model`: PyTorch模型实例
- `model_name`: 模型名称（必须唯一）

**异常:**
- `ValueError`: 模型名称已存在时抛出

**示例:**
```python
import torch.nn as nn
model = nn.Linear(10, 1)
coordinator.registerModel(model, "linear_model")
```

#### `getModel(model_name: str) -> Dict[str, Any]`
获取已注册的模型信息

**参数:**
- `model_name`: 模型名称

**返回:**
- 包含 'model' 和 'status' 键的字典

**异常:**
- `ValueError`: 模型不存在时抛出

**示例:**
```python
model_info = coordinator.getModel("linear_model")
model = model_info['model']
status = model_info['status']
```

### 环境管理方法

#### `registerEnv(env_cls) -> Callable`
注册强化学习环境类的装饰器

**参数:**
- `env_cls`: 环境类

**返回:**
- 装饰器函数

**示例:**
```python
@coordinator.registerEnv
class MyEnvironment:
    def __init__(self):
        self.name = "CartPole-v1"
        
env = MyEnvironment()  # 自动注册
```

#### `getEnv() -> Any`
获取已注册的环境实例

**返回:**
- 环境实例

**异常:**
- `ValueError`: 环境未注册时抛出

**示例:**
```python
env = coordinator.getEnv()
```

### 智能体管理方法

#### `registerAgent(agent_cls) -> Callable`
注册强化学习智能体类的装饰器

**参数:**
- `agent_cls`: 智能体类

**返回:**
- 装饰器函数

**异常:**
- `ValueError`: 尝试注册多个智能体时抛出

**示例:**
```python
@coordinator.registerAgent
class MyAgent:
    def __init__(self):
        self.name = "DQNAgent"
        
agent = MyAgent()  # 自动注册
```

#### `getAgent() -> Any`
获取已注册的智能体实例

**返回:**
- 智能体实例

**异常:**
- `ValueError`: 智能体未注册时抛出

**示例:**
```python
agent = coordinator.getAgent()
```

### 数据单元管理方法

#### `registerDataUnit(data_unit_cls) -> Callable`
注册数据单元类的装饰器

**参数:**
- `data_unit_cls`: 数据单元类

**返回:**
- 装饰器函数

**示例:**
```python
@coordinator.registerDataUnit
class MyDataUnit:
    def __init__(self, name="data_unit"):
        self.name = name
        
    def getUnitStatusDict(self):
        return {"status": "active", "size": 1000}
        
data_unit = MyDataUnit("experience_buffer")
```

#### `getDataUnit(unit_name: str) -> Any`
获取指定名称的数据单元实例

**参数:**
- `unit_name`: 数据单元名称

**返回:**
- 数据单元实例

**异常:**
- `ValueError`: 数据单元不存在时抛出

**示例:**
```python
data_unit = coordinator.getDataUnit("experience_buffer")
```

### 文件系统管理方法

#### `registerFileSystem(file_system_cls) -> Callable`
注册文件系统类的装饰器

**参数:**
- `file_system_cls`: 文件系统类

**返回:**
- 装饰器函数

**异常:**
- `ValueError`: 尝试注册多个文件系统时抛出

**示例:**
```python
@coordinator.registerFileSystem
class MyFileSystem:
    def configRunner(self, runner_name):
        pass
        
    def saveDataUnit(self, runner_name, data_unit_status):
        pass
        
fs = MyFileSystem()
```

#### `getFileSystem() -> Any`
获取已注册的文件系统实例

**返回:**
- 文件系统实例

**异常:**
- `ValueError`: 文件系统未注册时抛出

### 通信器管理方法

#### `registerCommunicator(communicator_cls) -> Callable`
注册通信器类的装饰器

**参数:**
- `communicator_cls`: 通信器类

**返回:**
- 装饰器函数

**异常:**
- `ValueError`: 尝试注册多个通信器时抛出

**示例:**
```python
@coordinator.registerCommunicator
class MyCommmunicator:
    def __init__(self):
        self.name = "MPICommunicator"
        
comm = MyCommmunicator()
```

#### `getCommunicator() -> Any`
获取已注册的通信器实例

**返回:**
- 通信器实例

**异常:**
- `ValueError`: 通信器未注册时抛出

### 运行器管理方法

#### `registerRunner(runner_name: str) -> None`
注册运行器名称

**参数:**
- `runner_name`: 运行器名称

**示例:**
```python
coordinator.registerRunner("dqn_experiment")
```

#### `getRunner() -> str`
获取已注册的运行器名称

**返回:**
- 运行器名称字符串

**异常:**
- `ValueError`: 运行器未注册时抛出

### 数据管理方法

#### `register_data_manager(data_manager_cls) -> Callable`
注册数据管理器类的装饰器

**参数:**
- `data_manager_cls`: 数据管理器类

**返回:**
- 装饰器函数

#### `get_data_manager() -> Any`
获取已注册的数据管理器实例

**返回:**
- 数据管理器实例

**异常:**
- `ValueError`: 数据管理器未注册时抛出

#### `saveDataUnitInfo() -> None`
保存所有数据单元信息到文件系统

**异常:**
- `ValueError`: 运行器未注册时抛出

**示例:**
```python
coordinator.saveDataUnitInfo()
```

### 设备管理方法

#### `get_device() -> str`
获取当前计算设备

**返回:**
- 当前设备字符串（如'cpu', 'cuda:0'等）

**示例:**
```python
device = coordinator.get_device()
print(f"当前设备: {device}")
```

#### `get_torch_device() -> torch.device`
获取PyTorch设备对象

**返回:**
- PyTorch设备对象

**示例:**
```python
torch_device = coordinator.get_torch_device()
tensor = torch.randn(5, 5).to(torch_device)
```

#### `set_device(device: str) -> bool`
设置计算设备

**参数:**
- `device`: 设备字符串（如'cpu', 'cuda:0'等）

**返回:**
- 布尔值，表示设置是否成功

**示例:**
```python
if coordinator.set_device("cuda:0"):
    print("成功设置为GPU 0")
else:
    print("GPU 0 不可用")
```

#### `get_available_devices() -> List[str]`
获取所有可用设备列表

**返回:**
- 设备字符串列表

**示例:**
```python
devices = coordinator.get_available_devices()
print(f"可用设备: {devices}")
```

#### `validate_device(device: str) -> bool`
验证设备是否可用

**参数:**
- `device`: 设备字符串

**返回:**
- 布尔值，表示设备是否可用

**示例:**
```python
if coordinator.validate_device("cuda:0"):
    print("GPU 0 可用")
```

#### `is_gpu_available() -> bool`
检查是否有可用的GPU

**返回:**
- 布尔值，表示GPU是否可用

**示例:**
```python
if coordinator.is_gpu_available():
    print("有GPU可用")
```

#### `get_device_info() -> Dict[str, Any]`
获取详细的设备信息

**返回:**
- 包含设备信息的字典，包含以下键：
  - `current_device`: 当前设备
  - `available_devices`: 可用设备列表
  - `gpu_available`: GPU是否可用
  - `gpu_count`: GPU数量
  - `gpu_details`: GPU详细信息（如果有）

**示例:**
```python
device_info = coordinator.get_device_info()
print(f"当前设备: {device_info['current_device']}")
print(f"GPU数量: {device_info['gpu_count']}")
```

### 状态和工具方法

#### `get_status() -> Dict[str, Any]`
获取协调器完整状态信息

**返回:**
- 包含状态信息的字典，包含以下键：
  - `initialized`: 是否已初始化
  - `components`: 组件统计信息
  - `device_info`: 设备信息
  - `runner_name`: 当前运行器名称

**示例:**
```python
status = coordinator.get_status()
print(f"初始化状态: {status['initialized']}")
print(f"注册的组件: {status['components']}")
```

#### `list_components() -> Dict[str, int]`
列出所有已注册的组件及其数量

**返回:**
- 组件类型及数量的字典

**示例:**
```python
components = coordinator.list_components()
print(f"模型数量: {components['models']}")
print(f"数据单元数量: {components['data_units']}")
```

### 上下文管理器方法

#### `__enter__() -> AquaMLCoordinator`
上下文管理器入口

#### `__exit__(exc_type, exc_val, exc_tb) -> None`
上下文管理器出口，自动调用shutdown()

**示例:**
```python
with get_coordinator() as coordinator:
    coordinator.registerModel(model, "test_model")
    # 自动清理资源
```

## 设备信息类和函数

### GPUInfo

GPU设备信息数据类

**属性:**
- `device_id`: 设备ID字符串
- `name`: GPU名称
- `memory_total`: 总内存（MB）
- `memory_free`: 可用内存（MB）
- `memory_used`: 已用内存（MB）
- `utilization`: 利用率百分比

### detect_gpu_devices() -> List[GPUInfo]
检测系统中所有可用的GPU设备

**返回:**
- GPUInfo对象列表

### get_optimal_device(gpu_list: List[GPUInfo]) -> str
从GPU列表中选择最优设备

**参数:**
- `gpu_list`: GPU信息列表

**返回:**
- 最优设备ID字符串

## 工具函数

### get_coordinator() -> AquaMLCoordinator
获取全局协调器实例

**返回:**
- AquaMLCoordinator单例实例

**示例:**
```python
coordinator = get_coordinator()
```

## 异常类

### AquaMLException
AquaML框架基础异常类

**示例:**
```python
try:
    coordinator.getModel("nonexistent_model")
except ValueError as e:
    print(f"模型不存在: {e}")
```

## 完整使用示例

```python
from AquaML.core.coordinator import get_coordinator
import torch.nn as nn

# 获取协调器
coordinator = get_coordinator()

# 初始化并设置设备
coordinator.initialize({"device": "cuda:0"})

# 注册模型
model = nn.Linear(10, 1)
coordinator.registerModel(model, "policy_network")

# 注册环境
@coordinator.registerEnv
class CartPoleEnv:
    def __init__(self):
        self.name = "CartPole-v1"
        
env = CartPoleEnv()

# 注册智能体
@coordinator.registerAgent
class DQNAgent:
    def __init__(self):
        self.name = "DQN"
        
agent = DQNAgent()

# 注册数据单元
@coordinator.registerDataUnit
class ExperienceBuffer:
    def __init__(self):
        self.name = "replay_buffer"
        
    def getUnitStatusDict(self):
        return {"size": 10000, "full": False}

buffer = ExperienceBuffer()

# 注册运行器
coordinator.registerRunner("experiment_1")

# 获取组件
model_info = coordinator.getModel("policy_network")
retrieved_env = coordinator.getEnv()
retrieved_agent = coordinator.getAgent()
retrieved_buffer = coordinator.getDataUnit("replay_buffer")

# 查看状态
status = coordinator.get_status()
print(f"系统状态: {status}")

# 设备管理
device_info = coordinator.get_device_info()
print(f"设备信息: {device_info}")

# 保存数据单元信息
coordinator.saveDataUnitInfo()

# 清理资源
coordinator.shutdown() 