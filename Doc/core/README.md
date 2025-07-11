# AquaML Core 模块使用教程

## 概述

AquaML Core模块是整个AquaML框架的核心基础设施，提供了组件管理、生命周期管理、注册系统等关键功能。该模块采用单例模式设计，确保全局只有一个协调器实例。

## 核心组件

### 1. AquaMLCoordinator (协调器)
- 框架的中央控制器
- 负责组件注册、配置管理和插件加载
- 实现单例模式，确保全局唯一性

### 2. ComponentRegistry (组件注册器)
- 中央化组件管理系统
- 支持组件注册、获取和元数据管理
- 提供类型检查和回调机制

### 3. LifecycleManager (生命周期管理器)
- 管理组件的启动和关闭流程
- 支持启动/关闭回调
- 跟踪组件状态

### 4. 异常系统
- 提供框架专用异常类
- 支持错误分类和处理

### 5. 设备管理系统
- 自动检测可用的GPU设备
- 智能设备选择和验证
- 支持CPU和GPU计算切换
- 提供设备信息查询接口

## 快速开始

### 基本使用模式

```python
from AquaML.core import AquaMLCoordinator, ComponentRegistry, LifecycleManager

# 获取协调器实例（单例模式）
coordinator = AquaMLCoordinator()

# 初始化框架
config = {
    "plugins": {
        "example_plugin": {
            "path": "plugins/example",
            "config": {"enabled": True}
        }
    }
}
coordinator.initialize(config)

# 使用上下文管理器
with AquaMLCoordinator() as coord:
    # 在这里使用协调器
    pass
```

### 组件注册

#### 方式1：使用装饰器注册
```python
from AquaML.core import AquaMLCoordinator

coordinator = AquaMLCoordinator()

# 注册环境
@coordinator.register_environment
class MyEnvironment:
    def __init__(self):
        self.name = "CustomEnvironment"
    
    def step(self, action):
        # 环境逻辑
        pass

# 注册智能体
@coordinator.register_agent
class MyAgent:
    def __init__(self):
        self.name = "CustomAgent"
    
    def act(self, observation):
        # 智能体逻辑
        pass

# 注册数据管理器
@coordinator.register_data_manager
class MyDataManager:
    def __init__(self):
        pass
    
    def save_data(self, data):
        # 数据保存逻辑
        pass
```

#### 方式2：直接注册到注册器
```python
from AquaML.core import ComponentRegistry

registry = ComponentRegistry()

# 注册组件
class MyModel:
    def __init__(self):
        self.name = "CustomModel"

model = MyModel()
registry.register(
    name="my_model",
    component=model,
    metadata={"version": "1.0", "type": "neural_network"}
)

# 获取组件
retrieved_model = registry.get("my_model")
```

### 生命周期管理

```python
from AquaML.core import LifecycleManager

lifecycle = LifecycleManager()

# 添加启动回调
def on_startup(config):
    print("系统启动中...")
    # 初始化数据库连接
    # 加载配置文件
    pass

def on_shutdown():
    print("系统关闭中...")
    # 保存状态
    # 关闭连接
    pass

lifecycle.add_startup_callback(on_startup)
lifecycle.add_shutdown_callback(on_shutdown)

# 初始化（会执行启动回调）
lifecycle.initialize({"database_url": "sqlite:///example.db"})

# 检查组件状态
lifecycle.set_component_state("model", "running")
if lifecycle.is_component_running("model"):
    print("模型正在运行")

# 关闭（会执行关闭回调）
lifecycle.shutdown()
```

### 设备管理

```python
from AquaML.core import AquaMLCoordinator

coordinator = AquaMLCoordinator()

# 初始化时指定设备
config = {
    "device": "cuda:0"  # 指定使用GPU 0，也可以是"cpu"
}
coordinator.initialize(config)

# 获取当前设备
current_device = coordinator.get_device()
print(f"当前设备: {current_device}")

# 获取PyTorch设备对象
torch_device = coordinator.get_torch_device()
print(f"PyTorch设备: {torch_device}")

# 获取所有可用设备
available_devices = coordinator.get_available_devices()
print(f"可用设备: {available_devices}")

# 设置设备
if coordinator.set_device("cuda:0"):
    print("成功设置为GPU 0")
else:
    print("设备设置失败")

# 验证设备
if coordinator.validate_device("cuda:0"):
    print("设备cuda:0可用")
else:
    print("设备cuda:0不可用")

# 检查GPU是否可用
if coordinator.is_gpu_available():
    print("GPU可用")
else:
    print("GPU不可用")

# 获取详细设备信息
device_info = coordinator.get_device_info()
print(f"设备信息: {device_info}")
```

## 高级用法

### 1. 组件元数据管理

```python
from AquaML.core import ComponentRegistry

registry = ComponentRegistry()

# 注册组件时添加元数据
registry.register(
    name="advanced_model",
    component=model,
    metadata={
        "version": "2.0",
        "author": "AI Team",
        "description": "高级神经网络模型",
        "requirements": ["tensorflow>=2.0", "numpy>=1.19"]
    }
)

# 获取元数据
metadata = registry.get_metadata("advanced_model")
print(f"模型版本: {metadata['version']}")
print(f"作者: {metadata['author']}")
```

### 2. 初始化回调

```python
from AquaML.core import ComponentRegistry

registry = ComponentRegistry()

# 定义回调函数
def on_model_registered(model):
    print(f"模型 {model.name} 已注册")
    # 执行模型验证
    model.validate()

# 添加回调
registry.add_initialization_callback("model", on_model_registered)

# 当注册名为"model"的组件时，回调会自动执行
registry.register("model", MyModel())
```

### 3. 组件查询和过滤

```python
from AquaML.core import ComponentRegistry

registry = ComponentRegistry()

# 列出所有组件
all_components = registry.list_components()
print(f"已注册组件: {all_components}")

# 按类型过滤
class BaseModel:
    pass

class MyModel(BaseModel):
    pass

registry.register("model1", MyModel())
models = registry.list_components(BaseModel)
print(f"模型组件: {models}")

# 检查组件是否存在
if "model1" in registry:
    model = registry["model1"]  # 支持字典语法
```

### 4. 错误处理

```python
from AquaML.core import (
    AquaMLException, 
    PluginError, 
    ConfigError,
    RegistryError,
    LifecycleError
)

try:
    # 尝试获取不存在的组件
    registry.get_strict("nonexistent_component")
except RegistryError as e:
    print(f"注册器错误: {e}")

try:
    # 尝试重复注册
    registry.register("existing_component", component)
except RegistryError as e:
    print(f"组件已存在: {e}")
```

## 最佳实践

### 1. 使用上下文管理器
```python
# 推荐：自动管理生命周期
with AquaMLCoordinator() as coordinator:
    coordinator.initialize(config)
    # 使用协调器
    # 退出时自动关闭
```

### 2. 组件命名规范
```python
# 推荐的命名方式
registry.register("environment.cartpole", env)
registry.register("agent.dqn", agent)
registry.register("model.neural_network", model)
```

### 3. 错误处理
```python
# 总是处理可能的异常
try:
    coordinator.initialize(config)
except AquaMLException as e:
    logger.error(f"初始化失败: {e}")
    # 执行清理工作
```

### 4. 生命周期管理
```python
# 在关闭时清理资源
def cleanup():
    # 保存模型状态
    # 关闭数据库连接
    # 清理临时文件
    pass

lifecycle.add_shutdown_callback(cleanup)
```

### 5. 设备管理最佳实践
```python
# 在模型训练前检查设备
coordinator = AquaMLCoordinator()
if coordinator.is_gpu_available():
    # 使用最优GPU设备
    coordinator.set_device("cuda:0")
else:
    # 回退到CPU
    coordinator.set_device("cpu")

# 验证设备可用性
device = coordinator.get_device()
if not coordinator.validate_device(device):
    coordinator.set_device("cpu")  # 回退方案

# 在模型中使用设备
model = MyModel()
model.to(coordinator.get_torch_device())
```

## 配置选项

### 协调器配置
```python
config = {
    "plugins": {
        "data_plugin": {
            "path": "plugins/data",
            "config": {
                "batch_size": 32,
                "buffer_size": 10000
            }
        }
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}

coordinator.initialize(config)
```

## 常见问题

### Q: 如何确保组件只被注册一次？
A: 使用 `replace=False` 参数（默认值），或者先检查组件是否存在：
```python
if not registry.has("my_component"):
    registry.register("my_component", component)
```

### Q: 如何在组件注册后执行特定操作？
A: 使用初始化回调：
```python
registry.add_initialization_callback("component_name", callback_function)
```

### Q: 如何处理组件初始化失败？
A: 使用try-catch结构和专门的异常类：
```python
try:
    coordinator.initialize(config)
except LifecycleError as e:
    # 处理生命周期错误
    pass
except ConfigError as e:
    # 处理配置错误
    pass
```

### Q: 如何查看系统当前状态？
A: 使用协调器的状态查询方法：
```python
status = coordinator.get_status()
print(f"系统状态: {status}")

# 查看所有组件状态
states = lifecycle.get_all_component_states()
print(f"组件状态: {states}")
```

### Q: 如何选择合适的计算设备？
A: 使用协调器的设备管理功能：
```python
# 自动选择最佳设备
coordinator.initialize()  # 会自动检测并选择最优设备

# 手动检查和选择
if coordinator.is_gpu_available():
    coordinator.set_device("cuda:0")
else:
    coordinator.set_device("cpu")
```

### Q: 如何处理设备不可用的情况？
A: 使用设备验证和回退策略：
```python
preferred_device = "cuda:0"
if coordinator.validate_device(preferred_device):
    coordinator.set_device(preferred_device)
else:
    print(f"设备 {preferred_device} 不可用，回退到CPU")
    coordinator.set_device("cpu")
```

### Q: 如何获取设备的详细信息？
A: 使用设备信息查询方法：
```python
device_info = coordinator.get_device_info()
print(f"当前设备: {device_info['current_device']}")
print(f"可用设备: {device_info['available_devices']}")
print(f"GPU数量: {device_info['gpu_count']}")
if device_info['gpu_available']:
    print(f"GPU详细信息: {device_info['gpu_details']}")
```

## 示例项目

完整的使用示例请参考：
- [基础示例](examples/basic_usage.py)
- [高级示例](examples/advanced_usage.py)
- [设备管理示例](examples/device_management.py)
- [插件开发示例](examples/plugin_development.py)

## 故障排除

### 设备相关问题

#### GPU无法识别
```python
# 检查GPU是否可用
coordinator = AquaMLCoordinator()
if not coordinator.is_gpu_available():
    print("GPU不可用，请检查：")
    print("1. 是否安装了CUDA")
    print("2. 是否安装了正确版本的PyTorch")
    print("3. 是否有足够的GPU内存")
```

#### 设备内存不足
```python
# 监控设备内存使用
device_info = coordinator.get_device_info()
if device_info['gpu_available']:
    # 检查GPU内存状态
    import torch
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
```

#### 设备切换失败
```python
# 安全的设备切换
def safe_device_switch(coordinator, target_device):
    if coordinator.validate_device(target_device):
        if coordinator.set_device(target_device):
            return True
        else:
            print(f"设备 {target_device} 验证成功但设置失败")
    else:
        print(f"设备 {target_device} 不可用")
    
    # 回退到CPU
    coordinator.set_device("cpu")
    return False
```

## 扩展阅读

- [插件开发指南](../plugins/README.md)
- [配置管理指南](../config/README.md)
- [API 参考文档](api_reference.md) 