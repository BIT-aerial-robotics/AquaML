# AquaML Manager System

这个文档描述了AquaML框架中新的管理器系统架构。

## 概述

AquaML协调器（coordinator）已经被重新设计，将各个组件的管理拆分成独立的专门管理器（manager）。每个管理器负责特定类型组件的注册、获取、状态管理等功能。

## 架构设计

### 核心组件

1. **AquaMLCoordinator** - 主协调器，提供统一的接口
2. **专门管理器** - 每个组件类型都有对应的管理器

### 管理器列表

- **ModelManager** - 模型管理器，处理模型的注册和获取
- **EnvironmentManager** - 环境管理器，管理环境实例
- **AgentManager** - 代理管理器，管理智能体实例
- **DataUnitManager** - 数据单元管理器，管理数据单元
- **FileSystemManager** - 文件系统管理器，管理文件系统实例
- **CommunicatorManager** - 通信器管理器，管理通信器实例
- **DataManager** - 数据管理器，管理数据管理器实例
- **RunnerManager** - 运行器管理器，管理运行器名称

## 使用方法

### 1. 通过协调器接口使用（推荐）

```python
from AquaML.core.coordinator import get_coordinator

# 获取协调器实例
coordinator = get_coordinator()

# 注册模型
coordinator.registerModel(my_model, "my_model_name")

# 获取模型
model_info = coordinator.getModel("my_model_name")

# 注册环境
@coordinator.registerEnv
class MyEnvironment:
    def __init__(self):
        self.name = "my_env"

# 获取环境
env = coordinator.getEnv()
```

### 2. 直接使用管理器

```python
from AquaML.core.coordinator import get_coordinator

coordinator = get_coordinator()

# 获取模型管理器
model_manager = coordinator.get_model_manager()

# 直接使用管理器方法
model_manager.register_model(my_model, "my_model_name")
model_instance = model_manager.get_model_instance("my_model_name")

# 检查模型是否存在
if model_manager.model_exists("my_model_name"):
    print("Model exists!")

# 列出所有模型
models = model_manager.list_models()
print(f"Registered models: {models}")
```

### 3. 管理器状态查询

```python
# 获取单个管理器状态
model_status = coordinator.get_model_manager().get_status()
print(f"Model manager status: {model_status}")

# 获取所有管理器状态
all_status = coordinator.get_all_managers_status()
print(f"All managers status: {all_status}")

# 获取组件列表
components = coordinator.list_components()
print(f"Components: {components}")
```

## 管理器功能对比

### 旧版本（直接在coordinator中）
```python
# 所有逻辑都在coordinator中
def registerModel(self, model, model_name):
    # 检查、注册、存储逻辑都在这里
    if model_name in self.models_dict_:
        raise ValueError("Model already exists")
    self.models_dict_[model_name] = {"model": model, "status": {}}
```

### 新版本（使用专门管理器）
```python
# coordinator只提供接口
def registerModel(self, model, model_name):
    self.model_manager.register_model(model, model_name)

# 实际逻辑在专门的管理器中
class ModelManager:
    def register_model(self, model, model_name):
        # 专门的模型管理逻辑
        # 更好的错误处理、状态管理、扩展性
```

## 优势

1. **模块化** - 每个管理器专注于特定功能
2. **可扩展性** - 容易添加新的管理器或扩展现有功能
3. **可测试性** - 每个管理器可以独立测试
4. **代码复用** - 管理器可以在其他地方重用
5. **清晰的职责分离** - 协调器负责协调，管理器负责具体管理

## 兼容性

新版本保持了与旧版本的API兼容性。所有原有的coordinator方法仍然可用，但内部实现已经改为使用专门的管理器。

## 扩展示例

### 添加新的管理器

```python
# 1. 创建新的管理器
class CustomManager:
    def __init__(self):
        self.items = {}
    
    def register_item(self, item, name):
        self.items[name] = item
    
    def get_item(self, name):
        return self.items[name]

# 2. 在coordinator中添加
class AquaMLCoordinator:
    def __init__(self):
        # ... 其他管理器
        self.custom_manager = CustomManager()
    
    def registerCustom(self, item, name):
        self.custom_manager.register_item(item, name)
```

## 迁移指南

如果你的代码直接访问了coordinator的内部属性（如`models_dict_`），你需要改为使用管理器接口：

```python
# 旧方式
models_dict = coordinator.models_dict_

# 新方式
model_manager = coordinator.get_model_manager()
models_list = model_manager.list_models()
```

这种新的架构提供了更好的组织结构和扩展性，同时保持了易用性和兼容性。

## 测试

新的管理器系统包含了完整的测试套件，位于 `tests/test_manager_system.py`。

### 测试覆盖范围

- **单元测试** - 每个管理器的独立功能测试
- **集成测试** - 管理器之间的交互测试
- **错误处理测试** - 异常情况的处理验证

### 运行测试

```bash
# 运行所有管理器系统测试
python -m pytest tests/test_manager_system.py -v

# 运行特定测试类
python -m pytest tests/test_manager_system.py::TestModelManager -v

# 运行特定测试方法
python -m pytest tests/test_manager_system.py::TestModelManager::test_model_registration -v
```

### 测试结构

每个管理器都有对应的测试类：

- `TestModelManager` - 模型管理器测试
- `TestEnvironmentManager` - 环境管理器测试  
- `TestAgentManager` - 代理管理器测试
- `TestDataUnitManager` - 数据单元管理器测试
- `TestFileSystemManager` - 文件系统管理器测试
- `TestRunnerManager` - 运行器管理器测试
- `TestManagerIntegration` - 集成测试
- `TestErrorHandling` - 错误处理测试

## 性能优化

新的管理器系统在性能方面的改进：

1. **内存使用** - 每个管理器独立管理自己的数据，避免大型字典的性能问题
2. **查找效率** - 直接访问对应管理器，减少查找时间
3. **并发安全** - 每个管理器可以独立加锁，提高并发性能
4. **可扩展性** - 新的管理器可以独立开发和优化

## 最佳实践

1. **使用协调器接口** - 尽量通过协调器接口访问功能，而不是直接使用管理器
2. **错误处理** - 适当处理 `ValueError` 异常，这是管理器系统的主要异常类型
3. **状态检查** - 使用 `*_exists()` 方法在访问组件前检查其是否存在
4. **清理资源** - 在应用结束时调用 `coordinator.shutdown()` 清理资源 