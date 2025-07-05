# AquaML Core 模块 API 参考文档

## 模块导入

```python
from AquaML.core import (
    AquaMLCoordinator,
    ComponentRegistry,
    LifecycleManager,
    AquaMLException,
    PluginError,
    ConfigError,
    RegistryError,
    LifecycleError
)
```

## AquaMLCoordinator

### 类描述
主要协调器类，采用单例模式，负责管理整个AquaML框架的生命周期和组件。

### 构造函数
```python
AquaMLCoordinator()
```
使用单例模式，多次调用返回同一实例。

### 方法

#### `initialize(config: Optional[Dict[str, Any]] = None) -> None`
初始化协调器

**参数:**
- `config`: 配置字典，可选

**异常:**
- `AquaMLException`: 初始化失败时抛出

**示例:**
```python
coordinator = AquaMLCoordinator()
coordinator.initialize({
    "plugins": {"example": {"path": "plugins/example"}}
})
```

#### `shutdown() -> None`
关闭协调器，清理所有资源

**示例:**
```python
coordinator.shutdown()
```

#### `register_environment(env_cls) -> Callable`
注册环境类的装饰器

**参数:**
- `env_cls`: 环境类

**返回:**
- 装饰器函数

**示例:**
```python
@coordinator.register_environment
class MyEnvironment:
    def __init__(self):
        self.name = "CustomEnv"
```

#### `register_agent(agent_cls) -> Callable`
注册智能体类的装饰器

**参数:**
- `agent_cls`: 智能体类

**返回:**
- 装饰器函数

**示例:**
```python
@coordinator.register_agent
class MyAgent:
    def __init__(self):
        self.name = "CustomAgent"
```

#### `register_data_manager(data_manager_cls) -> Callable`
注册数据管理器类的装饰器

**参数:**
- `data_manager_cls`: 数据管理器类

**返回:**
- 装饰器函数

#### `get_environment() -> Any`
获取已注册的环境实例

**返回:**
- 环境实例，如果未注册则返回None

#### `get_agent() -> Any`
获取已注册的智能体实例

**返回:**
- 智能体实例，如果未注册则返回None

#### `get_data_manager() -> Any`
获取已注册的数据管理器实例

**返回:**
- 数据管理器实例，如果未注册则返回None

#### `get_status() -> Dict[str, Any]`
获取协调器状态信息

**返回:**
- 包含状态信息的字典

#### `is_component_registered(component_name: str) -> bool`
检查组件是否已注册

**参数:**
- `component_name`: 组件名称

**返回:**
- 布尔值，表示组件是否已注册

#### `get_component_state(component_name: str) -> Optional[str]`
获取组件状态

**参数:**
- `component_name`: 组件名称

**返回:**
- 组件状态字符串，如果不存在返回None

#### `list_components() -> List[str]`
列出所有已注册的组件

**返回:**
- 组件名称列表

### 上下文管理器支持

```python
with AquaMLCoordinator() as coordinator:
    coordinator.initialize(config)
    # 使用协调器
    # 退出时自动关闭
```

### 兼容性方法

以下方法保持向后兼容：

#### `registerModel(model, model_name: str)`
注册模型

#### `registerEnv(env_cls)`
注册环境

#### `registerAgent(agent_cls)`
注册智能体

#### `getModel(model_name: str)`
获取模型

#### `getEnv()`
获取环境

#### `getAgent()`
获取智能体

---

## ComponentRegistry

### 类描述
组件注册系统，提供集中式的组件管理功能。

### 构造函数
```python
ComponentRegistry()
```

### 方法

#### `register(name: str, component: Any, component_type: Optional[Type] = None, metadata: Optional[Dict[str, Any]] = None, replace: bool = False) -> None`
注册组件

**参数:**
- `name`: 组件名称
- `component`: 组件实例
- `component_type`: 组件类型，可选
- `metadata`: 元数据字典，可选
- `replace`: 是否替换已存在的组件，默认False

**异常:**
- `RegistryError`: 组件已存在且replace=False时抛出

**示例:**
```python
registry.register(
    name="my_model",
    component=model,
    metadata={"version": "1.0"}
)
```

#### `get(name: str, default: Any = None) -> Any`
获取组件（安全模式）

**参数:**
- `name`: 组件名称
- `default`: 默认值，如果组件不存在返回此值

**返回:**
- 组件实例或默认值

#### `get_strict(name: str) -> Any`
获取组件（严格模式）

**参数:**
- `name`: 组件名称

**返回:**
- 组件实例

**异常:**
- `RegistryError`: 组件不存在时抛出

#### `unregister(name: str) -> None`
取消注册组件

**参数:**
- `name`: 组件名称

#### `has(name: str) -> bool`
检查组件是否存在

**参数:**
- `name`: 组件名称

**返回:**
- 布尔值

#### `list_components(component_type: Optional[Type] = None) -> List[str]`
列出组件

**参数:**
- `component_type`: 组件类型过滤器，可选

**返回:**
- 组件名称列表

#### `get_metadata(name: str) -> Dict[str, Any]`
获取组件元数据

**参数:**
- `name`: 组件名称

**返回:**
- 元数据字典

#### `set_metadata(name: str, metadata: Dict[str, Any]) -> None`
设置组件元数据

**参数:**
- `name`: 组件名称
- `metadata`: 元数据字典

**异常:**
- `RegistryError`: 组件不存在时抛出

#### `add_initialization_callback(component_name: str, callback: Callable) -> None`
添加初始化回调

**参数:**
- `component_name`: 组件名称
- `callback`: 回调函数

#### `clear() -> None`
清除所有注册的组件

### 特殊方法

#### `__len__() -> int`
返回注册的组件数量

#### `__contains__(name: str) -> bool`
支持 `in` 操作符检查组件是否存在

#### `__iter__()`
支持迭代器，遍历组件名称

### 全局函数

#### `get_global_registry() -> ComponentRegistry`
获取全局注册器实例

#### `register_component(name: str, component: Any, **kwargs) -> None`
向全局注册器注册组件

---

## LifecycleManager

### 类描述
生命周期管理器，负责管理组件的启动和关闭流程。

### 构造函数
```python
LifecycleManager()
```

### 方法

#### `initialize(config: Optional[Dict[str, Any]] = None) -> None`
初始化生命周期管理器

**参数:**
- `config`: 配置字典，可选

**异常:**
- `LifecycleError`: 初始化失败时抛出

#### `shutdown() -> None`
关闭生命周期管理器

#### `add_startup_callback(callback: Callable) -> None`
添加启动回调

**参数:**
- `callback`: 回调函数，接受config参数

**异常:**
- `LifecycleError`: 回调不可调用时抛出

#### `add_shutdown_callback(callback: Callable) -> None`
添加关闭回调

**参数:**
- `callback`: 回调函数

**异常:**
- `LifecycleError`: 回调不可调用时抛出

#### `remove_startup_callback(callback: Callable) -> None`
移除启动回调

**参数:**
- `callback`: 要移除的回调函数

#### `remove_shutdown_callback(callback: Callable) -> None`
移除关闭回调

**参数:**
- `callback`: 要移除的回调函数

#### `set_component_state(component_name: str, state: str) -> None`
设置组件状态

**参数:**
- `component_name`: 组件名称
- `state`: 状态字符串（如 'initializing', 'running', 'stopped'）

#### `get_component_state(component_name: str) -> Optional[str]`
获取组件状态

**参数:**
- `component_name`: 组件名称

**返回:**
- 状态字符串，如果不存在返回None

#### `is_component_running(component_name: str) -> bool`
检查组件是否运行中

**参数:**
- `component_name`: 组件名称

**返回:**
- 布尔值

#### `get_all_component_states() -> Dict[str, str]`
获取所有组件状态

**返回:**
- 组件名称到状态的映射字典

### 属性

#### `is_initialized: bool`
只读属性，表示是否已初始化

### 上下文管理器支持

```python
with LifecycleManager() as lifecycle:
    # 自动初始化
    # 使用生命周期管理器
    # 退出时自动关闭
```

### 全局函数

#### `get_global_lifecycle_manager() -> LifecycleManager`
获取全局生命周期管理器实例

#### `add_startup_callback(callback: Callable) -> None`
向全局生命周期管理器添加启动回调

#### `add_shutdown_callback(callback: Callable) -> None`
向全局生命周期管理器添加关闭回调

---

## 异常类

### AquaMLException
```python
class AquaMLException(Exception)
```
AquaML框架的基础异常类

### PluginError
```python
class PluginError(AquaMLException)
```
插件相关错误

### ConfigError
```python
class ConfigError(AquaMLException)
```
配置相关错误

### RegistryError
```python
class RegistryError(AquaMLException)
```
组件注册器相关错误

### LifecycleError
```python
class LifecycleError(AquaMLException)
```
生命周期管理相关错误

### EnvironmentError
```python
class EnvironmentError(AquaMLException)
```
环境相关错误

### LearningError
```python
class LearningError(AquaMLException)
```
学习算法相关错误

---

## 使用模式

### 单例模式
`AquaMLCoordinator` 使用单例模式，确保全局只有一个实例：

```python
coord1 = AquaMLCoordinator()
coord2 = AquaMLCoordinator()
assert coord1 is coord2  # True
```

### 装饰器模式
协调器提供装饰器方式注册组件：

```python
@coordinator.register_environment
class MyEnvironment:
    pass
```

### 上下文管理器模式
支持`with`语句自动管理生命周期：

```python
with AquaMLCoordinator() as coord:
    coord.initialize(config)
    # 使用协调器
    # 退出时自动关闭
```

### 回调模式
支持启动和关闭回调：

```python
def on_startup(config):
    print("系统启动")

def on_shutdown():
    print("系统关闭")

lifecycle.add_startup_callback(on_startup)
lifecycle.add_shutdown_callback(on_shutdown)
```

---

## 版本兼容性

当前API版本：1.0.0

向后兼容的方法：
- `registerModel()` → 推荐使用 `register_model()`
- `registerEnv()` → 推荐使用 `register_environment()`
- `registerAgent()` → 推荐使用 `register_agent()`
- `getModel()` → 推荐使用 `get_model()`
- `getEnv()` → 推荐使用 `get_environment()`
- `getAgent()` → 推荐使用 `get_agent()` 