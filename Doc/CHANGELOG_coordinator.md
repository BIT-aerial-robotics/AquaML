# AquaML Coordinator 更新日志

## 版本 2.0.0 - 简化协调器架构

### 主要变更

#### ✅ 架构简化
- **移除复杂系统**: 删除了 `ComponentRegistry` 和 `LifecycleManager`
- **直接存储**: 组件现在直接存储在协调器的属性中，便于访问和调试
- **简化API**: 统一的注册和获取方法，易于理解和使用

#### ✅ 组件管理
- **模型管理**: `registerModel()`, `getModel()` - 注册和获取机器学习模型
- **环境管理**: `registerEnv()`, `getEnv()` - 注册和获取强化学习环境  
- **智能体管理**: `registerAgent()`, `getAgent()` - 注册和获取强化学习智能体
- **数据单元管理**: `registerDataUnit()`, `getDataUnit()` - 注册和获取数据单元
- **文件系统管理**: `registerFileSystem()`, `getFileSystem()` - 注册和获取文件系统
- **通信器管理**: `registerCommunicator()`, `getCommunicator()` - 注册和获取通信器
- **运行器管理**: `registerRunner()`, `getRunner()` - 注册和获取运行器名称

#### ✅ 设备管理保留
- **自动检测**: GPU/CPU设备自动检测和选择
- **设备管理**: `set_device()`, `get_device()`, `validate_device()` 等方法
- **设备信息**: 详细的设备信息获取和状态报告

#### ✅ 错误处理
- **类型安全**: 明确的错误处理和异常抛出
- **重复检查**: 防止重复注册同类型组件
- **存在验证**: 获取不存在组件时抛出清晰的错误信息

### 迁移指南

#### 旧版本使用方式:
```python
# 旧版本 - 复杂的注册系统
coordinator.registry.register('my_component', component)
component = coordinator.registry.get('my_component')
```

#### 新版本使用方式:
```python
# 新版本 - 直接注册和访问
coordinator.registerModel(model, "my_model")
model_info = coordinator.getModel("my_model")
```

### 兼容性说明
- **保持兼容**: 原有的 `registerModel()`, `registerEnv()`, `registerAgent()` 等方法继续可用
- **设备管理**: 所有设备管理功能保持不变
- **单例模式**: 协调器仍使用单例模式，确保全局唯一实例

### 测试更新
- **全面测试**: 更新了所有测试用例以适应新架构
- **模拟组件**: 添加了完整的模拟组件用于测试
- **错误场景**: 测试了各种错误和边界情况
- **设备管理**: 包含完整的设备管理功能测试

### 文档更新
- **中文文档**: 更新了 `AquaML_模块文档.md` 中的核心模块部分
- **英文文档**: 更新了 `AquaML_Module_Documentation.md` 中的核心模块部分
- **API参考**: 完全重写了 `core/api_reference.md` 以反映新的API设计
- **使用示例**: 提供了完整的使用示例和最佳实践

### 优势

1. **更简单**: 移除了复杂的中间层，直接存储和访问组件
2. **更直观**: 组件注册和获取逻辑清晰明了
3. **更易调试**: 组件直接存储在协调器属性中，容易查找和调试
4. **更高效**: 减少了不必要的抽象层，提高了性能
5. **保持功能**: 保留了所有重要功能，特别是设备管理

### 未来规划
- 考虑添加组件依赖管理
- 可能增加组件生命周期回调（可选）
- 继续优化设备管理功能
- 支持更多的组件类型和注册方式

---

更新日期: 2024年12月
维护者: AquaML开发团队 