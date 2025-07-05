# AquaML 测试指南

本文档为 AquaML 项目开发者提供完整的测试策略、工具使用和最佳实践指南。

## 📋 目录

- [快速开始](#快速开始)
- [测试策略](#测试策略)
- [测试工具](#测试工具)
- [测试分类](#测试分类)
- [编写测试](#编写测试)
- [CI/CD 集成](#cicd-集成)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 立即运行测试

```bash
# 进入项目根目录
cd /path/to/AquaML

# 运行默认测试（单元测试）
python tests/run_tests.py

# 运行所有测试
python tests/run_tests.py --type all --verbose
```

### 开发工作流

```bash
# 1. 开发时的快速验证
python tests/run_tests.py --quick

# 2. 提交前的完整检查
python tests/run_tests.py --type all --coverage

# 3. 性能优化版本
python tests/run_tests.py --type all --parallel --coverage
```

## 📊 测试策略

### 测试金字塔

```
        🔺 E2E Tests (少量)
       🔺🔺 Integration Tests (适量)
    🔺🔺🔺🔺 Unit Tests (大量)
```

#### 单元测试 (70%)
- **目标**: 快速反馈，高覆盖率
- **特点**: 独立、快速、稳定
- **范围**: 单个函数、方法、类

#### 集成测试 (20%)
- **目标**: 验证组件间交互
- **特点**: 涉及多个模块
- **范围**: 组件接口、数据流

#### 端到端测试 (10%)
- **目标**: 验证完整业务流程
- **特点**: 全链路测试
- **范围**: 用户场景、API 流程

## 🛠️ 测试工具

### 核心工具栈

| 工具 | 用途 | 配置文件 |
|------|------|----------|
| **pytest** | 测试框架 | `pytest.ini` |
| **pytest-cov** | 覆盖率报告 | `.coveragerc` |
| **pytest-xdist** | 并行执行 | - |
| **unittest** | 标准测试库 | - |

### 自定义测试脚本

```bash
# 主要测试脚本
tests/run_tests.py

# 支持的参数
--type {unit,integration,legacy,all}  # 测试类型
--verbose                             # 详细输出
--coverage                            # 覆盖率报告
--parallel                            # 并行执行
--quick                               # 快速测试
```

## 🏷️ 测试分类

### 按功能分类

```python
# 单元测试
@pytest.mark.unit
def test_component_basic_functionality():
    pass

# 集成测试
@pytest.mark.integration
def test_components_interaction():
    pass

# 遗留API测试
@pytest.mark.legacy
def test_backward_compatibility():
    pass

# 慢速测试
@pytest.mark.slow
def test_performance_heavy_operation():
    pass
```

### 按模块分类

```
tests/
├── test_coordinator.py      # 协调器测试
├── test_registry.py         # 注册器测试
├── test_lifecycle.py        # 生命周期测试
├── test_plugins.py          # 插件系统测试
└── test_integration.py      # 集成测试
```

## 📝 编写测试

### 测试文件结构

```python
#!/usr/bin/env python3
"""
模块名测试

测试 AquaML 的 XXX 功能
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock

from AquaML.core import ComponentRegistry
from AquaML.exceptions import AquaMLException


class TestComponentRegistry(unittest.TestCase):
    """组件注册器测试类"""
    
    def setUp(self):
        """每个测试前的初始化"""
        self.registry = ComponentRegistry()
    
    def tearDown(self):
        """每个测试后的清理"""
        self.registry.clear()
    
    @pytest.mark.unit
    def test_register_component_success(self):
        """测试成功注册组件"""
        # Arrange
        component = Mock()
        component.name = "test_component"
        
        # Act
        self.registry.register("test", component)
        
        # Assert
        self.assertEqual(self.registry.get("test"), component)
    
    @pytest.mark.unit
    def test_register_duplicate_component_raises_error(self):
        """测试重复注册组件抛出异常"""
        # Arrange
        component = Mock()
        self.registry.register("test", component)
        
        # Act & Assert
        with self.assertRaises(AquaMLException):
            self.registry.register("test", component)
    
    @pytest.mark.integration
    def test_registry_lifecycle_integration(self):
        """测试注册器与生命周期管理的集成"""
        # 集成测试代码
        pass
    
    @pytest.mark.slow
    def test_registry_performance_with_many_components(self):
        """测试大量组件的性能"""
        # 性能测试代码
        pass


class TestComponentRegistryEdgeCases(unittest.TestCase):
    """边缘情况测试"""
    
    @pytest.mark.unit
    def test_get_nonexistent_component_returns_none(self):
        """测试获取不存在的组件返回None"""
        registry = ComponentRegistry()
        self.assertIsNone(registry.get("nonexistent"))
```

### 测试命名规范

```python
# 好的测试名称
def test_register_component_with_valid_name_succeeds():
    """测试使用有效名称注册组件成功"""
    pass

def test_register_component_with_empty_name_raises_value_error():
    """测试使用空名称注册组件抛出ValueError"""
    pass

def test_get_component_after_registration_returns_correct_instance():
    """测试注册后获取组件返回正确实例"""
    pass

# 避免的测试名称
def test_register():  # 太模糊
def test_component():  # 不描述具体行为
def test_error():  # 不明确什么错误
```

### Mock 和 Patch 使用

```python
from unittest.mock import Mock, patch, MagicMock

class TestAdvancedMocking(unittest.TestCase):
    
    def test_with_mock_object(self):
        """使用Mock对象测试"""
        # 创建Mock对象
        mock_component = Mock()
        mock_component.start.return_value = True
        mock_component.name = "test"
        
        # 测试
        registry = ComponentRegistry()
        registry.register("test", mock_component)
        
        # 验证
        mock_component.start.assert_called_once()
    
    @patch('AquaML.core.ComponentRegistry.validate_component')
    def test_with_patch_decorator(self, mock_validate):
        """使用patch装饰器测试"""
        mock_validate.return_value = True
        
        registry = ComponentRegistry()
        registry.register("test", Mock())
        
        mock_validate.assert_called_once()
    
    def test_with_context_manager(self):
        """使用上下文管理器测试"""
        with patch('AquaML.core.ComponentRegistry.validate_component') as mock_validate:
            mock_validate.return_value = True
            
            registry = ComponentRegistry()
            registry.register("test", Mock())
            
            mock_validate.assert_called_once()
```

## 🔄 CI/CD 集成

### GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        python tests/run_tests.py --type all --coverage --parallel
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

### 本地 Pre-commit 钩子

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: python tests/run_tests.py --quick
        language: system
        pass_filenames: false
        always_run: true
```

## ⚡ 性能优化

### 并行测试执行

```bash
# 自动检测CPU核心数
python tests/run_tests.py --parallel

# 手动指定进程数
pytest tests/ -n 4
```

### 测试数据优化

```python
# 使用 pytest fixtures 共享数据
@pytest.fixture(scope="session")
def large_dataset():
    """会话级别的大数据集"""
    return generate_large_dataset()

@pytest.fixture(scope="module")  
def module_config():
    """模块级别的配置"""
    return load_config()

@pytest.fixture(scope="function")
def fresh_registry():
    """每个测试都创建新的注册器"""
    return ComponentRegistry()
```

### 跳过慢速测试

```bash
# 开发时跳过慢速测试
python tests/run_tests.py --quick

# 只运行快速测试
pytest tests/ -m "not slow"
```

## 🐛 故障排除

### 常见问题解决方案

#### 1. 导入错误

```bash
# 问题：ModuleNotFoundError: No module named 'AquaML'
# 解决：确保在项目根目录运行
cd /path/to/AquaML
python tests/run_tests.py

# 或者设置 PYTHONPATH
export PYTHONPATH=/path/to/AquaML:$PYTHONPATH
```

#### 2. 权限问题

```bash
# 问题：PermissionError: [Errno 13] Permission denied
# 解决：检查文件权限
chmod +x tests/run_tests.py
```

#### 3. 依赖缺失

```bash
# 问题：ImportError: No module named 'pytest'
# 解决：安装测试依赖
pip install -r requirements-dev.txt
```

### 调试技巧

```bash
# 1. 详细错误信息
pytest tests/ --tb=long

# 2. 进入调试器
pytest tests/ --pdb

# 3. 只运行失败的测试
pytest tests/ --lf

# 4. 显示打印输出
pytest tests/ -s

# 5. 运行特定测试
pytest tests/test_coordinator.py::TestCoordinator::test_register_component
```

## 📚 最佳实践

### 1. 测试组织原则

```
✅ 好的做法：
- 一个测试类对应一个被测试类
- 测试方法名清晰描述行为
- 每个测试只验证一个行为
- 使用setUp/tearDown管理测试状态

❌ 避免的做法：
- 测试之间相互依赖
- 测试中包含复杂的业务逻辑
- 忽略边界条件和异常情况
- 测试名称含糊不清
```

### 2. 断言策略

```python
# 具体的断言
self.assertEqual(actual, expected)
self.assertIsNone(result)
self.assertIn(item, collection)
self.assertRaises(SpecificException)

# 避免宽泛的断言
self.assertTrue(condition)  # 不如 self.assertEqual(actual, expected)
self.assertFalse(condition)  # 不如 self.assertIsNone(result)
```

### 3. 测试数据管理

```python
# 使用工厂函数创建测试数据
def create_test_component(name="test", **kwargs):
    """创建测试组件"""
    component = Mock()
    component.name = name
    component.configure(**kwargs)
    return component

# 使用常量定义测试数据
TEST_COMPONENT_CONFIG = {
    "name": "test_component",
    "type": "processor",
    "settings": {"batch_size": 32}
}
```

### 4. 测试覆盖率目标

```bash
# 推荐覆盖率目标
- 核心模块: 90%+
- 工具模块: 80%+
- 示例代码: 70%+
- 总体覆盖率: 85%+

# 检查覆盖率
python tests/run_tests.py --coverage
```

### 5. 持续改进

```python
# 定期回顾测试质量
- 测试是否能捕获回归问题？
- 测试是否容易理解和维护？
- 测试是否执行得足够快？
- 测试是否覆盖了关键路径？
```

## 📖 参考资源

### 官方文档
- [pytest 文档](https://docs.pytest.org/)
- [unittest 文档](https://docs.python.org/3/library/unittest.html)
- [unittest.mock 文档](https://docs.python.org/3/library/unittest.mock.html)

### 内部文档
- [测试详细文档](../../tests/README.md)
- [开发者指南](./README.md)
- [API 文档](../AquaML_Module_Documentation.md)

### 相关工具
- [coverage.py](https://coverage.readthedocs.io/)
- [pytest-xdist](https://pytest-xdist.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)

---

**最后更新**: 2024年12月
**维护者**: AquaML 开发团队
**版本**: 1.0.0 