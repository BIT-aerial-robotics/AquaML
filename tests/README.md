# AquaML 测试套件

本目录包含 AquaML 框架的完整测试套件，确保代码质量和功能正确性。

## 📁 目录结构

```
tests/
├── __init__.py                # 测试包初始化
├── conftest.py               # pytest 配置和共享 fixtures
├── pytest.ini               # pytest 配置文件
├── run_tests.py             # 测试运行脚本
├── test_coordinator.py      # 核心 coordinator 测试
└── README.md               # 本文档
```

## 🚀 快速开始

### 基本运行

```bash
# 使用测试脚本（推荐）
python tests/run_tests.py

# 如果有执行权限，也可以直接运行
./tests/run_tests.py

# 或者直接使用 pytest
pytest tests/
```

### 运行特定类型的测试

```bash
# 单元测试（默认）
python tests/run_tests.py --type unit

# 集成测试
python tests/run_tests.py --type integration

# 遗留 API 兼容性测试
python tests/run_tests.py --type legacy

# 所有测试
python tests/run_tests.py --type all
```

### 详细输出和调试

```bash
# 详细输出
python tests/run_tests.py --verbose
python tests/run_tests.py -v

# 快速测试（跳过慢速测试）
python tests/run_tests.py --quick
python tests/run_tests.py -q
```

### 性能优化

```bash
# 并行执行（加速测试）
python tests/run_tests.py --parallel
python tests/run_tests.py -p

# 组合使用：并行运行所有测试
python tests/run_tests.py --type all --parallel
```

### 代码覆盖率

```bash
# 生成覆盖率报告
python tests/run_tests.py --coverage
python tests/run_tests.py -c

# 生成覆盖率报告并详细输出
python tests/run_tests.py --coverage --verbose
```

### 常用组合命令

```bash
# 开发时的快速测试
python tests/run_tests.py --quick

# 完整测试带覆盖率
python tests/run_tests.py --type all --coverage --verbose

# 并行运行所有测试
python tests/run_tests.py --type all --parallel

# 最全面的测试（推荐 CI/CD 使用）
python tests/run_tests.py --type all --coverage --verbose --parallel
```

## 🧪 测试类型

### 单元测试 (Unit Tests)
- **标记**: `@pytest.mark.unit`
- **特点**: 测试单个组件的功能
- **优势**: 不依赖外部服务或文件系统，执行速度快
- **适用**: 日常开发和快速验证

### 集成测试 (Integration Tests)
- **标记**: `@pytest.mark.integration`
- **特点**: 测试多个组件的交互
- **注意**: 可能涉及文件系统或网络，执行时间相对较长
- **适用**: 完整功能验证

### 遗留 API 测试 (Legacy API Tests)
- **标记**: `@pytest.mark.legacy`
- **特点**: 测试向后兼容性
- **目的**: 确保旧代码仍然可以工作，包含弃用警告检查
- **适用**: 版本升级前的兼容性检查

## 📊 测试覆盖率

运行带覆盖率的测试后，可以查看报告：

```bash
# 生成 HTML 覆盖率报告
python tests/run_tests.py --coverage

# 查看 HTML 报告（在浏览器中打开）
# 报告位置：htmlcov/index.html

# 终端中也会显示覆盖率摘要
```

## 📋 全部命令行选项

```bash
python tests/run_tests.py [OPTIONS]

选项：
  -t, --type {unit,integration,legacy,all}
                        测试类型（默认：unit）
  -v, --verbose         详细输出
  -c, --coverage        启用覆盖率报告
  -p, --parallel        并行运行测试
  -q, --quick           快速测试（排除慢速测试）
  -h, --help            显示帮助信息
```

## 🔧 添加新测试

1. 在 `tests/` 目录下创建新的测试文件，文件名以 `test_` 开头
2. 使用 `unittest.TestCase` 或 `pytest` 风格编写测试
3. 添加适当的测试标记（`@pytest.mark.unit`、`@pytest.mark.integration` 等）
4. 在测试文件中添加详细的文档字符串

### 示例测试文件

```python
import unittest
import pytest
from AquaML import coordinator

class TestNewFeature(unittest.TestCase):
    """测试新功能的测试类"""
    
    def setUp(self):
        """每个测试前的设置"""
        self.coordinator = coordinator
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """测试基本功能"""
        # 测试代码
        pass
    
    @pytest.mark.integration
    def test_integration_with_other_components(self):
        """测试与其他组件的集成"""
        # 测试代码
        pass
    
    @pytest.mark.slow
    def test_performance_heavy_operation(self):
        """测试性能密集型操作（标记为慢速）"""
        # 长时间运行的测试代码
        pass
```

## 📋 测试检查清单

在提交代码前，请确保：

- [ ] 所有测试都通过：`python tests/run_tests.py --type all`
- [ ] 新功能有对应的测试
- [ ] 测试覆盖率保持在合理水平：`python tests/run_tests.py --coverage`
- [ ] 遗留 API 兼容性测试通过：`python tests/run_tests.py --type legacy`
- [ ] 测试文档完整且准确

## 🐛 故障排除

### 常见问题

1. **导入错误**: 确保在项目根目录运行测试
2. **权限错误**: 确保有足够的权限创建临时文件
3. **依赖缺失**: 检查是否安装了所有测试依赖（pytest、pytest-cov、pytest-xdist等）

### 调试技巧

```bash
# 只运行失败的测试
pytest tests/ --lf

# 进入 PDB 调试器
pytest tests/ --pdb

# 显示本地变量
pytest tests/ --tb=long

# 显示测试输出
pytest tests/ -s
```

### 性能优化建议

```bash
# 如果测试很慢，尝试：
python tests/run_tests.py --quick     # 跳过慢速测试
python tests/run_tests.py --parallel  # 并行执行

# 如果只想测试特定功能：
pytest tests/test_specific_module.py -v
```

## 🤝 贡献

欢迎贡献更多测试用例！请遵循以下原则：

1. **独立性**: 测试应该独立且可重复
2. **命名**: 使用描述性的测试名称
3. **文档**: 添加适当的文档和注释
4. **兼容性**: 确保测试在不同环境下都能通过
5. **标记**: 正确使用 pytest 标记（unit/integration/legacy/slow）

### 测试标记指南

```python
# 单元测试（快速，无外部依赖）
@pytest.mark.unit

# 集成测试（可能较慢，有外部依赖）
@pytest.mark.integration

# 遗留API测试（向后兼容性）
@pytest.mark.legacy

# 慢速测试（长时间运行）
@pytest.mark.slow
```

## 📚 相关文档

- [开发者文档](../Doc/Developer/README.md)
- [核心模块文档](../Doc/AquaML_Module_Documentation.md)
- [项目根目录 README](../README.md) 