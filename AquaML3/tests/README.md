# AquaML 数据收集器测试套件

本目录包含 AquaML 数据收集器模块的完整测试套件，包括单元测试、集成测试和性能测试。

## 目录结构

```
AquaML/tests/
├── unit/                    # 单元测试
│   ├── __init__.py
│   ├── test_base_collector.py          # BaseCollector 测试
│   ├── test_rl_collector.py            # RLCollector 测试
│   ├── test_trajectory_collector.py    # TrajectoryCollector 测试
│   ├── test_buffer_collector.py        # BufferCollector 测试
│   └── test_collector_utils.py         # 工具类测试
├── integration/             # 集成测试
│   ├── __init__.py
│   └── test_collector_integration.py   # 收集器集成测试
├── performance/             # 性能测试
│   ├── __init__.py
│   └── test_collector_performance.py   # 性能基准测试
├── conftest.py             # 共享测试配置和 fixtures
├── pytest.ini             # pytest 配置文件
├── requirements.txt        # 测试依赖包
├── run_tests.py           # 测试运行脚本
└── README.md              # 本文档
```

## 快速开始

### 1. 安装测试依赖

```bash
pip install -r AquaML/tests/requirements.txt
```

### 2. 运行基本测试

```bash
# 运行所有单元测试（默认）
python AquaML/tests/run_tests.py

# 或者直接使用 pytest
cd AquaML && python -m pytest tests/unit/
```

### 3. 运行不同类型的测试

```bash
# 单元测试
python AquaML/tests/run_tests.py --unit

# 集成测试
python AquaML/tests/run_tests.py --integration

# 性能测试
python AquaML/tests/run_tests.py --performance

# 所有测试
python AquaML/tests/run_tests.py --all

# 快速测试（排除慢速测试）
python AquaML/tests/run_tests.py --fast
```

## 详细使用说明

### 测试运行选项

#### 基本选项

```bash
# 详细输出
python AquaML/tests/run_tests.py --verbose

# 生成覆盖率报告
python AquaML/tests/run_tests.py --coverage

# 并行运行（使用 4 个进程）
python AquaML/tests/run_tests.py --parallel 4

# 运行特定测试
python AquaML/tests/run_tests.py --test unit/test_base_collector.py

# 运行包含特定关键词的测试
python AquaML/tests/run_tests.py --keyword "initialization"
```

#### 输出格式

```bash
# JUnit XML 报告
python AquaML/tests/run_tests.py --output junit

# HTML 报告
python AquaML/tests/run_tests.py --output html
```

### 直接使用 pytest

如果你更喜欢直接使用 pytest：

```bash
cd AquaML

# 基本运行
python -m pytest tests/

# 详细输出
python -m pytest tests/ -v

# 覆盖率报告
python -m pytest tests/ --cov=data.collectors --cov-report=html

# 并行运行
python -m pytest tests/ -n 4

# 运行特定测试类
python -m pytest tests/unit/test_base_collector.py::TestBaseCollector

# 运行特定测试方法
python -m pytest tests/unit/test_base_collector.py::TestBaseCollector::test_initialization

# 使用标记过滤
python -m pytest tests/ -m "not slow"  # 排除慢速测试
python -m pytest tests/ -m "integration"  # 只运行集成测试
```

## 测试类型说明

### 单元测试 (Unit Tests)

位于 `tests/unit/` 目录，测试各个收集器类的独立功能：

- **test_base_collector.py**: 测试 BaseCollector 基础功能
  - 配置初始化
  - 数据缓冲
  - 保存/加载
  - 数据验证

- **test_rl_collector.py**: 测试 RLCollector 强化学习功能
  - 步骤收集
  - 轨迹管理
  - 异步收集
  - 数据导出

- **test_trajectory_collector.py**: 测试 TrajectoryCollector 轨迹功能
  - 轨迹收集
  - 过滤机制
  - 统计分析

- **test_buffer_collector.py**: 测试 BufferCollector 内存管理
  - 内存监控
  - 自动刷新
  - 文件合并

- **test_collector_utils.py**: 测试工具类
  - 数据缓冲区
  - 轨迹缓冲区
  - 工具函数

### 集成测试 (Integration Tests)

位于 `tests/integration/` 目录，测试收集器与环境的集成：

- 环境交互
- 多收集器协作
- 数据一致性
- 实际使用场景

### 性能测试 (Performance Tests)

位于 `tests/performance/` 目录，测试收集器的性能表现：

- 吞吐量测试
- 内存使用测试
- 大规模数据收集
- 并发性能

## 测试配置

### pytest.ini

配置文件包含：
- 测试发现规则
- 标记定义
- 默认选项
- 警告过滤

### conftest.py

提供共享的 fixtures：
- `temp_dir`: 临时目录
- `simple_env_info`: 基本环境配置
- `complex_env_info`: 复杂环境配置
- `sample_step_data`: 示例步骤数据
- `mock_environment`: 模拟环境

## 编写新测试

### 测试命名规范

```python
class TestCollectorName:
    """Test class for CollectorName."""
    
    def test_specific_functionality(self):
        """Test specific functionality description."""
        pass
```

### 使用 fixtures

```python
def test_with_temp_dir(self, temp_dir):
    """Test that uses temporary directory."""
    # temp_dir 是一个 Path 对象
    test_file = temp_dir / "test_data.pkl"
    # 测试代码...

def test_with_env_info(self, simple_env_info):
    """Test that uses environment configuration."""
    collector = RLCollector(name="test", save_path=None)
    collector.initialize_configs(simple_env_info)
    # 测试代码...
```

### 添加测试标记

```python
@pytest.mark.slow
def test_large_scale_operation(self):
    """Test that takes a long time to run."""
    pass

@pytest.mark.integration
def test_environment_integration(self):
    """Test integration with environment."""
    pass
```

## 故障排除

### 常见问题

1. **导入错误**: 确保 PYTHONPATH 包含项目根目录
   ```bash
   export PYTHONPATH=/path/to/aquaml2:$PYTHONPATH
   ```

2. **内存不足**: 性能测试可能需要大量内存，考虑：
   - 减少测试数据量
   - 增加系统内存
   - 跳过大规模测试

3. **依赖缺失**: 安装所有测试依赖
   ```bash
   pip install -r AquaML/tests/requirements.txt
   ```

### 调试测试

```bash
# 进入 pdb 调试器
python -m pytest tests/ --pdb

# 在第一个失败时停止
python -m pytest tests/ -x

# 显示局部变量
python -m pytest tests/ -l

# 更详细的输出
python -m pytest tests/ -vv
```

## 持续集成

测试套件设计为可以在 CI/CD 管道中运行：

```yaml
# GitHub Actions 示例
- name: Run tests
  run: |
    pip install -r AquaML/tests/requirements.txt
    python AquaML/tests/run_tests.py --all --coverage --output junit
```

## 贡献指南

添加新功能时，请：

1. 为新功能编写单元测试
2. 如果涉及外部集成，添加集成测试
3. 如果是性能敏感功能，添加性能测试
4. 确保所有测试通过
5. 保持测试覆盖率在 90% 以上

测试应该：
- 快速执行（单元测试 < 1s，集成测试 < 10s）
- 独立运行（不依赖其他测试）
- 确定性（每次运行结果一致）
- 有意义的测试名称和文档

## 联系方式

如有测试相关问题，请：
1. 查看现有测试作为参考
2. 检查 conftest.py 中可用的 fixtures
3. 参考 pytest 官方文档
4. 联系项目维护者 