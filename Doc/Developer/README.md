# 开发者文档

本部分用于存储在开发过程中，对部分模块的设计思路，该部分不是最终的设计文档。后期接口会进行调整，正式版会提供详细的设计文档。

## 📚 文档导航

### 🧪 测试相关
- **[测试指南](./Testing_Guide.md)** - 完整的测试策略、工具使用和最佳实践指南
- **[测试详细文档](../../tests/README.md)** - 测试套件使用说明和快速开始

### 🔧 设计思路
- **[Worker设计思路](./Worker设计思路.md)** - Worker模块的设计理念和实现思路

## 🚀 快速开始

### 运行测试
```bash
# 快速测试
python tests/run_tests.py --quick

# 完整测试
python tests/run_tests.py --type all --coverage
```

### 开发工作流
1. 编写代码
2. 运行相关测试：`python tests/run_tests.py --quick`
3. 提交前完整测试：`python tests/run_tests.py --type all --coverage`
4. 提交代码

详细的测试指南请参考 [测试指南](./Testing_Guide.md)。