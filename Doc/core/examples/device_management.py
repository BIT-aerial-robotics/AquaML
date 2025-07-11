#!/usr/bin/env python3
"""
AquaML Core - 设备管理示例

本示例展示了AquaML框架的设备管理功能，包括：
- 设备检测和选择
- 设备验证
- 设备信息查询
- 设备切换
- 与PyTorch集成
"""

import torch
import torch.nn as nn
from AquaML.core import AquaMLCoordinator


def demonstrate_device_detection():
    """演示设备检测功能"""
    print("=== 设备检测 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 获取设备信息
    device_info = coordinator.get_device_info()
    print(f"当前设备: {device_info['current_device']}")
    print(f"可用设备: {device_info['available_devices']}")
    print(f"GPU可用: {device_info['gpu_available']}")
    print(f"GPU数量: {device_info['gpu_count']}")
    
    if device_info['gpu_available']:
        print("GPU详细信息:")
        for i, gpu_info in enumerate(device_info['gpu_details']):
            print(f"  GPU {i}: {gpu_info}")
    
    print()


def demonstrate_device_selection():
    """演示设备选择功能"""
    print("=== 设备选择 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 检查GPU是否可用
    if coordinator.is_gpu_available():
        print("检测到GPU，尝试使用GPU")
        if coordinator.set_device("cuda:0"):
            print("✓ 成功设置为GPU 0")
        else:
            print("✗ 设置GPU失败，回退到CPU")
            coordinator.set_device("cpu")
    else:
        print("未检测到GPU，使用CPU")
        coordinator.set_device("cpu")
    
    print(f"最终选择的设备: {coordinator.get_device()}")
    print()


def demonstrate_device_validation():
    """演示设备验证功能"""
    print("=== 设备验证 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 测试不同设备的验证
    test_devices = ["cpu", "cuda:0", "cuda:1", "cuda:99"]
    
    for device in test_devices:
        is_valid = coordinator.validate_device(device)
        status = "✓ 可用" if is_valid else "✗ 不可用"
        print(f"设备 {device}: {status}")
    
    print()


def demonstrate_device_switching():
    """演示设备切换功能"""
    print("=== 设备切换 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 获取可用设备列表
    available_devices = coordinator.get_available_devices()
    print(f"可用设备: {available_devices}")
    
    # 依次切换到每个可用设备
    for device in available_devices:
        if coordinator.set_device(device):
            print(f"✓ 成功切换到 {device}")
        else:
            print(f"✗ 切换到 {device} 失败")
    
    print()


def demonstrate_pytorch_integration():
    """演示与PyTorch的集成"""
    print("=== PyTorch集成 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 自动选择最佳设备
    if coordinator.is_gpu_available():
        coordinator.set_device("cuda:0")
    else:
        coordinator.set_device("cpu")
    
    # 获取PyTorch设备对象
    torch_device = coordinator.get_torch_device()
    print(f"PyTorch设备: {torch_device}")
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 创建模型并移动到指定设备
    model = SimpleModel()
    model.to(torch_device)
    
    # 创建测试数据
    x = torch.randn(5, 10).to(torch_device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"输入数据设备: {x.device}")
    print(f"输出数据设备: {output.device}")
    print(f"输出形状: {output.shape}")
    
    print()


def demonstrate_robust_device_handling():
    """演示鲁棒的设备处理"""
    print("=== 鲁棒设备处理 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 定义首选设备列表（按优先级排序）
    preferred_devices = ["cuda:0", "cuda:1", "cpu"]
    
    selected_device = None
    for device in preferred_devices:
        if coordinator.validate_device(device):
            if coordinator.set_device(device):
                selected_device = device
                print(f"✓ 成功选择设备: {device}")
                break
            else:
                print(f"✗ 设备 {device} 验证成功但设置失败")
        else:
            print(f"✗ 设备 {device} 不可用")
    
    if selected_device:
        print(f"最终使用设备: {selected_device}")
    else:
        print("错误: 无法选择任何设备")
    
    print()


def demonstrate_config_based_device_selection():
    """演示基于配置的设备选择"""
    print("=== 配置驱动的设备选择 ===")
    
    # 模拟不同的配置场景
    configs = [
        {"device": "cuda:0", "name": "GPU优先配置"},
        {"device": "cpu", "name": "CPU强制配置"},
        {"device": "cuda:99", "name": "无效GPU配置"},
        {}, # 无设备配置，使用自动选择
    ]
    
    for i, config in enumerate(configs):
        print(f"配置 {i+1}: {config.get('name', '自动选择配置')}")
        
        coordinator = AquaMLCoordinator()
        coordinator.initialize(config)
        
        device = coordinator.get_device()
        is_valid = coordinator.validate_device(device)
        
        print(f"  选择的设备: {device}")
        print(f"  设备有效性: {'✓' if is_valid else '✗'}")
        print()


def main():
    """主函数"""
    print("🌊 AquaML Core - 设备管理示例 🌊")
    print("=" * 50)
    
    try:
        demonstrate_device_detection()
        demonstrate_device_selection()
        demonstrate_device_validation()
        demonstrate_device_switching()
        demonstrate_pytorch_integration()
        demonstrate_robust_device_handling()
        demonstrate_config_based_device_selection()
        
        print("✅ 所有示例运行完成！")
        
    except Exception as e:
        print(f"❌ 运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 