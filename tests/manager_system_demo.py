#!/usr/bin/env python3
"""
AquaML Manager System Demo

This script demonstrates the usage of the new AquaML manager system architecture.
"""

import torch
import torch.nn as nn
from AquaML.core.coordinator import get_coordinator


def main():
    """演示AquaML管理器系统的使用"""
    print("🌊 AquaML Manager System Demo 🌊")
    print("=" * 50)

    # 获取协调器实例
    coordinator = get_coordinator()

    print("\n1. 📋 初始状态")
    print("-" * 30)
    initial_status = coordinator.get_status()
    print(f"初始化状态: {initial_status['initialized']}")
    print(f"组件数量: {initial_status['components']}")

    print("\n2. 🤖 注册模型")
    print("-" * 30)

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )

        def forward(self, x):
            return self.layers(x)

    class ValueNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.layers(x)

    # 注册模型
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()

    coordinator.registerModel(policy_net, "policy_network")
    coordinator.registerModel(value_net, "value_network")

    # 获取模型管理器并显示状态
    model_manager = coordinator.get_model_manager()
    print(f"✅ 已注册模型: {model_manager.list_models()}")
    print(f"✅ 模型数量: {model_manager.get_models_count()}")

    print("\n3. 🏃 注册Runner")
    print("-" * 30)

    coordinator.registerRunner("demo_runner_v1.0")
    runner_manager = coordinator.get_runner_manager()
    print(f"✅ 运行器: {runner_manager.get_runner()}")

    print("\n4. 🌍 注册环境")
    print("-" * 30)

    @coordinator.registerEnv
    class DemoEnvironment:
        def __init__(self):
            self.name = "DemoEnv"
            self.state_dim = 64
            self.action_dim = 4
            self.current_step = 0

        def reset(self):
            self.current_step = 0
            return torch.randn(self.state_dim)

        def step(self, action):
            self.current_step += 1
            next_state = torch.randn(self.state_dim)
            reward = torch.randn(1).item()
            done = self.current_step >= 100
            return next_state, reward, done, {}

    env = DemoEnvironment()
    env_manager = coordinator.get_environment_manager()
    print(f"✅ 环境: {env.name}")
    print(f"✅ 状态维度: {env.state_dim}, 动作维度: {env.action_dim}")

    print("\n5. 🤖 注册Agent")
    print("-" * 30)

    @coordinator.registerAgent
    class DemoAgent:
        def __init__(self):
            self.name = "DemoAgent"
            self.episode_count = 0

        def act(self, state):
            # 简单的随机策略
            return torch.randint(0, 4, (1,)).item()

        def update(self, experience):
            # 模拟学习过程
            pass

        def new_episode(self):
            self.episode_count += 1

    agent = DemoAgent()
    agent_manager = coordinator.get_agent_manager()
    print(f"✅ 智能体: {agent.name}")

    print("\n6. 📊 注册数据单元")
    print("-" * 30)

    @coordinator.registerDataUnit
    class ExperienceBuffer:
        def __init__(self):
            self.name = "ExperienceBuffer"
            self.experiences = []
            self.max_size = 10000

        def add_experience(self, state, action, reward, next_state, done):
            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
            self.experiences.append(experience)
            if len(self.experiences) > self.max_size:
                self.experiences.pop(0)

        def sample_batch(self, batch_size=32):
            import random

            return random.sample(
                self.experiences, min(batch_size, len(self.experiences))
            )

        def getUnitStatusDict(self):
            return {
                "size": len(self.experiences),
                "max_size": self.max_size,
                "status": "active",
            }

    buffer = ExperienceBuffer()
    data_unit_manager = coordinator.get_data_unit_manager()
    print(f"✅ 数据单元: {buffer.name}")

    print("\n7. 🎯 运行简单的交互循环")
    print("-" * 30)

    # 运行几个步骤来演示系统交互
    env = coordinator.getEnv()
    agent = coordinator.getAgent()
    buffer = coordinator.getDataUnit("ExperienceBuffer")

    state = env.reset()
    total_reward = 0

    for step in range(5):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        # 存储经验
        buffer.add_experience(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        print(f"步骤 {step + 1}: 动作={action}, 奖励={reward:.3f}")

        if done:
            break

    print(f"✅ 总奖励: {total_reward:.3f}")
    print(f"✅ 经验缓冲区大小: {len(buffer.experiences)}")

    print("\n8. 📈 系统状态总览")
    print("-" * 30)

    # 显示所有管理器的状态
    all_status = coordinator.get_all_managers_status()
    for manager_name, status in all_status.items():
        print(f"📊 {manager_name}: {status}")

    print("\n9. 🔧 管理器直接访问示例")
    print("-" * 30)

    # 展示如何直接访问管理器进行更精细的控制
    model_manager = coordinator.get_model_manager()

    # 检查特定模型是否存在
    if model_manager.model_exists("policy_network"):
        policy_model = model_manager.get_model_instance("policy_network")
        print(
            f"✅ 策略网络参数数量: {sum(p.numel() for p in policy_model.parameters())}"
        )

    # 显示模型详细信息
    for model_name in model_manager.list_models():
        model_dict = model_manager.get_model(model_name)
        model_instance = model_dict["model"]
        print(f"✅ 模型 '{model_name}': {type(model_instance).__name__}")

    print("\n10. 💾 数据持久化演示")
    print("-" * 30)

    # 注册一个简单的文件系统
    @coordinator.registerFileSystem
    class DemoFileSystem:
        def __init__(self):
            self.saved_data = {}
            self.configured_runners = []

        def configRunner(self, runner_name):
            self.configured_runners.append(runner_name)
            print(f"✅ 配置运行器: {runner_name}")

        def saveDataUnit(self, runner_name, data_unit_status):
            self.saved_data[runner_name] = data_unit_status
            print(f"✅ 保存数据单元状态: {runner_name}")

    fs = DemoFileSystem()

    # 保存数据单元信息
    coordinator.saveDataUnitInfo()

    print("\n11. 📋 最终状态报告")
    print("-" * 30)

    final_status = coordinator.get_status()
    print(f"🎯 最终状态:")
    print(f"   - 初始化: {final_status['initialized']}")
    print(f"   - 组件: {final_status['components']}")
    print(f"   - 设备信息: {final_status['device_info']['current_device']}")
    print(f"   - 运行器: {final_status['runner_name']}")

    print("\n🎉 演示完成！新的管理器系统运行正常。")
    print("\n💡 提示:")
    print("   - 每个组件都有专门的管理器负责管理")
    print("   - 可以通过协调器统一访问，也可以直接访问管理器")
    print("   - 完整的错误处理和状态管理")
    print("   - 易于扩展和测试")


if __name__ == "__main__":
    main()
