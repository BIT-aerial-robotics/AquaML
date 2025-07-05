#!/usr/bin/env python3
"""
AquaML Core 模块基础用法示例

这个示例展示了如何使用 AquaML Core 模块的基本功能，
包括协调器初始化、组件注册和生命周期管理。
"""

import time
from AquaML.core import AquaMLCoordinator, ComponentRegistry, LifecycleManager
from AquaML.core import AquaMLException, RegistryError, LifecycleError


# 示例1：基本的协调器使用
def basic_coordinator_example():
    """基本协调器使用示例"""
    print("=== 基本协调器使用示例 ===")
    
    # 获取协调器实例（单例模式）
    coordinator = AquaMLCoordinator()
    
    # 初始化协调器
    try:
        coordinator.initialize({
            "logging": {"level": "INFO"},
            "debug": True
        })
        print("✓ 协调器初始化成功")
    except AquaMLException as e:
        print(f"✗ 协调器初始化失败: {e}")
        return
    
    # 检查协调器状态
    status = coordinator.get_status()
    print(f"协调器状态: {status}")
    
    # 关闭协调器
    coordinator.shutdown()
    print("✓ 协调器已关闭")


# 示例2：组件注册
def component_registration_example():
    """组件注册示例"""
    print("\n=== 组件注册示例 ===")
    
    coordinator = AquaMLCoordinator()
    
    # 定义示例组件
    @coordinator.register_environment
    class CartPoleEnvironment:
        def __init__(self):
            self.name = "CartPole"
            self.state = "初始化"
            print(f"环境 {self.name} 已创建")
        
        def reset(self):
            self.state = "重置"
            return [0, 0, 0, 0]  # 示例状态
        
        def step(self, action):
            self.state = "运行中"
            return [0, 0, 0, 0], 1.0, False, {}  # 状态, 奖励, 结束, 信息
    
    @coordinator.register_agent
    class DQNAgent:
        def __init__(self):
            self.name = "DQN"
            self.state = "初始化"
            print(f"智能体 {self.name} 已创建")
        
        def act(self, observation):
            self.state = "决策中"
            return 0  # 示例动作
        
        def learn(self, experience):
            self.state = "学习中"
            print("智能体正在学习...")
    
    @coordinator.register_data_manager
    class SimpleDataManager:
        def __init__(self):
            self.data = []
            print("数据管理器已创建")
        
        def store(self, data):
            self.data.append(data)
            print(f"数据已存储: {data}")
        
        def get_data(self):
            return self.data
    
    # 创建组件实例
    env = CartPoleEnvironment()
    agent = DQNAgent()
    data_manager = SimpleDataManager()
    
    # 获取已注册的组件
    retrieved_env = coordinator.get_environment()
    retrieved_agent = coordinator.get_agent()
    retrieved_data_manager = coordinator.get_data_manager()
    
    print(f"✓ 已注册环境: {retrieved_env.name}")
    print(f"✓ 已注册智能体: {retrieved_agent.name}")
    print(f"✓ 已注册数据管理器: {type(retrieved_data_manager).__name__}")
    
    # 使用组件
    observation = env.reset()
    action = agent.act(observation)
    next_obs, reward, done, info = env.step(action)
    
    # 存储数据
    experience = {
        "observation": observation,
        "action": action,
        "reward": reward,
        "next_observation": next_obs,
        "done": done
    }
    data_manager.store(experience)
    
    print(f"✓ 模拟交互完成，奖励: {reward}")
    
    coordinator.shutdown()


# 示例3：组件注册器使用
def registry_example():
    """组件注册器使用示例"""
    print("\n=== 组件注册器使用示例 ===")
    
    registry = ComponentRegistry()
    
    # 定义示例模型
    class NeuralNetwork:
        def __init__(self, layers):
            self.layers = layers
            self.name = f"NN_{len(layers)}_layers"
            self.trained = False
        
        def train(self, data):
            self.trained = True
            print(f"模型 {self.name} 训练完成")
        
        def predict(self, x):
            return f"预测结果 for {x}"
    
    # 注册模型
    model1 = NeuralNetwork([128, 64, 32])
    model2 = NeuralNetwork([256, 128, 64, 32])
    
    try:
        registry.register(
            name="small_model",
            component=model1,
            metadata={
                "version": "1.0",
                "layers": model1.layers,
                "parameters": sum(model1.layers)
            }
        )
        
        registry.register(
            name="large_model",
            component=model2,
            metadata={
                "version": "1.0",
                "layers": model2.layers,
                "parameters": sum(model2.layers)
            }
        )
        
        print("✓ 模型注册成功")
    except RegistryError as e:
        print(f"✗ 模型注册失败: {e}")
    
    # 获取组件
    small_model = registry.get("small_model")
    large_model = registry.get("large_model")
    
    print(f"✓ 小模型: {small_model.name}")
    print(f"✓ 大模型: {large_model.name}")
    
    # 获取元数据
    metadata = registry.get_metadata("small_model")
    print(f"小模型元数据: {metadata}")
    
    # 列出所有组件
    components = registry.list_components()
    print(f"所有注册的组件: {components}")
    
    # 使用模型
    small_model.train("训练数据")
    prediction = small_model.predict("测试数据")
    print(f"预测结果: {prediction}")
    
    # 清理
    registry.clear()
    print("✓ 注册器已清理")


# 示例4：生命周期管理
def lifecycle_example():
    """生命周期管理示例"""
    print("\n=== 生命周期管理示例 ===")
    
    lifecycle = LifecycleManager()
    
    # 定义回调函数
    def on_startup(config):
        print(f"🚀 系统启动中... 配置: {config}")
        # 模拟初始化工作
        time.sleep(0.5)
        print("✓ 数据库连接已建立")
        print("✓ 配置文件已加载")
    
    def on_shutdown():
        print("🔄 系统关闭中...")
        # 模拟清理工作
        time.sleep(0.5)
        print("✓ 数据已保存")
        print("✓ 连接已关闭")
    
    # 添加回调
    lifecycle.add_startup_callback(on_startup)
    lifecycle.add_shutdown_callback(on_shutdown)
    
    # 初始化（会执行启动回调）
    try:
        lifecycle.initialize({
            "database_url": "sqlite:///example.db",
            "cache_size": 1000
        })
        print("✓ 生命周期管理器初始化完成")
    except LifecycleError as e:
        print(f"✗ 生命周期管理器初始化失败: {e}")
        return
    
    # 设置组件状态
    lifecycle.set_component_state("database", "connecting")
    lifecycle.set_component_state("model", "loading")
    lifecycle.set_component_state("api", "starting")
    
    # 模拟组件启动过程
    time.sleep(1)
    
    lifecycle.set_component_state("database", "running")
    lifecycle.set_component_state("model", "ready")
    lifecycle.set_component_state("api", "running")
    
    # 检查组件状态
    all_states = lifecycle.get_all_component_states()
    print(f"所有组件状态: {all_states}")
    
    # 检查特定组件
    if lifecycle.is_component_running("database"):
        print("✓ 数据库正在运行")
    
    if lifecycle.is_component_running("model"):
        print("✓ 模型已就绪")
    
    # 关闭（会执行关闭回调）
    lifecycle.shutdown()
    print("✓ 生命周期管理器已关闭")


# 示例5：上下文管理器使用
def context_manager_example():
    """上下文管理器使用示例"""
    print("\n=== 上下文管理器使用示例 ===")
    
    # 使用协调器的上下文管理器
    config = {
        "logging": {"level": "DEBUG"},
        "plugins": {}
    }
    
    try:
        with AquaMLCoordinator() as coordinator:
            print("✓ 协调器已自动初始化")
            coordinator.initialize(config)
            
            # 在这里使用协调器
            status = coordinator.get_status()
            print(f"协调器状态: {status}")
            
            # 模拟一些工作
            time.sleep(1)
            
            print("✓ 工作完成")
        
        print("✓ 协调器已自动关闭")
    
    except AquaMLException as e:
        print(f"✗ 上下文管理器使用失败: {e}")


# 示例6：错误处理
def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    registry = ComponentRegistry()
    
    # 测试重复注册错误
    try:
        registry.register("test_component", "component1")
        registry.register("test_component", "component2")  # 应该失败
    except RegistryError as e:
        print(f"✓ 捕获到预期的注册错误: {e}")
    
    # 测试获取不存在的组件
    try:
        component = registry.get_strict("nonexistent_component")
    except RegistryError as e:
        print(f"✓ 捕获到预期的获取错误: {e}")
    
    # 安全获取组件
    component = registry.get("nonexistent_component", "默认值")
    print(f"✓ 安全获取组件: {component}")
    
    # 测试生命周期错误
    lifecycle = LifecycleManager()
    
    try:
        # 添加非可调用对象作为回调
        lifecycle.add_startup_callback("not_callable")
    except LifecycleError as e:
        print(f"✓ 捕获到预期的生命周期错误: {e}")
    
    print("✓ 所有错误处理测试完成")


def main():
    """主函数"""
    print("AquaML Core 模块基础用法示例")
    print("=" * 50)
    
    # 运行所有示例
    basic_coordinator_example()
    component_registration_example()
    registry_example()
    lifecycle_example()
    context_manager_example()
    error_handling_example()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成! 🎉")


if __name__ == "__main__":
    main() 