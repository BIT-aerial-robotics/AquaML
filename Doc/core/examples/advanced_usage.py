#!/usr/bin/env python3
"""
AquaML Core 模块高级用法示例

这个示例展示了 AquaML Core 模块的高级功能，包括：
- 复杂的组件注册和管理
- 高级生命周期管理
- 回调机制和事件处理
- 组件间的依赖关系
- 动态插件系统
"""

import time
import threading
from typing import Dict, Any, List, Optional
from AquaML.core import (
    AquaMLCoordinator, 
    ComponentRegistry, 
    LifecycleManager,
    AquaMLException,
    RegistryError,
    LifecycleError
)


# 高级组件基类
class BaseComponent:
    """组件基类，提供通用功能"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = "initialized"
        self.dependencies: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def start(self):
        """启动组件"""
        self.state = "running"
        print(f"🚀 {self.name} 已启动")
    
    def stop(self):
        """停止组件"""
        self.state = "stopped"
        print(f"⏹️  {self.name} 已停止")
    
    def health_check(self) -> bool:
        """健康检查"""
        return self.state == "running"


# 高级环境组件
class AdvancedEnvironment(BaseComponent):
    """高级环境组件"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config
        self.episode_count = 0
        self.total_steps = 0
        self.rewards_history = []
        self.dependencies = ["data_manager", "logger"]
    
    def reset(self):
        """重置环境"""
        self.episode_count += 1
        print(f"🔄 环境 {self.name} 重置，第 {self.episode_count} 轮")
        return {"observation": [0, 0, 0, 0], "info": {"episode": self.episode_count}}
    
    def step(self, action):
        """执行动作"""
        self.total_steps += 1
        reward = self._calculate_reward(action)
        self.rewards_history.append(reward)
        
        done = self.total_steps % 100 == 0  # 每100步结束一轮
        
        return {
            "observation": [0, 0, 0, 0],
            "reward": reward,
            "done": done,
            "info": {"step": self.total_steps, "episode": self.episode_count}
        }
    
    def _calculate_reward(self, action):
        """计算奖励"""
        # 模拟奖励计算
        base_reward = 1.0
        penalty = 0.1 if action < 0 else 0
        return base_reward - penalty
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            "episodes": self.episode_count,
            "total_steps": self.total_steps,
            "average_reward": sum(self.rewards_history) / len(self.rewards_history) if self.rewards_history else 0,
            "total_reward": sum(self.rewards_history)
        }


# 高级智能体组件
class AdvancedAgent(BaseComponent):
    """高级智能体组件"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config
        self.memory = []
        self.learning_rate = config.get("learning_rate", 0.01)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.dependencies = ["model", "data_manager"]
    
    def act(self, observation):
        """选择动作"""
        if self._should_explore():
            action = self._random_action()
        else:
            action = self._greedy_action(observation)
        
        return action
    
    def learn(self, experience):
        """学习经验"""
        self.memory.append(experience)
        
        # 每收集够一定数量的经验就学习
        if len(self.memory) >= self.config.get("batch_size", 32):
            self._update_model()
    
    def _should_explore(self):
        """判断是否应该探索"""
        return self.exploration_rate > 0.05  # 简化的探索策略
    
    def _random_action(self):
        """随机动作"""
        import random
        return random.choice([-1, 0, 1])
    
    def _greedy_action(self, observation):
        """贪婪动作"""
        # 简化的贪婪策略
        return 1 if sum(observation["observation"]) > 0 else -1
    
    def _update_model(self):
        """更新模型"""
        print(f"🧠 {self.name} 正在学习，内存中有 {len(self.memory)} 个经验")
        # 模拟学习过程
        self.memory.clear()


# 高级数据管理器
class AdvancedDataManager(BaseComponent):
    """高级数据管理器"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config
        self.data_buffer = []
        self.max_buffer_size = config.get("max_buffer_size", 10000)
        self.save_interval = config.get("save_interval", 100)
        self.step_count = 0
        self.dependencies = ["logger"]
    
    def store_experience(self, experience):
        """存储经验"""
        self.data_buffer.append(experience)
        self.step_count += 1
        
        # 检查缓冲区是否满了
        if len(self.data_buffer) > self.max_buffer_size:
            self._evict_old_data()
        
        # 定期保存数据
        if self.step_count % self.save_interval == 0:
            self._save_data()
    
    def get_batch(self, batch_size: int):
        """获取批量数据"""
        if len(self.data_buffer) < batch_size:
            return self.data_buffer.copy()
        
        import random
        return random.sample(self.data_buffer, batch_size)
    
    def _evict_old_data(self):
        """删除旧数据"""
        evict_count = len(self.data_buffer) - self.max_buffer_size
        self.data_buffer = self.data_buffer[evict_count:]
        print(f"📦 数据管理器删除了 {evict_count} 条旧数据")
    
    def _save_data(self):
        """保存数据"""
        print(f"💾 数据管理器保存了 {len(self.data_buffer)} 条数据")
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            "buffer_size": len(self.data_buffer),
            "max_buffer_size": self.max_buffer_size,
            "total_steps": self.step_count,
            "utilization": len(self.data_buffer) / self.max_buffer_size
        }


# 组件依赖管理器
class DependencyManager:
    """组件依赖管理器"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.dependency_graph: Dict[str, List[str]] = {}
        self.resolved_order: List[str] = []
    
    def register_dependencies(self, component_name: str, dependencies: List[str]):
        """注册组件依赖"""
        self.dependency_graph[component_name] = dependencies
    
    def resolve_dependencies(self) -> List[str]:
        """解析依赖关系，返回启动顺序"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"循环依赖检测到: {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            # 先处理依赖
            for dep in self.dependency_graph.get(node, []):
                visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        # 处理所有组件
        for component in self.dependency_graph:
            visit(component)
        
        self.resolved_order = result
        return result
    
    def start_components_in_order(self):
        """按依赖顺序启动组件"""
        print("🚀 按依赖顺序启动组件:")
        for component_name in self.resolved_order:
            component = self.registry.get(component_name)
            if component and hasattr(component, 'start'):
                component.start()
                time.sleep(0.1)  # 模拟启动时间
    
    def stop_components_in_reverse_order(self):
        """按相反顺序停止组件"""
        print("⏹️  按相反顺序停止组件:")
        for component_name in reversed(self.resolved_order):
            component = self.registry.get(component_name)
            if component and hasattr(component, 'stop'):
                component.stop()
                time.sleep(0.1)  # 模拟停止时间


# 事件系统
class EventSystem:
    """事件系统"""
    
    def __init__(self):
        self.listeners: Dict[str, List[callable]] = {}
    
    def subscribe(self, event_name: str, callback: callable):
        """订阅事件"""
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(callback)
    
    def unsubscribe(self, event_name: str, callback: callable):
        """取消订阅事件"""
        if event_name in self.listeners:
            self.listeners[event_name].remove(callback)
    
    def emit(self, event_name: str, data: Any = None):
        """发送事件"""
        if event_name in self.listeners:
            for callback in self.listeners[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"❌ 事件处理错误 {event_name}: {e}")


# 高级协调器示例
def advanced_coordinator_example():
    """高级协调器使用示例"""
    print("=== 高级协调器使用示例 ===")
    
    # 创建协调器和注册器
    coordinator = AquaMLCoordinator()
    registry = ComponentRegistry()
    
    # 创建组件
    env_config = {
        "max_steps": 1000,
        "reward_threshold": 200
    }
    
    agent_config = {
        "learning_rate": 0.01,
        "exploration_rate": 0.1,
        "batch_size": 32
    }
    
    data_config = {
        "max_buffer_size": 5000,
        "save_interval": 50
    }
    
    # 注册组件
    environment = AdvancedEnvironment("CartPole-v1", env_config)
    agent = AdvancedAgent("DQN-Agent", agent_config)
    data_manager = AdvancedDataManager("ExperienceReplay", data_config)
    
    # 添加一个简单的日志组件
    class Logger(BaseComponent):
        def __init__(self):
            super().__init__("Logger")
            self.logs = []
        
        def log(self, message):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            self.logs.append(log_entry)
            print(f"📝 {log_entry}")
    
    logger = Logger()
    
    # 注册所有组件
    registry.register("environment", environment)
    registry.register("agent", agent)
    registry.register("data_manager", data_manager)
    registry.register("logger", logger)
    
    # 设置依赖关系
    dependency_manager = DependencyManager(registry)
    dependency_manager.register_dependencies("environment", ["data_manager", "logger"])
    dependency_manager.register_dependencies("agent", ["data_manager"])
    dependency_manager.register_dependencies("data_manager", ["logger"])
    dependency_manager.register_dependencies("logger", [])
    
    # 解析并启动组件
    startup_order = dependency_manager.resolve_dependencies()
    print(f"组件启动顺序: {startup_order}")
    
    dependency_manager.start_components_in_order()
    
    # 创建事件系统
    event_system = EventSystem()
    
    # 订阅事件
    def on_episode_end(data):
        stats = data["environment"].get_statistics()
        logger.log(f"第 {stats['episodes']} 轮结束，总奖励: {stats['total_reward']:.2f}")
    
    def on_learning_complete(data):
        logger.log(f"智能体学习完成，内存使用: {len(data['agent'].memory)}")
    
    event_system.subscribe("episode_end", on_episode_end)
    event_system.subscribe("learning_complete", on_learning_complete)
    
    # 运行模拟
    print("\n🎮 开始运行模拟...")
    
    for episode in range(3):
        observation = environment.reset()
        
        for step in range(20):  # 每轮20步
            action = agent.act(observation)
            result = environment.step(action)
            
            # 创建经验
            experience = {
                "observation": observation,
                "action": action,
                "reward": result["reward"],
                "next_observation": result["observation"],
                "done": result["done"]
            }
            
            # 存储经验
            data_manager.store_experience(experience)
            
            # 智能体学习
            agent.learn(experience)
            
            observation = result
            
            if result["done"]:
                event_system.emit("episode_end", {"environment": environment})
                break
        
        # 发送学习完成事件
        event_system.emit("learning_complete", {"agent": agent})
    
    # 显示统计信息
    print("\n📊 统计信息:")
    print(f"环境统计: {environment.get_statistics()}")
    print(f"数据管理器统计: {data_manager.get_statistics()}")
    
    # 健康检查
    print("\n🏥 健康检查:")
    components = [environment, agent, data_manager, logger]
    for component in components:
        status = "✅ 健康" if component.health_check() else "❌ 异常"
        print(f"{component.name}: {status}")
    
    # 停止所有组件
    dependency_manager.stop_components_in_reverse_order()
    
    print("✅ 高级协调器示例完成")


# 多线程组件管理示例
def multithreaded_component_example():
    """多线程组件管理示例"""
    print("\n=== 多线程组件管理示例 ===")
    
    registry = ComponentRegistry()
    
    # 创建线程安全的组件
    class ThreadSafeComponent(BaseComponent):
        def __init__(self, name: str):
            super().__init__(name)
            self.lock = threading.Lock()
            self.counter = 0
        
        def increment(self):
            with self.lock:
                self.counter += 1
                print(f"🔢 {self.name} 计数器: {self.counter}")
        
        def get_count(self):
            with self.lock:
                return self.counter
    
    # 创建多个组件
    components = []
    for i in range(3):
        component = ThreadSafeComponent(f"Component-{i}")
        components.append(component)
        registry.register(f"component_{i}", component)
    
    # 创建工作线程
    def worker(component_name, iterations):
        component = registry.get(component_name)
        for _ in range(iterations):
            component.increment()
            time.sleep(0.01)  # 模拟工作
    
    # 启动多个线程
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=worker,
            args=(f"component_{i}", 10)
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 显示最终结果
    print("\n🏁 最终计数器值:")
    for i in range(3):
        component = registry.get(f"component_{i}")
        print(f"Component-{i}: {component.get_count()}")
    
    print("✅ 多线程组件管理示例完成")


# 动态组件加载示例
def dynamic_component_loading_example():
    """动态组件加载示例"""
    print("\n=== 动态组件加载示例 ===")
    
    registry = ComponentRegistry()
    
    # 模拟动态加载的组件类
    component_classes = {
        "ModelA": lambda: type("ModelA", (BaseComponent,), {
            "__init__": lambda self: BaseComponent.__init__(self, "ModelA"),
            "predict": lambda self, x: f"ModelA预测: {x * 2}"
        })(),
        
        "ModelB": lambda: type("ModelB", (BaseComponent,), {
            "__init__": lambda self: BaseComponent.__init__(self, "ModelB"),
            "predict": lambda self, x: f"ModelB预测: {x * 3}"
        })(),
        
        "ModelC": lambda: type("ModelC", (BaseComponent,), {
            "__init__": lambda self: BaseComponent.__init__(self, "ModelC"),
            "predict": lambda self, x: f"ModelC预测: {x * 5}"
        })()
    }
    
    # 动态加载组件
    def load_component(component_type: str):
        if component_type in component_classes:
            component = component_classes[component_type]()
            registry.register(component_type.lower(), component)
            print(f"✅ 动态加载组件: {component_type}")
            return component
        else:
            print(f"❌ 未知组件类型: {component_type}")
            return None
    
    # 加载不同类型的组件
    for model_type in ["ModelA", "ModelB", "ModelC"]:
        load_component(model_type)
    
    # 使用动态加载的组件
    print("\n🧪 测试动态加载的组件:")
    for model_name in ["modela", "modelb", "modelc"]:
        model = registry.get(model_name)
        if model:
            result = model.predict(10)
            print(f"{model.name}: {result}")
    
    # 运行时替换组件
    print("\n🔄 运行时组件替换:")
    new_model = type("ModelA_v2", (BaseComponent,), {
        "__init__": lambda self: BaseComponent.__init__(self, "ModelA_v2"),
        "predict": lambda self, x: f"ModelA_v2预测: {x * 10}"
    })()
    
    registry.register("modela", new_model, replace=True)
    
    updated_model = registry.get("modela")
    result = updated_model.predict(10)
    print(f"更新后的模型: {result}")
    
    print("✅ 动态组件加载示例完成")


def main():
    """主函数"""
    print("AquaML Core 模块高级用法示例")
    print("=" * 60)
    
    # 运行高级示例
    advanced_coordinator_example()
    multithreaded_component_example()
    dynamic_component_loading_example()
    
    print("\n" + "=" * 60)
    print("所有高级示例运行完成! 🎉")


if __name__ == "__main__":
    main() 