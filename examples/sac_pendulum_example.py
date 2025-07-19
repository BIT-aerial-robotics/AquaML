#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.off_policy.sac import SAC, SACCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig

# 自动初始化默认文件系统
from AquaML import coordinator


class PendulumSACPolicy(GaussianModel):
    """Pendulum环境SAC策略模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 更深的网络结构 - Pendulum观察空间: (cos(theta), sin(theta), angular_velocity) = 3
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # SAC需要分离的均值和log标准差输出
        self.mean_layer = nn.Linear(64, 1)  # Pendulum动作空间: 1
        self.log_std_layer = nn.Linear(64, 1)
    
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取状态数据
        if "state" in data_dict:
            states = data_dict["state"]
        else:
            states = list(data_dict.values())[0]
        
        # 处理维度
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
        
        # 前向传播
        features = self.net(states)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # 数值稳定性约束
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        # 确保输出形状正确
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(0)
        
        return {"mean_actions": mean, "log_std": log_std}
    
    def act(self, data_dict: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """生成动作的SAC实现"""
        outputs = self.compute(data_dict)
        mean = outputs["mean_actions"]
        log_std = outputs["log_std"]
        std = torch.exp(log_std)
        
        if deterministic:
            # 评估时使用均值动作
            actions = torch.tanh(mean)
            log_prob = torch.zeros_like(actions)
        else:
            # 从高斯分布采样并应用tanh
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化技巧
            actions = torch.tanh(x_t)
            
            # 计算带tanh修正的对数概率
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # 缩放动作到Pendulum的动作范围[-2, 2]
        actions = actions * 2.0
        
        return {
            "actions": actions,
            "log_prob": log_prob,
            "mean_actions": mean,
            "log_std": log_std
        }
    
    def random_act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """生成随机动作用于探索"""
        batch_size = list(data_dict.values())[0].shape[0]
        device = list(data_dict.values())[0].device
        
        # Pendulum环境的随机动作范围[-2, 2]
        actions = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0
        log_prob = torch.zeros_like(actions)
        
        return {"actions": actions, "log_prob": log_prob}


class PendulumSACCritic(Model):
    """Pendulum环境SAC价值模型(Q函数)"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 更深的网络结构 - 输入: 状态(3) + 动作(1) = 4
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取状态数据
        if "state" in data_dict:
            states = data_dict["state"]
        else:
            states = list(data_dict.values())[0]
        
        # 获取动作数据
        if "actions" in data_dict:
            actions = data_dict["actions"]
        elif "action" in data_dict:
            actions = data_dict["action"]
        else:
            raise ValueError("No action found in data_dict for critic")
        
        # 处理维度
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
            
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        elif actions.dim() > 2:
            actions = actions.view(-1, actions.size(-1))
        
        # 连接状态和动作
        state_action = torch.cat([states, actions], dim=-1)
        
        # 前向传播
        q_value = self.net(state_action)
        
        # 确保输出形状正确
        if q_value.dim() == 1:
            q_value = q_value.unsqueeze(-1)
            
        return {"values": q_value}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


def main():
    # 1. 简单注册运行器（自动使用当前时间生成名称，自动创建workspace结构）
    runner_name = coordinator.registerRunner()
    print(f"✓ 运行器已注册: {runner_name}")
    
    # 获取文件系统实例（已自动初始化）
    fs = coordinator.getFileSystem()
    
    # 2. 创建环境
    env = GymnasiumWrapper("Pendulum-v1")
    
    # 3. 创建模型配置
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 4. 创建SAC模型
    policy = PendulumSACPolicy(model_cfg)
    critic_1 = PendulumSACCritic(model_cfg)
    critic_2 = PendulumSACCritic(model_cfg)
    target_critic_1 = PendulumSACCritic(model_cfg)
    target_critic_2 = PendulumSACCritic(model_cfg)
    
    # 5. 配置SAC参数 - 新数据流架构的关键参数
    sac_cfg = SACCfg()
    sac_cfg.device = "cpu"
    sac_cfg.memory_size = 10000
    sac_cfg.batch_size = 256  # 📊 关键参数：批量大小
    sac_cfg.gradient_steps = 1
    sac_cfg.learning_starts = 1000
    sac_cfg.random_timesteps = 1000
    
    # 学习率设置
    sac_cfg.actor_learning_rate = 3e-4
    sac_cfg.critic_learning_rate = 3e-4
    sac_cfg.entropy_learning_rate = 3e-4
    
    # SAC超参数
    sac_cfg.discount_factor = 0.99
    sac_cfg.polyak = 0.005
    sac_cfg.initial_entropy_value = 0.2
    sac_cfg.learn_entropy = True
    sac_cfg.target_entropy = -1.0  # -action_dim for Pendulum
    sac_cfg.mixed_precision = False
    
    # 6. 创建SAC智能体
    models = {
        "policy": policy,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2
    }
    
    action_space = {"shape": (1,)}  # Pendulum动作空间维度
    agent = SAC(models, sac_cfg, action_space=action_space)
    
    # 7. 创建训练器配置 - 简化配置，自动从agent读取参数
    trainer_cfg = TrainerConfig(
        timesteps=50000,
        headless=True,
        disable_progressbar=False
    )
    
    # 8. 创建训练器并开始训练 - 使用新数据流架构
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    
    print(f"🌊 开始使用新数据流架构训练SAC:")
    print(f"  📊 内存大小: {sac_cfg.memory_size}")
    print(f"  📦 批量大小: {sac_cfg.batch_size}")
    print(f"  🔄 总时间步: {trainer_cfg.timesteps}")
    print(f"  📥 数据缓存格式: (num_env, steps, dims)")
    
    trainer.train()
    
    # 9. 显示训练统计信息
    status = trainer.get_enhanced_status()
    print(f"\n📈 训练统计:")
    print(f"  收集步数: {status['data_flow_architecture']['collected_steps']}")
    print(f"  训练轮次: {status['data_flow_architecture']['training_episodes']}")
    print(f"  数据效率: {status['data_flow_architecture']['collected_steps']/trainer_cfg.timesteps:.2f}")
    
    # 10. 保存模型到工作目录
    agent.save(fs.getModelPath(runner_name, "trained_sac_model.pt"))
    print("✅ 训练完成！模型已保存")
    
    # 11. 验证数据缓存功能
    print(f"\n🔍 验证数据缓存:")
    available_buffers = trainer.list_available_buffers()
    print(f"  可用缓存: {available_buffers}")
    
    # 显示部分缓存数据信息
    for buffer_name in available_buffers[:3]:  # 只显示前3个
        data = trainer.get_collected_buffer_data(buffer_name)
        if data is not None:
            print(f"  {buffer_name}: 形状 {data.shape}")
    
    print(f"\n🌊 新数据流架构SAC演示完成！")


if __name__ == "__main__":
    main()