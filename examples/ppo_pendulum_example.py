#!/usr/bin/env python3
"""
PPO Pendulum训练示例

使用AquaML标准trainer进行PPO训练
"""

import torch
import torch.nn as nn
from typing import Dict
from loguru import logger

# 使用标准的环境包装器和训练器
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.ppo import PPO, PPOCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


class PendulumPolicy(GaussianModel):
    """Pendulum环境策略模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 更深的网络结构
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 学习log标准差参数
        self.log_std_parameter = nn.Parameter(torch.zeros(1))
        
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
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        
        # 确保输出形状正确
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(0)
        
        return {"mean_actions": mean, "log_std": log_std}


class PendulumValue(Model):
    """Pendulum环境价值模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 更深的网络结构
        self.net = nn.Sequential(
            nn.Linear(3, 128),
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
        
        # 处理维度
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
            
        # 前向传播
        values = self.net(states)
        
        # 确保输出形状正确
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


def main():
    """主训练函数"""
    print("=== PPO Pendulum训练示例 ===")
    print("使用AquaML标准trainer进行训练\n")
    
    # 1. 创建环境
    env = GymnasiumWrapper("Pendulum-v1")
    print("✓ 创建Pendulum环境")
    
    # 2. 创建模型配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelCfg(
        device=device,
        inputs_name=["state"],
        concat_dict=False
    )
    print(f"✓ 使用设备: {device}")
    
    # 3. 创建策略和价值模型
    policy = PendulumPolicy(model_cfg)
    value = PendulumValue(model_cfg)
    print("✓ 创建策略和价值网络")
    
    # 4. 配置PPO参数 - 调整为能训练出好结果的参数
    ppo_cfg = PPOCfg()
    ppo_cfg.device = device
    ppo_cfg.rollouts = 32          # 适中的rollouts
    ppo_cfg.memory_size = 2048     # 适中的内存大小
    ppo_cfg.learning_epochs = 8    # 适中的学习epoch
    ppo_cfg.mini_batches = 4       # 适中的mini_batches
    ppo_cfg.learning_rate = 3e-4   # 标准学习率
    ppo_cfg.discount_factor = 0.99
    ppo_cfg.lambda_value = 0.95
    ppo_cfg.ratio_clip = 0.2
    ppo_cfg.value_clip = 0.2
    ppo_cfg.entropy_loss_scale = 0.01
    ppo_cfg.value_loss_scale = 0.5
    ppo_cfg.grad_norm_clip = 0.5
    ppo_cfg.mixed_precision = False
    
    # 5. 创建PPO智能体
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    print("✓ 创建PPO智能体")
    
    # 6. 创建训练器配置 - 设置足够的训练步数
    trainer_cfg = TrainerConfig(
        timesteps=10000,        # 适中的训练步数
        headless=True,          # 无头模式
        disable_progressbar=False,
        close_environment_at_exit=True,
        environment_info="episode",
        checkpoint_interval=2000,  # 每2000步保存一次
        device=device
    )
    
    # 7. 创建并启动训练器
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    print("✓ 创建训练器")
    print(f"开始训练 {trainer_cfg.timesteps} 步...")
    
    try:
        # 开始训练
        trainer.train()
        print("✓ 训练完成")
        
        # 保存最终模型
        agent.save("./trained_pendulum_model_final.pt")
        print("✓ 最终模型已保存")
        
        # 运行评估
        print("\n开始评估...")
        eval_cfg = TrainerConfig(
            timesteps=1000,
            headless=True,           # 评估时也使用无头模式
            disable_progressbar=False,
            stochastic_evaluation=False,  # 使用确定性策略
            device=device
        )
        
        eval_trainer = SequentialTrainer(env, agent, eval_cfg)
        eval_trainer.eval()
        print("✓ 评估完成")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 即使出错也保存当前模型
        try:
            agent.save("./trained_pendulum_model_interrupted.pt")
            print("✓ 已保存中断时的模型")
        except:
            pass
    
    finally:
        # 关闭环境
        if hasattr(env, 'close'):
            env.close()
        print("✓ 环境已关闭")


if __name__ == "__main__":
    main()