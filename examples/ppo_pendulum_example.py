#!/usr/bin/env python3
"""
简化的PPO Pendulum训练例子

这个例子演示了如何使用AquaML的训练系统进行PPO训练，
但使用了更简单的模型定义和参数设置。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import os

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.ppo import PPO, PPOCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


class SimplePendulumPolicy(GaussianModel):
    """简化的摆环境策略模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 简单的网络结构
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 固定的log_std参数
        self.log_std_parameter = nn.Parameter(torch.zeros(1))
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算均值和log_std"""
        # 从字典中获取状态
        if "state" in data_dict:
            states = data_dict["state"]
        else:
            states = list(data_dict.values())[0]
        
        # 确保状态是二维的 (batch_size, features)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
        
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        
        return {"mean_actions": mean, "log_std": log_std}


class SimplePendulumValue(Model):
    """简化的摆环境价值模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算价值"""
        if "state" in data_dict:
            states = data_dict["state"]
        else:
            states = list(data_dict.values())[0]
        
        # 确保状态是二维的
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
            
        values = self.net(states)
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """价值函数的act方法"""
        return self.compute(data_dict)


def create_simple_setup():
    """创建简化的PPO设置"""
    
    # 1. 创建环境
    env = GymnasiumWrapper("Pendulum-v1")
    print("✓ 创建Pendulum环境")
    
    # 2. 设备设置
    device = "cpu"  # 强制使用CPU避免设备问题
    print(f"✓ 使用设备: {device}")
    
    # 3. 创建模型配置
    model_cfg = ModelCfg(
        device=device,
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 4. 创建模型
    policy = SimplePendulumPolicy(model_cfg)
    value = SimplePendulumValue(model_cfg)
    print("✓ 创建策略和价值模型")
    
    # 5. 创建简化的PPO配置
    ppo_cfg = PPOCfg()
    ppo_cfg.device = device
    ppo_cfg.memory_size = 200  # 减小内存大小
    ppo_cfg.rollouts = 32  # 减小rollout数量
    ppo_cfg.learning_epochs = 4
    ppo_cfg.mini_batches = 2
    ppo_cfg.learning_rate = 3e-4
    ppo_cfg.discount_factor = 0.99
    ppo_cfg.lambda_value = 0.95
    ppo_cfg.ratio_clip = 0.2
    ppo_cfg.value_clip = 0.2
    ppo_cfg.entropy_loss_scale = 0.01
    ppo_cfg.value_loss_scale = 0.5
    ppo_cfg.grad_norm_clip = 0.5
    ppo_cfg.mixed_precision = False  # 关闭混合精度
    
    # 6. 创建PPO智能体
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    print("✓ 创建PPO智能体")
    
    return env, agent


def main():
    """主函数"""
    print("=== 简化PPO Pendulum训练示例 ===\n")
    
    try:
        # 创建环境和智能体
        env, agent = create_simple_setup()
        
        # 创建训练器配置
        trainer_cfg = TrainerConfig(
            timesteps=1000,  # 减少训练步数
            headless=True,
            disable_progressbar=False,
            environment_info="episode"
        )
        
        # 创建训练器
        trainer = SequentialTrainer(env, agent, trainer_cfg)
        print("✓ 创建顺序训练器")
        
        # 开始训练
        print("\n--- 开始训练 ---")
        trainer.train()
        print("✓ 训练完成")
        
        # 保存模型
        os.makedirs("./runs", exist_ok=True)
        save_path = "./runs/simple_ppo_pendulum.pt"
        agent.save(save_path)
        print(f"✓ 模型保存至 {save_path}")
        
        print("\n=== 训练成功完成 ===")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()