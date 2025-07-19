# SAC (Soft Actor-Critic) Algorithm in AquaML

## 概述

SAC (Soft Actor-Critic) 是一种先进的离策略强化学习算法，基于最大熵原理。本实现参考了skrl框架，完全适配AquaML的架构。

## 主要特性

- **双Critic网络**: 使用两个独立的Q网络减少价值函数的过估计
- **自适应熵系数**: 自动调整探索程度，平衡探索与利用
- **软更新目标网络**: 增强训练稳定性
- **经验回放**: 高效的样本利用
- **连续动作空间**: 专为连续控制任务设计

## 使用方法

### 1. 导入必要模块

```python
from AquaML.learning.reinforcement.off_policy.sac import SAC, SACCfg
from AquaML.learning.model.gaussian import GaussianModel
```

### 2. 配置SAC参数

```python
sac_cfg = SACCfg()
sac_cfg.device = "cpu"
sac_cfg.memory_size = 10000
sac_cfg.batch_size = 256
sac_cfg.actor_learning_rate = 3e-4
sac_cfg.critic_learning_rate = 3e-4
sac_cfg.discount_factor = 0.99
sac_cfg.learn_entropy = True
```

### 3. 创建模型和智能体

```python
models = {
    "policy": policy,
    "critic_1": critic_1,
    "critic_2": critic_2,
    "target_critic_1": target_critic_1,
    "target_critic_2": target_critic_2
}

agent = SAC(models, sac_cfg, action_space=action_space)
```

### 4. 训练

```python
trainer = SequentialTrainer(env, agent, trainer_cfg)
trainer.train()
```

## 完整示例

参考 `examples/sac_pendulum_example.py` 获取完整的训练示例。