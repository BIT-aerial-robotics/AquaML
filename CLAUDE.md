# 软件使用教程

以下是一个完整的PPO训练Pendulum环境的例子：

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.ppo import PPO, PPOCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig

# 定义策略网络
class PendulumPolicy(GaussianModel):
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(1))
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        return {"mean_actions": mean, "log_std": log_std}

# 定义价值网络
class PendulumValue(Model):
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        values = self.net(states)
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)

def main():
    # 1. 创建环境
    env = GymnasiumWrapper("Pendulum-v1")
    
    # 2. 创建模型配置
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 3. 创建模型
    policy = PendulumPolicy(model_cfg)
    value = PendulumValue(model_cfg)
    
    # 4. 配置PPO参数
    ppo_cfg = PPOCfg()
    ppo_cfg.device = "cpu"
    ppo_cfg.memory_size = 200
    ppo_cfg.rollouts = 32
    ppo_cfg.learning_epochs = 4
    ppo_cfg.mini_batches = 2
    ppo_cfg.learning_rate = 3e-4
    ppo_cfg.mixed_precision = False
    
    # 5. 创建PPO智能体
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # 6. 创建训练器配置
    trainer_cfg = TrainerConfig(
        timesteps=1000,
        headless=True,
        disable_progressbar=False
    )
    
    # 7. 创建训练器并开始训练
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    trainer.train()
    
    # 8. 保存模型
    agent.save("./trained_model.pt")
    print("训练完成！")

if __name__ == "__main__":
    main()
```

# 软件架构

```
AquaML 强化学习框架架构图

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AquaML Framework                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                            Core - 协调器系统                                        ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ││
│  │  │   Coordinator   │  │   Device Info   │  │   Registry      │  │   Tensor Tool   │  ││
│  │  │   全局协调器     │  │   设备管理       │  │   组件注册       │  │   张量工具       │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  ││
│  │                                                                                     ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐││
│  │  │                           Managers - 管理器系统                                  │││
│  │  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │││
│  │  │ │Agent Manager│ │Model Manager│ │Environment  │ │Data Manager │ │File System  │ │││
│  │  │ │智能体管理    │ │模型管理      │ │Manager      │ │数据管理      │ │Manager      │ │││
│  │  │ │             │ │             │ │环境管理      │ │             │ │文件系统管理  │ │││
│  │  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │││
│  │  │                                                                                 │││
│  │  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │││
│  │  │ │Communicator │ │Data Unit    │ │Runner       │ │   Plugin    │                │││
│  │  │ │Manager      │ │Manager      │ │Manager      │ │   System    │                │││
│  │  │ │通信管理      │ │数据单元管理  │ │运行器管理    │ │   插件系统   │                │││
│  │  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘                │││
│  │  └─────────────────────────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                          Environment - 环境系统                                     ││
│  │  ┌─────────────────┐  ┌─────────────────────────────────────────────────────────────┐││
│  │  │   Base Env      │  │                  Wrappers - 环境包装器                      │││
│  │  │   环境基类       │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │││
│  │  │                 │  │  │ Gymnasium   │ │ Isaac Lab   │ │    Brax     │ │ Petting│ │││
│  │  │                 │  │  │ Wrapper     │ │ Wrapper     │ │  Wrapper    │ │  Zoo   │ │││
│  │  │                 │  │  │             │ │             │ │             │ │ Wrapper│ │││
│  │  └─────────────────┘  │  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │││
│  │                       └─────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                          Learning - 学习系统                                        ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐││
│  │  │                           Model - 模型系统                                      │││
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │││
│  │  │  │   Base Model    │  │  Gaussian Model │  │   Model Config  │                  │││
│  │  │  │   模型基类       │  │  高斯模型        │  │   模型配置       │                  │││
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │││
│  │  └─────────────────────────────────────────────────────────────────────────────────┘││
│  │                                                                                     ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐││
│  │  │                     Reinforcement - 强化学习算法                                │││
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │││
│  │  │  │   Base Agent    │  │   On-Policy     │  │   Off-Policy    │                  │││
│  │  │  │   智能体基类     │  │   在策略算法     │  │   离策略算法     │                  │││
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │││
│  │  └─────────────────────────────────────────────────────────────────────────────────┘││
│  │                                                                                     ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐││
│  │  │                        Trainers - 训练器系统                                    │││
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │││
│  │  │  │  Base Trainer   │  │Sequential Trainer│  │  Trainer Config │                  │││
│  │  │  │  训练器基类      │  │  顺序训练器      │  │  训练器配置      │                  │││
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │││
│  │  └─────────────────────────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                          Data - 数据处理系统                                        ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ││
│  │  │   Base Unit     │  │   Tensor Unit   │  │   NumPy Unit    │  │   Config Status │  ││
│  │  │   数据单元基类   │  │   张量数据单元   │  │   NumPy数据单元  │  │   配置状态       │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                          Config - 配置系统                                          ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ││
│  │  │  Config Class   │  │ Config Manager  │  │   Defaults      │  │    Schemas      │  ││
│  │  │  配置类装饰器    │  │  配置管理器      │  │   默认配置       │  │    配置架构      │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                          Utils - 工具模块                                           ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ││
│  │  │ Preprocessors   │  │   Schedulers    │  │  TensorBoard    │  │  File System    │  ││
│  │  │ 预处理器         │  │   调度器        │  │  Manager        │  │  文件系统工具    │  ││
│  │  │                 │  │                 │  │  TensorBoard管理 │  │                 │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## 数据流向图

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   环境       │───▶│   环境包装器     │───▶│   数据预处理     │───▶│   智能体        │
│ Environment │    │   Wrappers      │    │  Preprocessors  │    │   Agent         │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
       ▲                                                                   │
       │                                                                   ▼
       │           ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
       │           │   模型管理器     │◀───│   训练器        │◀───│   学习算法      │
       │           │ Model Manager   │    │   Trainer       │    │  RL Algorithm   │
       │           └─────────────────┘    └─────────────────┘    └─────────────────┘
       │                    ▲                       │
       │                    │                       ▼
       │           ┌─────────────────┐    ┌─────────────────┐
       └───────────│   协调器        │◀───│   数据管理器     │
                   │  Coordinator    │    │  Data Manager   │
                   └─────────────────┘    └─────────────────┘
```

更多内容参考文件doc/Architecture.md