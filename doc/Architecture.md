# AquaML 框架架构文档

## 概述

AquaML是一个模块化的强化学习框架，专门为机器人学习任务设计。框架名称寓意"像水一样适应各种环境"，采用字典式数据结构和协调器模式，提供统一的组件管理和设备管理能力。

## 架构图

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

## 核心组件详解

### 1. 协调器系统 (Core)

协调器系统是AquaML的核心控制层，负责全局资源管理和组件协调。

#### 主要组件：
- **coordinator.py** (`/AquaML/core/coordinator.py`): 主协调器，全局设备管理和日志配置
- **device_info.py** (`/AquaML/core/device_info.py`): GPU设备检测和最优设备选择
- **registry.py** (`/AquaML/core/registry.py`): 组件注册和管理
- **tensor_tool.py** (`/AquaML/core/tensor_tool.py`): 张量操作工具

#### 管理器模块 (Managers):
- **agent_manager.py**: 智能体生命周期管理
- **model_manager.py**: 模型保存、加载和版本控制
- **environment_manager.py**: 环境实例管理和资源分配
- **data_manager.py**: 数据收集和批处理管理
- **file_system_manager.py**: 文件系统操作和实验管理
- **communicator_manager.py**: 通信管理器
- **data_unit_manager.py**: 数据单元管理器
- **runner_manager.py**: 运行器管理器

### 2. 环境系统 (Environment)

环境系统提供统一的接口来集成多种强化学习环境。

#### 核心组件：
- **base_env.py** (`/AquaML/environment/base_env.py`): 环境基类，定义字典式数据接口
- **gymnasium_envs.py** (`/AquaML/environment/gymnasium_envs.py`): Gymnasium环境直接封装
- **gymnasium_vector_envs.py** (`/AquaML/environment/gymnasium_vector_envs.py`): Gymnasium向量环境支持
- **virtual_env.py** (`/AquaML/environment/virtual_env.py`): 虚拟环境实现

#### 环境包装器 (Wrappers):
- **base.py** (`/AquaML/environment/wrappers/base.py`): 包装器基类
- **gymnasium_envs.py** (`/AquaML/environment/wrappers/gymnasium_envs.py`): Gymnasium适配器
- **isaaclab_envs.py** (`/AquaML/environment/wrappers/isaaclab_envs.py`): Isaac Lab环境适配
- **brax_envs.py** (`/AquaML/environment/wrappers/brax_envs.py`): Brax物理仿真环境适配
- **utils.py** (`/AquaML/environment/wrappers/utils.py`): 包装器工具函数

#### 支持的环境类型：
| 环境类型 | 单智能体 | 多智能体 | 特殊功能 |
|---------|---------|---------|---------|
| Gymnasium | ✅ | ✅ | 预配置环境，向量化支持 |
| Isaac Lab | ✅ | ✅ | Policy/Critic分离 |
| Brax | ✅ | ❌ | 物理仿真特性 |
| PettingZoo | 🔄 | 🔄 | 标准多智能体（规划中） |

### 3. 学习系统 (Learning)

学习系统包含模型、强化学习算法和训练器三个子系统。

#### 模型系统 (Model):
- **base.py** (`/AquaML/learning/model/base.py`): 模型基类，支持字典式输入输出
- **gaussian.py** (`/AquaML/learning/model/gaussian.py`): 高斯策略模型
- **model_cfg.py** (`/AquaML/learning/model/model_cfg.py`): 模型配置类

#### 强化学习算法 (Reinforcement):
- **base.py** (`/AquaML/learning/reinforcement/base.py`): 智能体基类
- **on_policy/** (`/AquaML/learning/reinforcement/on_policy/`): 在策略算法目录
  - **ppo.py**: PPO算法实现
- **off_policy/** (`/AquaML/learning/reinforcement/off_policy/`): 离策略算法目录（预留扩展）

#### PPO算法特性：
- ✅ 修复GAE计算时机
- ✅ KL散度自适应学习率调度
- ✅ 预处理器支持(状态/值标准化)
- ✅ 增强的数据收集和批处理
- ✅ 详细的训练监控和日志

#### 训练器系统 (Trainers):
- **base.py** (`/AquaML/learning/trainers/base.py`): 训练器基类
- **sequential.py** (`/AquaML/learning/trainers/sequential.py`): 顺序训练器实现

#### 扩展模块:
- **learning/base.py** (`/AquaML/learning/base.py`): 学习系统基础类
- **teacher_student/** (`/AquaML/learning/teacher_student/`): 师生学习框架（扩展功能）

### 4. 数据处理系统 (Data)

数据处理系统提供统一的数据格式和处理流程。

#### 核心组件：
- **base_unit.py** (`/AquaML/data/base_unit.py`): 数据单元基类
- **tensor_unit.py** (`/AquaML/data/tensor_unit.py`): PyTorch张量数据单元
- **numpy_unit.py** (`/AquaML/data/numpy_unit.py`): NumPy数组数据单元
- **cfg_status.py** (`/AquaML/data/cfg_status.py`): 配置状态管理

#### 数据格式规范：
- 所有数据形状规范为: `(num_machines, num_envs, feature_dim)`
- 支持字典式数据结构
- 自动类型转换和设备管理

### 5. 配置系统 (Config)

配置系统提供灵活的参数管理和配置验证。

#### 核心组件：
- **configclass.py** (`/AquaML/config/configclass.py`): 配置类装饰器
- **manager.py** (`/AquaML/config/manager.py`): 配置管理器
- **defaults/**: 默认配置文件夹
- **schemas/**: 配置架构验证

### 6. 工具模块 (Utils)

工具模块提供各种辅助功能和实用程序。

#### 核心工具：
- **preprocessors.py** (`/AquaML/utils/preprocessors.py`): 状态和值预处理器
- **schedulers.py** (`/AquaML/utils/schedulers.py`): 学习率调度器
- **tensorboard_manager.py** (`/AquaML/utils/tensorboard_manager.py`): TensorBoard集成
- **file_system/**: 文件系统工具和实验管理
- **array.py** (`/AquaML/utils/array.py`): 数组操作工具
- **dict.py** (`/AquaML/utils/dict.py`): 字典操作工具
- **string.py** (`/AquaML/utils/string.py`): 字符串处理工具
- **tool.py** (`/AquaML/utils/tool.py`): 通用工具函数
- **collector/**: 数据收集器模块

## 架构特点

### 1. 字典式数据架构
- 原生支持复杂的观察和动作空间
- 统一的数据格式处理不同环境
- 灵活的数据键值对管理

### 2. 模块化设计
- 环境、智能体、训练器相互独立
- 易于扩展和自定义
- 清晰的接口定义

### 3. 协调器模式
- 统一的组件管理和设备管理
- 智能的CPU/GPU设备选择
- 全局资源协调

### 4. 环境兼容性
- 支持多种主流强化学习环境
- 自动环境类型检测和包装
- 高性能数据转换 (< 1ms per step)

## 核心工作流程

### 1. 环境初始化
```python
env = GymnasiumWrapper("Pendulum-v1")
```

### 2. 模型创建
```python
policy = PendulumPolicy(model_cfg)
value = PendulumValue(model_cfg)
models = {"policy": policy, "value": value}
```

### 3. 智能体配置
```python
ppo_cfg = PPOCfg()
agent = PPO(models, ppo_cfg)
```

### 4. 训练执行
```python
trainer_cfg = TrainerConfig(timesteps=10000)
trainer = SequentialTrainer(env, agent, trainer_cfg)
trainer.train()
```

### 5. 模型管理
```python
agent.save("./trained_model.pt")
agent.load("./trained_model.pt")
```

## 性能特点

- **高性能**: 数据转换开销 < 1ms per step
- **内存高效**: 智能的批处理和内存管理
- **设备优化**: 自动选择最优计算设备
- **并行支持**: 多环境并行训练

## 扩展性

AquaML框架设计时充分考虑了扩展性：

1. **自定义环境**: 继承BaseEnv实现新环境
2. **自定义算法**: 继承Agent基类实现新算法
3. **自定义模型**: 继承Model基类实现新网络结构
4. **自定义训练器**: 继承BaseTrainer实现新训练流程

## 与其他框架的比较

### 与SKRL的相似性：
- 简单的API设计
- 灵活的配置管理
- 模块化架构

### AquaML的独特优势：
- 字典式数据架构
- 协调器统一管理
- 机器人学习优化
- 更广泛的环境支持

## 文件结构映射

```
AquaML/
├── core/                    # 协调器系统
│   ├── coordinator.py       # 主协调器
│   ├── coordinator_backup.py # 协调器备份
│   ├── device_info.py       # 设备管理
│   ├── registry.py          # 组件注册
│   ├── tensor_tool.py       # 张量工具
│   ├── exceptions.py        # 异常定义
│   ├── lifecycle.py         # 生命周期管理
│   └── managers/            # 各类管理器
│       ├── agent_manager.py
│       ├── model_manager.py
│       ├── environment_manager.py
│       ├── data_manager.py
│       ├── file_system_manager.py
│       ├── communicator_manager.py
│       ├── data_unit_manager.py
│       └── runner_manager.py
├── environment/             # 环境系统
│   ├── base_env.py          # 环境基类
│   ├── gymnasium_envs.py    # Gymnasium环境
│   ├── gymnasium_vector_envs.py # 向量环境
│   ├── virtual_env.py       # 虚拟环境
│   └── wrappers/            # 环境包装器
│       ├── base.py
│       ├── gymnasium_envs.py
│       ├── isaaclab_envs.py
│       ├── brax_envs.py
│       └── utils.py
├── learning/                # 学习系统
│   ├── base.py              # 学习基类
│   ├── model/               # 模型系统
│   │   ├── base.py
│   │   ├── gaussian.py
│   │   └── model_cfg.py
│   ├── reinforcement/       # 强化学习算法
│   │   ├── base.py          # 智能体基类
│   │   ├── on_policy/       # 在策略算法
│   │   │   └── ppo.py       # PPO算法实现
│   │   └── off_policy/      # 离策略算法（预留扩展）
│   ├── trainers/            # 训练器系统
│   │   ├── base.py
│   │   └── sequential.py
│   └── teacher_student/     # 师生学习框架
├── data/                    # 数据处理系统
│   ├── base_unit.py
│   ├── tensor_unit.py
│   ├── numpy_unit.py
│   └── cfg_status.py
├── config/                  # 配置系统
│   ├── configclass.py
│   ├── manager.py
│   ├── defaults/
│   └── schemas/
├── utils/                   # 工具模块
│   ├── preprocessors.py
│   ├── schedulers.py
│   ├── tensorboard_manager.py
│   ├── array.py
│   ├── dict.py
│   ├── string.py
│   ├── tool.py
│   ├── collector/
│   │   └── concat.py
│   └── file_system/
│       ├── base_file_system.py
│       ├── default_file_system.py
│       └── experiment_utils.py
├── plugin/                  # 插件系统
└── enum.py                  # 枚举定义
```

这个架构设计使得AquaML既保持了简单易用的API，又具备了强大的扩展能力和环境兼容性，特别适合机器人学习任务的需求。