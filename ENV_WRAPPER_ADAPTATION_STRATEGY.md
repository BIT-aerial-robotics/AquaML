# AquaML环境包装器适配策略

## 核心差异分析

### skrl环境包装器特点：
1. **数据格式**: 使用tensor format，直接返回torch.Tensor
2. **接口设计**: 返回格式为 `(observation, reward, terminated, truncated, info)`
3. **数据扁平化**: 自动处理复杂space到tensor的转换
4. **向量化支持**: 原生支持向量化环境
5. **多agent支持**: 专门的MultiAgentEnvWrapper基类

### AquaML环境包装器特点：
1. **数据格式**: 使用字典格式，保持结构化数据
2. **接口设计**: 返回格式为 `(obs_dict, reward_dict, done, truncated, info)`
3. **维度规范**: 固定使用 `(num_machines, num_envs, feature_dim)` 维度
4. **配置驱动**: 通过 `observation_cfg_`, `action_cfg_`, `reward_cfg_` 配置数据结构

## 适配策略设计

### 1. 保持AquaML字典数据特性
- 所有环境包装器输出保持字典格式
- 自动处理skrl tensor格式到AquaML字典格式的转换
- 维护AquaML的数据维度规范

### 2. 创建适配层架构
```
skrl环境 -> AquaML适配包装器 -> AquaML统一接口
         (tensor格式)      (字典格式)
```

### 3. 核心组件设计

#### BaseWrapperAdapter (基础适配器)
- 继承AquaML的BaseEnv
- 内部封装skrl的Wrapper
- 处理tensor到字典的转换
- 统一维度处理

#### 具体环境适配器
- GymnasiumWrapperAdapter
- IsaacLabWrapperAdapter 
- BraxWrapperAdapter
- 等等...

### 4. 数据转换策略

#### 观察空间转换
```python
# skrl: torch.Tensor -> (batch_size, obs_dim)
# AquaML: Dict[str, np.ndarray] -> {"state": (1, 1, obs_dim)}
```

#### 动作空间转换
```python
# AquaML: {"action": (1, 1, action_dim)} 
# skrl: torch.Tensor -> (batch_size, action_dim)
```

#### 奖励转换
```python
# skrl: torch.Tensor -> (batch_size, 1)
# AquaML: {"reward": (1, 1, 1)}
```

### 5. 自动检测和包装机制
- 实现auto_wrap_env函数
- 自动检测环境类型
- 选择对应的适配器

## 实现优先级

### 高优先级 (核心功能)
1. BaseWrapperAdapter - 基础适配框架
2. GymnasiumWrapperAdapter - 最常用的Gym环境
3. auto_wrap_env - 自动包装函数

### 中优先级 (扩展功能)  
4. IsaacLabWrapperAdapter - 机器人仿真环境
5. BraxWrapperAdapter - 物理仿真环境
6. MultiAgentWrapperAdapter - 多智能体支持

### 低优先级 (特殊用途)
7. 其他专用环境适配器
8. 高级功能和优化