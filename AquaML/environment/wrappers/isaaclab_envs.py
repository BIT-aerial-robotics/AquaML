"""
Isaac Lab环境包装器

基于开源实现，适配AquaML框架的字典数据格式和风格
"""

from typing import Any, Dict, Tuple, Union, Mapping
import numpy as np
import gymnasium
import torch

from .base import Wrapper, MultiAgentEnvWrapper
from AquaML.data import unitCfg
from AquaML import coordinator


@coordinator.registerEnv
class IsaacLabWrapper(Wrapper):
    """Isaac Lab单智能体环境包装器
    
    基于开源实现，适配AquaML的字典数据格式
    """
    
    def __init__(self, env: Any) -> None:
        """初始化Isaac Lab包装器
        
        Args:
            env: Isaac Lab环境实例
        """
        self.env = env
        super().__init__(env)
        
        self._reset_once = True
        self._observations = None
        self._info = {}
        
        # Isaac Lab环境总是向量化的
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # 重新设置配置
        self._setup_aquaml_configs()
    
    def _setup_aquaml_configs(self):
        """设置Isaac Lab特定的数据配置"""
        # 观察空间配置 - Isaac Lab使用policy观察
        obs_shape = self._get_space_shape(self.observation_space)
        self.observation_cfg_ = {
            'state': unitCfg(
                name='state',
                dtype=np.float32,
                single_shape=obs_shape,
            ),
        }
        
        # 如果有critic观察，也添加
        if self.state_space is not None:
            state_shape = self._get_space_shape(self.state_space)
            self.observation_cfg_['critic_state'] = unitCfg(
                name='critic_state',
                dtype=np.float32,
                single_shape=state_shape,
            )
        
        # 动作空间配置
        action_shape = self._get_space_shape(self.action_space)
        self.action_cfg_ = {
            'action': unitCfg(
                name='action',
                dtype=np.float32,
                single_shape=action_shape,
            ),
        }
        
        # 奖励配置
        self.reward_cfg_ = {
            'reward': unitCfg(
                name='reward',
                dtype=np.float32,
                single_shape=(1,),
            ),
        }
    
    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """状态空间"""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except (KeyError, AttributeError):
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None
    
    @property
    def observation_space(self) -> gymnasium.Space:
        """观察空间"""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except (KeyError, AttributeError):
            return self._unwrapped.observation_space["policy"]
    
    @property
    def action_space(self) -> gymnasium.Space:
        """动作空间"""
        try:
            return self._unwrapped.single_action_space
        except AttributeError:
            return self._unwrapped.action_space
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Any]:
        """重置环境
        
        Returns:
            (observation_dict, info): 观察字典和信息
        """
        if self._reset_once:
            observations, self._info = self.env.reset()
            
            # 处理policy观察
            policy_obs = observations["policy"]
            if isinstance(policy_obs, torch.Tensor):
                policy_obs = policy_obs.detach().cpu().numpy()
            else:
                policy_obs = np.array(policy_obs)
            
            # 转换为AquaML格式 (num_machines, num_envs, feature_dim)
            if policy_obs.ndim == 2:  # (num_envs, feature_dim)
                policy_obs = np.expand_dims(policy_obs, axis=0)  # (1, num_envs, feature_dim)
            else:
                policy_obs = policy_obs.reshape(1, self.num_envs, -1)
            
            self._observations = {'state': policy_obs.astype(np.float32)}
            
            # 如果有critic观察，也添加
            if "critic" in observations:
                critic_obs = observations["critic"]
                if isinstance(critic_obs, torch.Tensor):
                    critic_obs = critic_obs.detach().cpu().numpy()
                else:
                    critic_obs = np.array(critic_obs)
                
                if critic_obs.ndim == 2:
                    critic_obs = np.expand_dims(critic_obs, axis=0)
                else:
                    critic_obs = critic_obs.reshape(1, self.num_envs, -1)
                
                self._observations['critic_state'] = critic_obs.astype(np.float32)
            
            self._reset_once = False
        
        return self._observations, self._info
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, Any
    ]:
        """执行一步
        
        Args:
            action: 动作字典，格式为 {'action': np.ndarray}
            
        Returns:
            (next_observation_dict, reward_dict, terminated, truncated, info)
        """
        # 从AquaML格式提取动作
        action_data = action['action']
        
        # 转换动作维度 (num_machines, num_envs, feature_dim) -> (num_envs, feature_dim)
        if action_data.ndim == 3:
            action_data = action_data[0]  # 去掉num_machines维度
        
        # 转换为torch tensor并处理设备
        try:
            if not isinstance(action_data, torch.Tensor):
                action_tensor = torch.from_numpy(action_data)
            else:
                action_tensor = action_data
            
            # 安全的设备转换
            if action_tensor.device != self.device:
                action_tensor = action_tensor.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to convert action to tensor on device {self.device}: {e}")
        
        # 执行步骤
        observations, reward, terminated, truncated, self._info = self.env.step(action_tensor)
        
        # 处理policy观察
        policy_obs = observations["policy"]
        if isinstance(policy_obs, torch.Tensor):
            policy_obs = policy_obs.detach().cpu().numpy()
        else:
            policy_obs = np.array(policy_obs)
        
        if policy_obs.ndim == 2:
            policy_obs = np.expand_dims(policy_obs, axis=0)
        else:
            policy_obs = policy_obs.reshape(1, self.num_envs, -1)
        
        self._observations = {'state': policy_obs.astype(np.float32)}
        
        # 如果有critic观察，也添加
        if "critic" in observations:
            critic_obs = observations["critic"]
            if isinstance(critic_obs, torch.Tensor):
                critic_obs = critic_obs.detach().cpu().numpy()
            else:
                critic_obs = np.array(critic_obs)
            
            if critic_obs.ndim == 2:
                critic_obs = np.expand_dims(critic_obs, axis=0)
            else:
                critic_obs = critic_obs.reshape(1, self.num_envs, -1)
            
            self._observations['critic_state'] = critic_obs.astype(np.float32)
        
        # 转换奖励到AquaML格式
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        else:
            reward = np.array(reward)
        
        reward = reward.reshape(1, self.num_envs, 1)  # (1, num_envs, 1)
        reward_dict = {'reward': reward.astype(np.float32)}
        
        # 转换终止标志到AquaML格式
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        else:
            terminated = np.array(terminated)
        
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.detach().cpu().numpy()
        else:
            truncated = np.array(truncated)
        
        terminated = terminated.reshape(1, self.num_envs)  # (1, num_envs)
        truncated = truncated.reshape(1, self.num_envs)    # (1, num_envs)
        
        return self._observations, reward_dict, terminated, truncated, self._info
    
    def render(self, *args, **kwargs) -> None:
        """渲染环境"""
        # Isaac Lab环境通常不支持渲染
        return None
    
    def close(self) -> None:
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()


@coordinator.registerEnv
class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    """Isaac Lab多智能体环境包装器
    
    基于开源实现，适配AquaML的字典数据格式
    """
    
    def __init__(self, env: Any) -> None:
        """初始化Isaac Lab多智能体包装器
        
        Args:
            env: Isaac Lab多智能体环境实例
        """
        self.env = env
        super().__init__(env)
        
        self._reset_once = True
        self._observations = None
        self._info = {}
        
        # Isaac Lab环境总是向量化的
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # 重新设置配置
        self._setup_multi_agent_configs()
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置多智能体环境
        
        Returns:
            (observation_dict, info_dict): 观察字典和信息字典
        """
        if self._reset_once:
            observations, self._info = self.env.reset()
            
            self._observations = {}
            for agent_id, obs in observations.items():
                if isinstance(obs, torch.Tensor):
                    obs = obs.detach().cpu().numpy()
                else:
                    obs = np.array(obs)
                
                # 转换为AquaML格式
                if obs.ndim == 2:
                    obs = np.expand_dims(obs, axis=0)
                else:
                    obs = obs.reshape(1, self.num_envs, -1)
                
                self._observations[f"{agent_id}_state"] = obs.astype(np.float32)
            
            self._reset_once = False
        
        return self._observations, self._info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray], 
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Any]
    ]:
        """执行多智能体步骤
        
        Args:
            actions: 动作字典，格式为 {f"{agent_id}_action": np.ndarray}
            
        Returns:
            (observations, rewards, terminated, truncated, info)
        """
        # 转换动作格式
        env_actions = {}
        for agent_id in self.agents:
            action_key = f"{agent_id}_action"
            if action_key in actions:
                action_data = actions[action_key]
                
                # 转换动作维度
                if action_data.ndim == 3:
                    action_data = action_data[0]
                
                # 转换为torch tensor并处理设备
                try:
                    if not isinstance(action_data, torch.Tensor):
                        action_tensor = torch.from_numpy(action_data)
                    else:
                        action_tensor = action_data
                    
                    # 安全的设备转换
                    if action_tensor.device != self.device:
                        action_tensor = action_tensor.to(self.device)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert action for agent {agent_id} to tensor on device {self.device}: {e}")
                
                env_actions[agent_id] = action_tensor
        
        # 执行步骤
        observations, rewards, terminated, truncated, self._info = self.env.step(env_actions)
        
        # 转换观察
        observation_dict = {}
        for agent_id, obs in observations.items():
            if isinstance(obs, torch.Tensor):
                obs = obs.detach().cpu().numpy()
            else:
                obs = np.array(obs)
            
            if obs.ndim == 2:
                obs = np.expand_dims(obs, axis=0)
            else:
                obs = obs.reshape(1, self.num_envs, -1)
            
            observation_dict[f"{agent_id}_state"] = obs.astype(np.float32)
        
        # 转换奖励
        reward_dict = {}
        for agent_id, reward in rewards.items():
            if isinstance(reward, torch.Tensor):
                reward = reward.detach().cpu().numpy()
            else:
                reward = np.array(reward)
            
            reward = reward.reshape(1, self.num_envs, 1)
            reward_dict[f"{agent_id}_reward"] = reward.astype(np.float32)
        
        # 转换终止标志
        terminated_dict = {}
        truncated_dict = {}
        for agent_id in self.agents:
            if agent_id in terminated:
                term = terminated[agent_id]
                if isinstance(term, torch.Tensor):
                    term = term.detach().cpu().numpy()
                else:
                    term = np.array(term)
                terminated_dict[f"{agent_id}_terminated"] = term.reshape(1, self.num_envs)
            
            if agent_id in truncated:
                trunc = truncated[agent_id]
                if isinstance(trunc, torch.Tensor):
                    trunc = trunc.detach().cpu().numpy()
                else:
                    trunc = np.array(trunc)
                truncated_dict[f"{agent_id}_truncated"] = trunc.reshape(1, self.num_envs)
        
        self._observations = observation_dict
        
        return observation_dict, reward_dict, terminated_dict, truncated_dict, self._info
    
    def state(self) -> Union[torch.Tensor, None]:
        """获取环境状态"""
        try:
            state = self.env.state()
        except AttributeError:
            state = self._unwrapped.state()
        
        if state is not None:
            # 转换为AquaML格式
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()
            else:
                state = np.array(state)
            
            if state.ndim == 2:
                state = np.expand_dims(state, axis=0)
            else:
                state = state.reshape(1, self.num_envs, -1)
            
            return torch.from_numpy(state.astype(np.float32))
        
        return state
    
    def render(self, *args, **kwargs) -> None:
        """渲染环境"""
        return None
    
    def close(self) -> None:
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()