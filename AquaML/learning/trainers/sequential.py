"""Sequential trainer for AquaML framework

This module provides a sequential trainer that implements the training loop
for reinforcement learning agents in AquaML.
"""

from typing import Dict, Any, List, Optional, Union
import sys
import copy
try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
import torch
import numpy as np
from loguru import logger

from .base import BaseTrainer, TrainerConfig
from AquaML.environment.base_env import BaseEnv
from AquaML.learning.reinforcement.base import Agent
from AquaML.utils.tensorboard_manager import TensorboardManager
from AquaML.data import TensorUnit, NumpyUnit, unitCfg
from AquaML import coordinator


class SequentialTrainer(BaseTrainer):
    """Sequential trainer for AquaML framework with separated execution and training loops
    
    This trainer implements the new data flow architecture:
    1. Execution Loop (Real-time): Environment ↔ Model Compute ↔ Environment
    2. Data Collection: Store data in (num_env, steps, dims) format buffers
    3. Training Loop (Batch): Agent training from collected batch data
    
    Key features:
    - Execution loop and training process are separated
    - Data is collected to unified (num_env, steps, dims) format buffers
    - Supports multi-environment parallel execution and real-time data collection
    - Agents can access batch training data from data pools
    """
    
    def __init__(self, 
                 env: BaseEnv,
                 agents: Union[Agent, List[Agent]],
                 cfg: Optional[TrainerConfig] = None):
        """Initialize the sequential trainer
        
        Args:
            env: AquaML environment instance
            agents: Agent or list of agents to train
            cfg: Trainer configuration (optional)
        """
        if cfg is None:
            cfg = TrainerConfig()
            
        super().__init__(env, agents, cfg)
        
        # Initialize agents
        for agent in self.agents:
            agent.init(trainer_cfg=cfg)
        
        # Execution loop configuration - automatically read from agent config
        self.collect_interval = self._get_collect_interval_from_agent(cfg)
        self.execution_training_ratio = getattr(cfg, 'execution_training_ratio', 4)  # 4:1 ratio
        self.enable_parallel_training = getattr(cfg, 'enable_parallel_training', True)
        
        # Data collection statistics
        self.collected_steps = 0
        self.training_episodes = 0
            
        # Setup enhanced data collection buffers with new format
        self.data_unit_manager = coordinator.get_data_unit_manager()
        self._setup_enhanced_data_buffers()

        # Tensorboard setup
        self.tensorboard_manager = None
        if self.cfg.tensorboard:
            # Assume single agent for now
            log_dir = self.agents[0].experiment_dir
            self.tensorboard_manager = TensorboardManager(log_dir)
            
        logger.info("Sequential trainer initialized with enhanced data flow architecture")
    
    def _get_collect_interval_from_agent(self, cfg: TrainerConfig) -> int:
        """Get collect_interval from agent config, with fallback to trainer config
        
        Args:
            cfg: Trainer configuration
            
        Returns:
            collect_interval value
        """
        # First try to get from trainer config if explicitly set
        if hasattr(cfg, 'collect_interval') and cfg.collect_interval is not None:
            return cfg.collect_interval
            
        # Then try to get rollouts from first agent's config
        if self.agents and hasattr(self.agents[0], 'cfg'):
            agent_cfg = self.agents[0].cfg
            if hasattr(agent_cfg, 'rollouts'):
                logger.info(f"Using agent's rollouts ({agent_cfg.rollouts}) as collect_interval")
                return agent_cfg.rollouts
                
        # Default fallback
        default_interval = 200
        logger.info(f"Using default collect_interval: {default_interval}")
        return default_interval
        
    def _setup_enhanced_data_buffers(self) -> None:
        """Setup enhanced data buffers following (num_env, steps, dims) format"""
        env_info = self.env.getEnvInfo()
        num_envs = env_info['num_envs']
        
        # Use the collect_interval (which now automatically reads from agent's rollouts)
        rollouts = self.collect_interval
        
        # Setup observation buffers
        for obs_name, obs_cfg in env_info['observation_cfg'].items():
            buffer_name = f"observations_{obs_name}"
            self.data_unit_manager.setup_data_buffer(
                buffer_name=buffer_name,
                num_envs=num_envs,
                max_steps=rollouts,
                data_shape=obs_cfg['single_shape'],
                dtype=obs_cfg['dtype']
            )
        
        # Setup action buffers  
        for act_name, act_cfg in env_info['action_cfg'].items():
            buffer_name = f"actions_{act_name}"
            self.data_unit_manager.setup_data_buffer(
                buffer_name=buffer_name,
                num_envs=num_envs,
                max_steps=rollouts,
                data_shape=act_cfg['single_shape'],
                dtype=act_cfg['dtype']
            )
        
        # Setup essential buffers
        for buffer_name, dtype, shape in [
            ("rewards", np.float32, (1,)),
            ("terminated", np.bool_, (1,)),
            ("truncated", np.bool_, (1,)),
        ]:
            self.data_unit_manager.setup_data_buffer(
                buffer_name=buffer_name,
                num_envs=num_envs,
                max_steps=rollouts,
                data_shape=shape,
                dtype=dtype
            )
        
        logger.info(f"Enhanced data buffers setup for {num_envs} envs, {rollouts} rollouts")
    
    def _enhanced_data_collection(self, 
                                 observations: Dict[str, Any],
                                 actions: Dict[str, Any], 
                                 rewards: Any,
                                 terminated: Any,
                                 truncated: Any) -> bool:
        """Enhanced data collection to buffers in (num_env, steps, dims) format
        
        Args:
            observations: Environment observations
            actions: Actions taken
            rewards: Rewards received  
            terminated: Terminal flags
            truncated: Truncation flags
            
        Returns:
            True if rollout is complete (time to train)
        """
        # Collect observations
        for obs_name, obs_data in observations.items():
            buffer_name = f"observations_{obs_name}"
            if isinstance(obs_data, torch.Tensor):
                obs_data = obs_data.cpu().numpy()
            self.data_unit_manager.add_data_to_buffer(buffer_name, obs_data)
        
        # Collect actions - only collect actions that have corresponding buffers
        available_buffers = self.data_unit_manager.list_buffers()
        for act_name, act_data in actions.items():
            buffer_name = f"actions_{act_name}"
            if buffer_name in available_buffers:
                if isinstance(act_data, torch.Tensor):
                    act_data = act_data.cpu().numpy()
                self.data_unit_manager.add_data_to_buffer(buffer_name, act_data)
        
        # Collect rewards
        reward_values = self._extract_reward_values(rewards)
        buffer_full = self.data_unit_manager.add_data_to_buffer("rewards", reward_values)
        
        # Collect terminal and truncation flags
        self._collect_boolean_flags("terminated", terminated)
        self._collect_boolean_flags("truncated", truncated)
        
        self.collected_steps += 1
        
        return buffer_full  # Return True when rollout is complete
    
    def _extract_reward_values(self, rewards: Any) -> np.ndarray:
        """Extract reward values and format for buffer"""
        if isinstance(rewards, dict):
            reward_values = rewards.get('reward', list(rewards.values())[0])
        else:
            reward_values = rewards
        
        if isinstance(reward_values, torch.Tensor):
            reward_values = reward_values.cpu().numpy()
        
        # Ensure correct shape for (num_env, 1)
        if isinstance(reward_values, (int, float)):
            reward_values = np.array([[reward_values]])
        elif reward_values.ndim == 0:
            reward_values = reward_values.reshape(1, 1)
        elif reward_values.ndim == 1:
            reward_values = reward_values.reshape(-1, 1)
        
        return reward_values
    
    def _collect_boolean_flags(self, buffer_name: str, flags: Any) -> None:
        """Collect boolean flags to buffer"""
        if isinstance(flags, torch.Tensor):
            flags = flags.cpu().numpy()
        if isinstance(flags, (bool, int, float)):
            flags = np.array([[flags]], dtype=np.bool_)
        elif flags.ndim == 0:
            flags = flags.reshape(1, 1).astype(np.bool_)
        elif flags.ndim == 1:
            flags = flags.reshape(-1, 1).astype(np.bool_)
        
        self.data_unit_manager.add_data_to_buffer(buffer_name, flags)
    
    def _trigger_rollout_training(self) -> None:
        """Trigger training when rollout is complete (rollouts steps collected)"""
        try:
            # Get collected rollout data from all buffers
            rollout_data = {}
            for buffer_name in self.data_unit_manager.list_buffers():
                buffer_data = self.data_unit_manager.get_buffer_data(buffer_name)
                if buffer_data is not None:
                    rollout_data[buffer_name] = buffer_data
            
            if rollout_data:
                # Trigger standard agent training using collected rollout data
                for agent in self.agents:
                    # Use standard post_interaction which should handle the training
                    agent.post_interaction(timestep=self.collected_steps, timesteps=self.timesteps)
                
                # Clear buffers for next rollout
                self.data_unit_manager.clear_all_buffers()
                self.training_episodes += 1
                
                logger.debug(f"Rollout training {self.training_episodes} completed, buffers cleared")
                
        except Exception as e:
            logger.error(f"Error in rollout training: {e}")
        
    def train(self) -> None:
        """Train the agents using enhanced sequential training with rollout-based data collection
        
        This method implements the new data flow architecture:
        1. 🔄 Execution Loop: Environment ↔ Model Compute ↔ Environment  
        2. 📥 Data Collection: Store in (num_env, steps, dims) buffers
        3. 📊 Rollout Training: When rollouts steps collected, trigger training
        4. 🔧 Environment Adaptation: Unified wrapper handling
        """
        logger.info("Starting enhanced sequential training with rollout-based data collection")
        
        # Set agents to training mode for data collection
        for agent in self.agents:
            agent.set_running_mode("train")
            
        # Clear any existing data buffers
        self.data_unit_manager.clear_all_buffers()
            
        # Reset environment to get initial states
        states, infos = self.env.reset()
        
        # Convert to tensors for agents
        states_tensor = self._convert_to_tensors(states)
        
        # Main training loop
        timestep_range = range(self.initial_timestep, self.timesteps)
        if HAS_TQDM and not self.disable_progressbar:
            timestep_range = tqdm.tqdm(timestep_range, file=sys.stdout, desc="Training")
        
        for timestep in timestep_range:
            
            # Pre-interaction phase
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
                
            with torch.no_grad():
                # Generate actions from all agents
                actions_list = []
                for agent in self.agents:
                    agent_outputs = agent.act(states_tensor, timestep=timestep, timesteps=self.timesteps)
                    actions_list.append(agent_outputs)
                
                # For single agent, use the first (and only) action
                if len(self.agents) == 1:
                    actions_tensor = actions_list[0]
                else:
                    actions_tensor = self._combine_agent_actions(actions_list)
                
                actions_numpy = self._convert_to_numpy(actions_tensor)
                
                if 'actions' in actions_numpy and 'action' not in actions_numpy:
                    actions_numpy['action'] = actions_numpy['actions']
                
                next_states, rewards, terminated, truncated, infos = self.env.step(actions_numpy)
                
                if not self.headless:
                    try:
                        self.env.render()
                    except:
                        pass
                
                next_states_tensor = self._convert_to_tensors(next_states)
                
                if isinstance(rewards, dict):
                    rewards_tensor = self._convert_to_tensors(rewards)
                    rewards_value = rewards_tensor.get('reward', list(rewards_tensor.values())[0])
                else:
                    rewards_value = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                
                terminated_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.device)
                truncated_tensor = torch.tensor(truncated, dtype=torch.bool, device=self.device)
                
                # Standard agent transition recording (for compatibility)
                for agent in self.agents:
                    agent.record_transition(
                        states=states_tensor,
                        actions=actions_tensor,
                        rewards=rewards_value,
                        next_states=next_states_tensor,
                        terminated=terminated_tensor,
                        truncated=truncated_tensor,
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps
                    )
                
                # 📥 Enhanced Data Collection: Store in (num_env, steps, dims) buffers
                rollout_complete = self._enhanced_data_collection(
                    observations=next_states,
                    actions=actions_numpy,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated
                )
                
                # Legacy data unit update (for backward compatibility)
                self._update_data_units(
                    observations=next_states,
                    actions=actions_numpy,
                    rewards=rewards,
                    dones=terminated,
                    truncated=truncated
                )
                
                if self.environment_info in infos:
                    self._log_environment_info(infos[self.environment_info])
                    
            # 📊 Rollout Training: Check if rollout is complete and trigger training
            if rollout_complete:
                self._trigger_rollout_training()
            
            for agent in self.agents:
                # Only do post_interaction if not already handled by rollout training
                if not rollout_complete:
                    agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

                # Checkpoint and log
                if agent.checkpoint_interval > 0 and timestep % agent.checkpoint_interval == 0:
                    agent.write_checkpoint(timestep=timestep, timesteps=self.timesteps)

                if self.tensorboard_manager:
                    for tag, values in agent.tracking_data.items():
                        if values:
                            self.tensorboard_manager.write(tag, values[-1], timestep)
                    agent.tracking_data.clear()

            if terminated.any() or truncated.any():
                # update best checkpoint
                for agent in self.agents:
                    # A simple way to get episode reward
                    episode_reward = rewards_value.sum().item()
                    agent.update_best_checkpoint(timestep, episode_reward)
                with torch.no_grad():
                    states, infos = self.env.reset()
                    states_tensor = self._convert_to_tensors(states)
            else:
                states = next_states
                states_tensor = next_states_tensor

        # Final training from any remaining buffer data
        final_rollout_data = {}
        for buffer_name in self.data_unit_manager.list_buffers():
            buffer_data = self.data_unit_manager.get_buffer_data(buffer_name)
            if buffer_data is not None and buffer_data.shape[1] > 0:  # Has steps
                final_rollout_data[buffer_name] = buffer_data
        
        if final_rollout_data:
            logger.info("Processing final rollout data")
            for agent in self.agents:
                agent.post_interaction(timestep=self.timesteps, timesteps=self.timesteps)
        
        # Final checkpoint
        for agent in self.agents:
            agent.write_checkpoint(timestep=self.timesteps, timesteps=self.timesteps)

        if self.tensorboard_manager:
            self.tensorboard_manager.close()
        
        # Log enhanced training statistics
        logger.info(f"Enhanced training completed successfully")
        logger.info(f"Total collected steps: {self.collected_steps}")
        logger.info(f"Training episodes triggered: {self.training_episodes}")
        logger.info(f"Data collection efficiency: {self.collected_steps / self.timesteps:.2f}")
        
    def eval(self) -> None:
        """Evaluate the agents using the sequential evaluation loop
        
        This method is similar to train() but uses deterministic actions
        and doesn't update the agents.
        """
        logger.info("Starting sequential evaluation")
        
        # Set agents to evaluation mode
        for agent in self.agents:
            agent.set_running_mode("eval")
            
        # Reset environment to get initial states
        states, infos = self.env.reset()
        
        # Convert to tensors for agents
        states_tensor = self._convert_to_tensors(states)
        
        # Main evaluation loop
        timestep_range = range(self.initial_timestep, self.timesteps)
        if HAS_TQDM and not self.disable_progressbar:
            timestep_range = tqdm.tqdm(timestep_range, file=sys.stdout, desc="Evaluation")
        
        for timestep in timestep_range:
            
            # Pre-interaction phase
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
                
            with torch.no_grad():
                # Generate actions from all agents
                actions_list = []
                for agent in self.agents:
                    agent_outputs = agent.act(states_tensor, timestep=timestep, timesteps=self.timesteps)
                    
                    # Use deterministic actions for evaluation if available
                    if not self.stochastic_evaluation and isinstance(agent_outputs, dict):
                        actions_tensor = agent_outputs.get("mean_actions", agent_outputs.get("actions"))
                    else:
                        actions_tensor = agent_outputs.get("actions") if isinstance(agent_outputs, dict) else agent_outputs
                        
                    actions_list.append(actions_tensor)
                
                # For single agent, use the first (and only) action
                if len(self.agents) == 1:
                    actions_tensor = actions_list[0]
                else:
                    # For multiple agents, combine actions
                    actions_tensor = self._combine_agent_actions(actions_list)
                
                # Convert actions to numpy for environment
                actions_numpy = self._convert_to_numpy(actions_tensor)
                
                # Ensure actions have the right key structure for environment
                if 'actions' in actions_numpy and 'action' not in actions_numpy:
                    actions_numpy['action'] = actions_numpy['actions']
                
                # Step the environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions_numpy)
                
                # Render if not headless
                if not self.headless:
                    try:
                        self.env.render()
                    except:
                        pass  # Rendering might not be supported
                
                # Convert environment outputs to tensors
                next_states_tensor = self._convert_to_tensors(next_states)
                
                # Handle rewards - extract the reward value from the dictionary
                if isinstance(rewards, dict):
                    rewards_tensor = self._convert_to_tensors(rewards)
                    # Get the actual reward values
                    rewards_value = rewards_tensor.get('reward', list(rewards_tensor.values())[0])
                else:
                    rewards_value = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                
                terminated_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.device)
                truncated_tensor = torch.tensor(truncated, dtype=torch.bool, device=self.device)
                
                # Record transitions for logging (but not for training)
                for agent in self.agents:
                    agent.record_transition(
                        states=states_tensor,
                        actions=actions_tensor,
                        rewards=rewards_value,
                        next_states=next_states_tensor,
                        terminated=terminated_tensor,
                        truncated=truncated_tensor,
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps
                    )
                
                # Log environment info
                if self.environment_info in infos:
                    self._log_environment_info(infos[self.environment_info])
                    
            # Post-interaction phase (evaluation mode)
            for agent in self.agents:
                # In evaluation mode, we don't want to trigger training updates
                # So we can skip post_interaction or call it with care
                if hasattr(agent, 'post_interaction'):
                    try:
                        agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
                    except:
                        # If post_interaction fails in eval mode, just skip it
                        pass
                
            # Handle environment resets
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
                    states_tensor = self._convert_to_tensors(states)
            else:
                states = next_states
                states_tensor = next_states_tensor
                
        logger.info("Evaluation completed successfully")
        
    def _combine_agent_actions(self, actions_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine actions from multiple agents
        
        Args:
            actions_list: List of action dictionaries from different agents
            
        Returns:
            Combined action dictionary
        """
        if len(actions_list) == 1:
            return actions_list[0]
            
        # For multiple agents, we need to combine their actions
        # This is a simple concatenation approach
        combined_actions = {}
        
        for key in actions_list[0].keys():
            if key in ["actions", "action"]:
                # Combine action tensors
                action_tensors = [actions[key] for actions in actions_list]
                combined_actions[key] = torch.cat(action_tensors, dim=0)
            else:
                # For other keys, take the first one (or implement specific logic)
                combined_actions[key] = actions_list[0][key]
                
        return combined_actions
        
    def _log_environment_info(self, env_info: Dict[str, Any]):
        """Log environment information
        
        Args:
            env_info: Environment information dictionary
        """
        for key, value in env_info.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                for agent in self.agents:
                    if hasattr(agent, 'track_data'):
                        agent.track_data(f"Info / {key}", value.item())
            elif isinstance(value, (int, float)):
                for agent in self.agents:
                    if hasattr(agent, 'track_data'):
                        agent.track_data(f"Info / {key}", value)
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced trainer status including data flow metrics
        
        Returns:
            Enhanced status dictionary
        """
        base_status = self.get_status()
        
        enhanced_status = {
            **base_status,
            "data_flow_architecture": {
                "collect_interval": getattr(self, 'collect_interval', 200),
                "execution_training_ratio": getattr(self, 'execution_training_ratio', 4),
                "enable_parallel_training": getattr(self, 'enable_parallel_training', True),
                "collected_steps": getattr(self, 'collected_steps', 0),
                "training_episodes": getattr(self, 'training_episodes', 0),
                "buffer_status": self.data_unit_manager.get_status() if hasattr(self, 'data_unit_manager') else {}
            }
        }
        
        return enhanced_status
    
    def clear_enhanced_buffers(self) -> None:
        """Clear all enhanced data buffers"""
        if hasattr(self, 'data_unit_manager'):
            self.data_unit_manager.clear_all_buffers()
        if hasattr(self, 'collected_steps'):
            self.collected_steps = 0
        if hasattr(self, 'training_episodes'):
            self.training_episodes = 0
        logger.info("Enhanced data buffers cleared")
    
    def get_collected_buffer_data(self, buffer_name: str) -> Optional[np.ndarray]:
        """Get data from specific buffer
        
        Args:
            buffer_name: Name of the buffer
            
        Returns:
            Buffer data in (num_env, steps, dims) format
        """
        if hasattr(self, 'data_unit_manager'):
            return self.data_unit_manager.get_buffer_data(buffer_name)
        return None
    
    def list_available_buffers(self) -> List[str]:
        """List all available data buffers
        
        Returns:
            List of buffer names
        """
        if hasattr(self, 'data_unit_manager'):
            return self.data_unit_manager.list_buffers()
        return []