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


class SequentialTrainer(BaseTrainer):
    """Sequential trainer for AquaML framework
    
    This trainer implements the standard RL training loop:
    1. Reset environment
    2. For each timestep:
       - Pre-interaction
       - Generate actions
       - Step environment
       - Record transitions
       - Post-interaction
    3. Handle environment resets when episodes end
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
            
        logger.info("Sequential trainer initialized successfully")
        
    def train(self) -> None:
        """Train the agents using the sequential training loop
        
        This method implements the main training loop following the pattern:
        - Reset environment
        - For each timestep:
          - Pre-interaction (for all agents)
          - Generate actions (for all agents)
          - Step environment with actions
          - Record transitions (for all agents)
          - Post-interaction (for all agents)
        - Handle environment resets when episodes end
        """
        logger.info("Starting sequential training")
        
        # Set agents to training mode
        for agent in self.agents:
            agent.set_running_mode("train")
            
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
                    # For multiple agents, we need to combine actions
                    # This is a simplified approach - in practice, you might need
                    # more sophisticated action combination strategies
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
                
                # Record transitions for all agents
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
                
                # Update data units
                self._update_data_units(
                    observations=next_states,
                    actions=actions_numpy,
                    rewards=rewards,
                    dones=terminated,
                    truncated=truncated
                )
                
                # Log environment info
                if self.environment_info in infos:
                    self._log_environment_info(infos[self.environment_info])
                    
            # Post-interaction phase
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
                
            # Handle environment resets
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
                    states_tensor = self._convert_to_tensors(states)
            else:
                states = next_states
                states_tensor = next_states_tensor
                
        logger.info("Training completed successfully")
        
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