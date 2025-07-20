#!/usr/bin/env python3
"""
SARSA (State-Action-Reward-State-Action) training example for discrete control environments.

This example demonstrates how to use the SARSA algorithm from AquaML
to train on-policy agents in discrete environments like CartPole, FrozenLake, etc.

SARSA is an on-policy temporal difference learning algorithm that updates Q-values
using the actual next action taken by the policy (unlike Q-Learning which uses the max action).
This makes SARSA more conservative and often more stable in stochastic environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.off_policy.sarsa import SARSA, SARSACfg
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


# Define a simple Q-network for function approximation
class SARSAQNetwork(Model):
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 4, action_dim: int = 2, epsilon: float = 0.1):
        super().__init__(model_cfg)
        self.epsilon = epsilon  # For epsilon-greedy policy
        self.action_dim = action_dim
        self.action_space_size = action_dim  # Required for random action generation
        
        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        q_values = self.q_net(states)
        return {"q_values": q_values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Epsilon-greedy action selection (important for SARSA)"""
        outputs = self.compute(data_dict)
        q_values = outputs["q_values"]
        
        batch_size = q_values.shape[0]
        
        # Epsilon-greedy action selection
        if self.training:
            # During training, use epsilon-greedy
            random_mask = torch.rand(batch_size, device=q_values.device) < self.epsilon
            random_actions = torch.randint(0, self.action_dim, (batch_size,), device=q_values.device)
            greedy_actions = torch.argmax(q_values, dim=1)
            
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            # During evaluation, use greedy actions
            actions = torch.argmax(q_values, dim=1)
        
        outputs["actions"] = actions.unsqueeze(1)
        return outputs
    
    def random_act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate random actions"""
        batch_size = list(data_dict.values())[0].shape[0]
        random_actions = torch.randint(
            0, self.action_dim, (batch_size, 1), 
            device=list(data_dict.values())[0].device
        )
        return {"actions": random_actions}
    
    def select_action(self, q_values: torch.Tensor) -> torch.Tensor:
        """Custom action selection (epsilon-greedy for SARSA)"""
        return self.act({"q_values": q_values})["actions"]


# Define a tabular Q-function for discrete environments like FrozenLake
class TabularSARSAPolicy(Model):
    def __init__(self, model_cfg: ModelCfg, state_space_size: int = 16, action_space_size: int = 4, epsilon: float = 0.1):
        super().__init__(model_cfg)
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = torch.zeros(1, state_space_size, action_space_size, device=model_cfg.device)
        
    def table(self):
        """Return the Q-table (required for tabular SARSA)"""
        return self.q_table
    
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Get Q-values for current states
        state_indices = states.long()
        q_values = self.q_table[0, state_indices]
        
        return {"q_values": q_values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Epsilon-greedy action selection for tabular SARSA"""
        outputs = self.compute(data_dict)
        q_values = outputs["q_values"]
        
        batch_size = q_values.shape[0]
        
        # Epsilon-greedy action selection
        if self.training:
            random_mask = torch.rand(batch_size, device=q_values.device) < self.epsilon
            random_actions = torch.randint(0, self.action_space_size, (batch_size,), device=q_values.device)
            greedy_actions = torch.argmax(q_values, dim=1)
            
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            actions = torch.argmax(q_values, dim=1)
        
        outputs["actions"] = actions.unsqueeze(1)
        return outputs
    
    def random_act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate random actions"""
        batch_size = list(data_dict.values())[0].shape[0]
        random_actions = torch.randint(
            0, self.action_space_size, (batch_size, 1), 
            device=list(data_dict.values())[0].device
        )
        return {"actions": random_actions}


def main():
    """Main training function"""
    print("Starting SARSA training for discrete control...")
    
    # Environment setup - choose between function approximation and tabular
    use_tabular = True  # Set to False for function approximation with CartPole
    
    if use_tabular:
        # Use a discrete environment like FrozenLake for tabular SARSA
        env = GymnasiumWrapper("FrozenLake-v1", render_mode=None)
        env_name = "FrozenLake-v1"
        obs_dim = 16  # FrozenLake has 16 states
        action_dim = 4  # 4 actions (up, down, left, right)
        print(f"Using tabular SARSA with {env_name}")
    else:
        # Use CartPole for function approximation
        env = GymnasiumWrapper("CartPole-v1")
        env_name = "CartPole-v1"
        obs_dim = 4  # CartPole observation dimension
        action_dim = 2  # 2 actions (left, right)
        print(f"Using function approximation SARSA with {env_name}")
    
    print(f"Environment: {env_name}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create model configuration
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # Create policy model
    if use_tabular:
        policy = TabularSARSAPolicy(model_cfg, obs_dim, action_dim, epsilon=0.1)
    else:
        policy = SARSAQNetwork(model_cfg, obs_dim, action_dim, epsilon=0.1)
    
    # Configure SARSA parameters
    sarsa_cfg = SARSACfg()
    sarsa_cfg.device = "cpu"
    sarsa_cfg.learning_rate = 0.1 if use_tabular else 0.001  # Higher LR for tabular
    sarsa_cfg.discount_factor = 0.99
    sarsa_cfg.random_timesteps = 1000  # Initial exploration
    sarsa_cfg.learning_starts = 0
    sarsa_cfg.memory_size = 10000
    sarsa_cfg.mixed_precision = False
    
    print(f"SARSA Configuration:")
    print(f"  - Learning rate: {sarsa_cfg.learning_rate}")
    print(f"  - Discount factor: {sarsa_cfg.discount_factor}")
    print(f"  - Random timesteps: {sarsa_cfg.random_timesteps}")
    print(f"  - Memory size: {sarsa_cfg.memory_size}")
    
    # Create SARSA agent
    models = {"policy": policy}
    agent = SARSA(
        models=models,
        cfg=sarsa_cfg,
        observation_space={"state": obs_dim},
        action_space={"action": action_dim}
    )
    
    # Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=50000 if use_tabular else 20000,
        headless=True,
        disable_progressbar=False
    )
    
    print(f"Training for {trainer_cfg.timesteps} timesteps...")
    
    # Create trainer and start training
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        print("This might be expected with some environments - adjust hyperparameters if needed")
    
    # Save model
    model_path = f"./trained_sarsa_{'tabular' if use_tabular else 'function'}_model.pt"
    agent.save(model_path)
    print(f"Training completed! Model saved to {model_path}")
    
    # Demonstrate policy evaluation
    print("\n" + "="*60)
    print("Policy Evaluation:")
    print("="*60)
    
    # Test the trained policy
    agent.policy.eval()
    test_episodes = 5
    total_rewards = []
    
    print(f"Testing policy for {test_episodes} episodes...")
    
    with torch.no_grad():
        for episode in range(test_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0
            max_steps = 1000 if use_tabular else 500
            
            while not done and steps < max_steps:
                obs_dict = {"state": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
                action_dict = agent.act(obs_dict, 0, 0)  # timestep=0 for evaluation
                action = action_dict["actions"].squeeze().item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Steps = {steps}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Print algorithm characteristics
    print("\n" + "="*60)
    print("SARSA Algorithm Characteristics:")
    print("="*60)
    print("1. On-policy: SARSA learns the value of the policy being followed")
    print("2. Uses actual next action: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]")
    print("3. More conservative than Q-Learning in stochastic environments")
    print("4. Better suited for environments where exploration can be dangerous")
    print("5. Converges to the optimal policy under the current exploration strategy")
    print("6. The policy improvement happens through the action selection mechanism (e.g., epsilon-greedy)")
    
    if use_tabular:
        print("\nTabular SARSA:")
        print("- Direct Q-table updates")
        print("- Guaranteed convergence with proper learning rate decay")
        print("- Suitable for small discrete state/action spaces")
        print("- Memory efficient for small problems")
    else:
        print("\nFunction Approximation SARSA:")
        print("- Neural network approximates Q-function")
        print("- Scales to large state/action spaces")
        print("- May require experience replay for stability")
        print("- More flexible but potentially less stable")
    
    print("="*60)


if __name__ == "__main__":
    main()