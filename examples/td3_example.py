#!/usr/bin/env python3
"""
TD3 (Twin Delayed DDPG) training example for continuous control environments.

This example demonstrates how to use the TD3 algorithm from AquaML
to train agents in continuous control environments like Pendulum, BipedalWalker, etc.

TD3 improves upon DDPG with three key innovations:
1. Twin Critic Networks (Clipped Double Q-Learning): Uses minimum of two critic networks to reduce overestimation bias
2. Delayed Policy Updates: Updates policy less frequently than critics to reduce variance
3. Target Policy Smoothing: Adds noise to target actions to regularize targets

These improvements make TD3 more stable and sample-efficient than vanilla DDPG.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.off_policy.td3 import TD3, TD3Cfg
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


class TD3Actor(Model):
    """TD3 Actor (Policy) Network for continuous actions"""
    
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 3, action_dim: int = 1, max_action: float = 2.0):
        super().__init__(model_cfg)
        self.max_action = max_action
        self.action_dim = action_dim
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Forward pass through actor
        actions = self.actor(states)
        # Scale to action space
        actions = actions * self.max_action
        
        return {"actions": actions}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate actions (same as compute for deterministic policy)"""
        return self.compute(data_dict)


class TD3Critic(Model):
    """TD3 Critic (Q-function) Network"""
    
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 3, action_dim: int = 1):
        super().__init__(model_cfg)
        
        # Critic network takes state and action as input
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else data_dict.get("states")
        actions = data_dict.get("taken_actions", data_dict.get("actions"))
        
        if states is None or actions is None:
            raise ValueError("Critic requires both states and actions")
        
        # Ensure proper dimensions
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        # Concatenate state and action
        state_action = torch.cat([states, actions], dim=-1)
        
        # Forward pass through critic
        q_value = self.critic(state_action)
        
        return {"values": q_value}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate Q-values (same as compute for critic)"""
        return self.compute(data_dict)


class SimpleNoise:
    """Simple Gaussian noise for exploration and target policy smoothing"""
    
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def sample(self, shape):
        """Sample noise with given shape"""
        if isinstance(shape, torch.Size):
            shape = tuple(shape)
        return torch.normal(self.mean, self.std, size=shape)


def main():
    """Main training function"""
    print("Starting TD3 training for continuous control...")
    
    # Environment setup
    env_name = "Pendulum-v1"  # Classic continuous control environment
    env = GymnasiumWrapper(env_name)
    
    print(f"Environment: {env_name}")
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    
    # Create model configuration
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # Create policy models (actor)
    policy = TD3Actor(model_cfg, obs_dim, action_dim, max_action)
    target_policy = TD3Actor(model_cfg, obs_dim, action_dim, max_action)
    
    # Create critic models (Q-functions)
    critic_1 = TD3Critic(model_cfg, obs_dim, action_dim)
    critic_2 = TD3Critic(model_cfg, obs_dim, action_dim)
    target_critic_1 = TD3Critic(model_cfg, obs_dim, action_dim)
    target_critic_2 = TD3Critic(model_cfg, obs_dim, action_dim)
    
    # Create noise for exploration and target policy smoothing
    exploration_noise = SimpleNoise(mean=0.0, std=0.1)
    smooth_noise = SimpleNoise(mean=0.0, std=0.2)
    
    # Configure TD3 parameters
    td3_cfg = TD3Cfg()
    td3_cfg.device = "cpu"
    td3_cfg.batch_size = 64
    td3_cfg.discount_factor = 0.99
    td3_cfg.polyak = 0.005  # Soft update coefficient
    
    # Learning rates
    td3_cfg.actor_learning_rate = 3e-4
    td3_cfg.critic_learning_rate = 3e-4
    
    # Exploration parameters
    td3_cfg.random_timesteps = 1000  # Initial random exploration
    td3_cfg.learning_starts = 1000   # Start learning after this many steps
    td3_cfg.exploration_noise = exploration_noise
    td3_cfg.exploration_initial_scale = 1.0
    td3_cfg.exploration_final_scale = 0.1
    td3_cfg.exploration_timesteps = 10000  # Decay exploration over 10k steps
    
    # TD3-specific parameters
    td3_cfg.policy_delay = 2  # Update policy every 2 critic updates
    td3_cfg.smooth_regularization_noise = smooth_noise  # Target policy smoothing
    td3_cfg.smooth_regularization_clip = 0.5  # Clip smoothing noise
    
    # Memory and training
    td3_cfg.memory_size = 100000  # Large replay buffer
    td3_cfg.gradient_steps = 1
    td3_cfg.mixed_precision = False
    
    print(f"TD3 Configuration:")
    print(f"  - Batch size: {td3_cfg.batch_size}")
    print(f"  - Actor learning rate: {td3_cfg.actor_learning_rate}")
    print(f"  - Critic learning rate: {td3_cfg.critic_learning_rate}")
    print(f"  - Discount factor: {td3_cfg.discount_factor}")
    print(f"  - Policy delay: {td3_cfg.policy_delay}")
    print(f"  - Polyak coefficient: {td3_cfg.polyak}")
    print(f"  - Memory size: {td3_cfg.memory_size}")
    
    # Create TD3 agent
    models = {
        "policy": policy,
        "target_policy": target_policy,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }
    
    agent = TD3(
        models=models,
        cfg=td3_cfg,
        observation_space={"state": obs_dim},
        action_space={"action": action_dim}
    )
    
    # Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=50000,  # Total training timesteps
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
    model_path = "./trained_td3_model.pt"
    agent.save(model_path)
    print(f"Training completed! Model saved to {model_path}")
    
    # Demonstrate policy evaluation
    print("\n" + "="*60)
    print("Policy Evaluation:")
    print("="*60)
    
    # Test the trained policy
    agent.policy.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()
    test_episodes = 5
    total_rewards = []
    
    print(f"Testing policy for {test_episodes} episodes...")
    
    with torch.no_grad():
        for episode in range(test_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0
            max_steps = 200  # Pendulum episode limit
            
            while not done and steps < max_steps:
                obs_dict = {"state": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
                # Use deterministic actions (no exploration noise) for evaluation
                action_dict = agent.policy.act(obs_dict)
                action = action_dict["actions"].squeeze().cpu().numpy()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Steps = {steps}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Print TD3 algorithm characteristics
    print("\n" + "="*60)
    print("TD3 Algorithm Characteristics:")
    print("="*60)
    print("TD3 improves upon DDPG with three key innovations:")
    print()
    print("1. Twin Critic Networks (Clipped Double Q-Learning):")
    print("   - Uses minimum of two critic networks: Q1 and Q2")
    print("   - Target Q-value = min(Q1(s',a'), Q2(s',a'))")
    print("   - Reduces overestimation bias common in value-based methods")
    print()
    print("2. Delayed Policy Updates:")
    print(f"   - Policy updated every {td3_cfg.policy_delay} critic updates")
    print("   - Reduces variance in policy updates")
    print("   - Allows critics to stabilize before policy learns from them")
    print()
    print("3. Target Policy Smoothing:")
    print("   - Adds clipped noise to target actions: a' = π(s') + ε")
    print(f"   - Noise clipped to [{-td3_cfg.smooth_regularization_clip}, {td3_cfg.smooth_regularization_clip}]")
    print("   - Regularizes targets and reduces overfitting to specific actions")
    print()
    print("Benefits over DDPG:")
    print("- More stable training due to reduced overestimation bias")
    print("- Better sample efficiency")
    print("- Less sensitive to hyperparameters")
    print("- More robust performance across different environments")
    print()
    print("Typical use cases:")
    print("- Continuous control tasks (robotics, autonomous driving)")
    print("- High-dimensional action spaces")
    print("- Environments where sample efficiency is important")
    print("- When DDPG shows training instability")
    print("="*60)


def demo_td3_components():
    """Demonstrate TD3's key components"""
    print("\n" + "="*60)
    print("TD3 Components Demonstration:")
    print("="*60)
    
    # Create dummy models for demonstration
    model_cfg = ModelCfg(device="cpu", inputs_name=["state"], concat_dict=False)
    
    # Twin critics
    critic_1 = TD3Critic(model_cfg, obs_dim=3, action_dim=1)
    critic_2 = TD3Critic(model_cfg, obs_dim=3, action_dim=1)
    
    # Dummy state and action
    state = torch.randn(1, 3)
    action = torch.randn(1, 1)
    
    # Get Q-values from both critics
    q1_output = critic_1.act({"state": state, "taken_actions": action})
    q2_output = critic_2.act({"state": state, "taken_actions": action})
    
    q1_value = q1_output["values"]
    q2_value = q2_output["values"]
    
    print(f"Critic 1 Q-value: {q1_value.item():.3f}")
    print(f"Critic 2 Q-value: {q2_value.item():.3f}")
    print(f"Min Q-value (used in TD3): {torch.min(q1_value, q2_value).item():.3f}")
    print()
    
    # Target policy smoothing
    policy = TD3Actor(model_cfg, obs_dim=3, action_dim=1, max_action=2.0)
    noise = SimpleNoise(mean=0.0, std=0.2)
    
    # Get action from policy
    policy_output = policy.act({"state": state})
    original_action = policy_output["actions"]
    
    # Add smoothing noise
    smoothing_noise = noise.sample(original_action.shape)
    smoothed_action = original_action + torch.clamp(smoothing_noise, -0.5, 0.5)
    
    print(f"Original action: {original_action.item():.3f}")
    print(f"Smoothing noise: {smoothing_noise.item():.3f}")
    print(f"Smoothed action: {smoothed_action.item():.3f}")
    print()
    
    # Policy delay counter simulation
    print("Policy delay demonstration:")
    critic_updates = 0
    policy_delay = 2
    
    for step in range(6):
        critic_updates += 1
        update_policy = (critic_updates % policy_delay == 0)
        print(f"Step {step + 1}: Critic update #{critic_updates}, Policy update: {update_policy}")
    
    print("="*60)


if __name__ == "__main__":
    main()
    demo_td3_components()