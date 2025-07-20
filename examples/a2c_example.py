#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) training example for Pendulum environment.

This example demonstrates how to use the A2C algorithm from AquaML
to train an agent on the Pendulum-v1 environment.
"""

import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.a2c import A2C, A2CCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


# Define policy network
class PendulumPolicy(GaussianModel):
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
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


# Define value network
class PendulumValue(Model):
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
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
    """Main training function"""
    print("Starting A2C training on Pendulum-v1...")
    
    # 1. Create environment
    env = GymnasiumWrapper("Pendulum-v1")
    
    # 2. Create model configuration
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 3. Create models
    policy = PendulumPolicy(model_cfg)
    value = PendulumValue(model_cfg)
    
    # 4. Configure A2C parameters
    a2c_cfg = A2CCfg()
    a2c_cfg.device = "cpu"
    a2c_cfg.memory_size = 200
    a2c_cfg.rollouts = 32  # Number of steps before updating
    a2c_cfg.mini_batches = 1  # A2C typically uses only 1 mini-batch
    a2c_cfg.learning_rate = 3e-4
    a2c_cfg.mixed_precision = False
    a2c_cfg.entropy_loss_scale = 0.01  # Small entropy bonus for exploration
    a2c_cfg.grad_norm_clip = 0.5
    
    # GAE parameters
    a2c_cfg.discount_factor = 0.99
    a2c_cfg.lambda_value = 0.95
    
    print(f"A2C Configuration:")
    print(f"  - Rollouts: {a2c_cfg.rollouts}")
    print(f"  - Learning rate: {a2c_cfg.learning_rate}")
    print(f"  - Entropy scale: {a2c_cfg.entropy_loss_scale}")
    print(f"  - Discount factor: {a2c_cfg.discount_factor}")
    print(f"  - Lambda (GAE): {a2c_cfg.lambda_value}")
    
    # 5. Create A2C agent
    models = {"policy": policy, "value": value}
    agent = A2C(models, a2c_cfg)
    
    # 6. Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=2000,  # Shorter training for demo
        headless=True,
        disable_progressbar=False
    )
    
    print(f"Training for {trainer_cfg.timesteps} timesteps...")
    
    # 7. Create trainer and start training
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    trainer.train()
    
    # 8. Save model
    model_path = "./trained_a2c_model.pt"
    agent.save(model_path)
    print(f"Training completed! Model saved to {model_path}")
    
    # 9. Optional: Test the trained model
    print("\nTesting trained model...")
    test_episodes = 5
    total_rewards = []
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 200:  # Pendulum max steps
            # Convert state to proper format
            state_dict = {"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)}
            
            # Get action from trained policy
            with torch.no_grad():
                outputs = agent.act(state_dict, 0, 0)
                action = outputs["actions"].cpu().numpy().flatten()
            
            # Take environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average test reward over {test_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    main()