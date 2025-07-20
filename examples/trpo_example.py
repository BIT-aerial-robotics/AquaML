#!/usr/bin/env python3
"""
TRPO Example for AquaML

This example demonstrates how to use the TRPO (Trust Region Policy Optimization) 
algorithm with AquaML on the Pendulum environment.

TRPO is an on-policy algorithm that improves upon policy gradient methods by 
constraining policy updates to a trust region, ensuring more stable training.
"""

import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.trpo import TRPO, TRPOCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


# Define policy network
class PendulumPolicy(GaussianModel):
    """Policy network for Pendulum environment using Gaussian distribution"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        # Simple 2-layer network for Pendulum (3 states -> 1 action)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Learnable log standard deviation
        self.log_std_parameter = nn.Parameter(torch.zeros(1))
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute mean actions and log standard deviation"""
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        return {"mean_actions": mean, "log_std": log_std}


# Define value network
class PendulumValue(Model):
    """Value network for Pendulum environment"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        # Simple 2-layer network for state value estimation
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute state values"""
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        values = self.net(states)
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Act method for compatibility"""
        return self.compute(data_dict)


def main():
    """Main training function"""
    print("🚀 Starting TRPO training on Pendulum-v1")
    
    # 1. Create environment
    env = GymnasiumWrapper("Pendulum-v1")
    print("✓ Environment created")
    
    # 2. Create model configuration
    model_cfg = ModelCfg(
        device="cpu",  # Use "cuda" if you have GPU
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 3. Create models
    policy = PendulumPolicy(model_cfg)
    value = PendulumValue(model_cfg)
    print("✓ Models created")
    
    # 4. Configure TRPO parameters
    trpo_cfg = TRPOCfg()
    
    # Basic training parameters
    trpo_cfg.device = "cpu"
    trpo_cfg.memory_size = 2048
    trpo_cfg.rollouts = 32  # Collect 32 rollouts before updating
    trpo_cfg.learning_epochs = 10  # Value function update epochs
    trpo_cfg.mini_batches = 4
    
    # TRPO-specific parameters
    trpo_cfg.max_kl_divergence = 0.01  # Trust region constraint
    trpo_cfg.conjugate_gradient_steps = 10  # CG algorithm iterations
    trpo_cfg.damping = 0.1  # Damping for numerical stability
    trpo_cfg.max_backtrack_steps = 10  # Line search steps
    trpo_cfg.accept_ratio = 0.5  # Minimum improvement ratio
    
    # Value function parameters
    trpo_cfg.value_learning_rate = 3e-4
    trpo_cfg.grad_norm_clip = 0.5
    trpo_cfg.value_loss_scale = 1.0
    
    # GAE parameters
    trpo_cfg.discount_factor = 0.99
    trpo_cfg.lambda_value = 0.95
    
    print("✓ TRPO configuration set")
    
    # 5. Create TRPO agent
    models = {"policy": policy, "value": value}
    agent = TRPO(models, trpo_cfg)
    print("✓ TRPO agent created")
    
    # 6. Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=50000,  # Total training timesteps
        headless=True,    # No rendering
        disable_progressbar=False  # Show progress
    )
    
    # 7. Create trainer and start training
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    print("🎯 Starting training...")
    
    trainer.train()
    
    print("✅ Training completed!")
    
    # 8. Save the trained model
    model_path = "./trpo_pendulum_model.pt"
    agent.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    # 9. Test the trained model
    print("🧪 Testing trained model...")
    
    # Reset environment for testing
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    # Run a test episode
    while steps < 200:  # Pendulum max episode length
        # Convert state to the expected format
        state_dict = {"state": torch.tensor(state, dtype=torch.float32)}
        
        # Get action from trained policy
        with torch.no_grad():
            action_dict = agent.act(state_dict, 0, 1000)
            action = action_dict["actions"].numpy()
        
        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"🏆 Test episode: {steps} steps, total reward: {total_reward:.2f}")
    
    print("\n🎉 TRPO example completed successfully!")
    print(f"📊 Training details:")
    print(f"   - Algorithm: TRPO (Trust Region Policy Optimization)")
    print(f"   - Environment: Pendulum-v1")
    print(f"   - Total timesteps: {trainer_cfg.timesteps}")
    print(f"   - Max KL divergence: {trpo_cfg.max_kl_divergence}")
    print(f"   - Rollouts per update: {trpo_cfg.rollouts}")


if __name__ == "__main__":
    main()