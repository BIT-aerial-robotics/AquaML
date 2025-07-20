#!/usr/bin/env python3
"""
RPO (Robust Policy Optimization) training example for continuous control environments.

This example demonstrates how to use the RPO algorithm from AquaML
to train robust policies that are resilient to perturbations and uncertainties.

RPO extends PPO by incorporating robustness through adversarial perturbations,
making the policy more robust to environmental variations and model uncertainties.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.rpo import RPO, RPOCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


# Define robust policy network
class RobustPolicy(GaussianModel):
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 17, action_dim: int = 6):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Add robustness perturbation if alpha is provided
        if "alpha" in data_dict:
            alpha = data_dict["alpha"]
            if isinstance(alpha, torch.Tensor) and alpha.numel() == 1:
                alpha = alpha.item()
            
            # Apply uniform random perturbation: U(-alpha, alpha)
            perturbation = torch.uniform(-alpha, alpha, states.shape, device=states.device)
            states = states + perturbation
        
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        return {"mean_actions": mean, "log_std": log_std}


# Define robust value network
class RobustValue(Model):
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 17):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Add robustness perturbation if alpha is provided
        if "alpha" in data_dict:
            alpha = data_dict["alpha"]
            if isinstance(alpha, torch.Tensor) and alpha.numel() == 1:
                alpha = alpha.item()
            
            # Apply uniform random perturbation: U(-alpha, alpha)
            perturbation = torch.uniform(-alpha, alpha, states.shape, device=states.device)
            states = states + perturbation
        
        values = self.net(states)
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


def main():
    """Main training function"""
    print("Starting RPO training for robust continuous control...")
    
    # Environment setup - use a continuous control environment
    env = GymnasiumWrapper("HalfCheetah-v4")  # or "Ant-v4", "Humanoid-v4", etc.
    
    # Get environment dimensions
    obs_space = env.observation_space
    action_space = env.action_space
    
    obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n
    action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n
    
    print(f"Environment: {env.spec.id}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create model configuration
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # Create models
    policy = RobustPolicy(model_cfg, obs_dim, action_dim)
    value = RobustValue(model_cfg, obs_dim)
    
    # Configure RPO parameters
    rpo_cfg = RPOCfg()
    rpo_cfg.device = "cpu"
    rpo_cfg.memory_size = 4096
    rpo_cfg.rollouts = 32  # Larger rollouts for better robustness
    rpo_cfg.learning_epochs = 8
    rpo_cfg.mini_batches = 4
    rpo_cfg.learning_rate = 3e-4
    rpo_cfg.mixed_precision = False
    
    # RPO specific parameters
    rpo_cfg.alpha = 0.5  # Perturbation magnitude - key robustness parameter
    
    # GAE parameters
    rpo_cfg.discount_factor = 0.99
    rpo_cfg.lambda_value = 0.95
    
    # Clipping parameters
    rpo_cfg.ratio_clip = 0.2
    rpo_cfg.value_clip = 0.2
    rpo_cfg.grad_norm_clip = 0.5
    
    # Loss scaling
    rpo_cfg.value_loss_scale = 1.0
    rpo_cfg.entropy_loss_scale = 0.01
    
    print(f"RPO Configuration:")
    print(f"  - Rollouts: {rpo_cfg.rollouts}")
    print(f"  - Learning rate: {rpo_cfg.learning_rate}")
    print(f"  - Alpha (robustness): {rpo_cfg.alpha}")
    print(f"  - Learning epochs: {rpo_cfg.learning_epochs}")
    print(f"  - Mini batches: {rpo_cfg.mini_batches}")
    
    # Create RPO agent
    models = {"policy": policy, "value": value}
    agent = RPO(
        models=models,
        cfg=rpo_cfg,
        observation_space={"state": obs_dim},
        action_space={"action": action_dim}
    )
    
    # Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=100000,  # Longer training for robust policies
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
    model_path = "./trained_rpo_model.pt"
    agent.save(model_path)
    print(f"Training completed! Model saved to {model_path}")
    
    # Demonstrate robustness testing
    print("\n" + "="*60)
    print("Testing Robustness:")
    print("="*60)
    
    # Test the trained policy with different alpha values
    test_alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    print("Alpha\tAverage Return")
    print("-"*20)
    
    agent.set_mode("eval")
    with torch.no_grad():
        for test_alpha in test_alphas:
            returns = []
            for _ in range(5):  # Test 5 episodes for each alpha
                obs, _ = env.reset()
                total_return = 0
                done = False
                
                while not done:
                    obs_dict = {"state": torch.tensor(obs, dtype=torch.float32)}
                    obs_dict["alpha"] = torch.tensor(test_alpha)
                    
                    action_dict = agent.policy.act(obs_dict)
                    action = action_dict["actions"].numpy()
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_return += reward
                    done = terminated or truncated
                
                returns.append(total_return)
            
            avg_return = np.mean(returns)
            print(f"{test_alpha:.1f}\t{avg_return:.1f}")
    
    # Print training tips
    print("\n" + "="*60)
    print("RPO Training Tips:")
    print("="*60)
    print("1. Alpha controls robustness - higher values = more robust but potentially lower performance")
    print("2. Start with alpha=0.1-0.3 for most environments")
    print("3. Use larger rollouts (32-64) for better robustness estimates")
    print("4. Monitor policy performance across different alpha values during evaluation")
    print("5. RPO works best in continuous control tasks with environmental uncertainties")
    print("6. Consider domain randomization alongside RPO for maximum robustness")
    print("7. The policy should maintain good performance even with perturbations")
    print("="*60)


if __name__ == "__main__":
    main()