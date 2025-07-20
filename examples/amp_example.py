#!/usr/bin/env python3
"""
AMP (Adversarial Motion Priors) training example for humanoid environments.

This example demonstrates how to use the AMP algorithm from AquaML
to train an agent with natural motion priors using reference motion data.

AMP combines task rewards with style rewards from a discriminator that
learns to distinguish between agent motions and reference motions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Callable

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.amp import AMP, AMPCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


# Define policy network
class HumanoidPolicy(GaussianModel):
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 376, action_dim: int = 17):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        return {"mean_actions": mean, "log_std": log_std}


# Define value network
class HumanoidValue(Model):
    def __init__(self, model_cfg: ModelCfg, obs_dim: int = 376):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        values = self.net(states)
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


# Define discriminator network
class HumanoidDiscriminator(Model):
    def __init__(self, model_cfg: ModelCfg, amp_obs_dim: int = 108):
        super().__init__(model_cfg)
        self.net = nn.Sequential(
            nn.Linear(amp_obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        logits = self.net(states)
        return {"values": logits}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


# Mock reference motion data (replace with actual motion capture data)
def create_mock_reference_motions(batch_size: int, amp_obs_dim: int = 108) -> torch.Tensor:
    """Create mock reference motion data for demonstration"""
    # In practice, this would load actual motion capture data
    # The data should represent natural human motions
    return torch.randn(batch_size, amp_obs_dim)


# Mock AMP observation extraction (replace with actual environment-specific implementation)
def extract_amp_observations(env_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract AMP-specific observations from environment observations"""
    # In practice, this would extract relevant kinematic features like:
    # - Joint positions and velocities
    # - Root orientation and velocity
    # - Body part positions relative to root
    
    # For demonstration, we'll create mock AMP observations
    batch_size = env_obs["state"].shape[0] if env_obs["state"].dim() > 1 else 1
    amp_obs_dim = 108  # Typical for humanoid AMP observations
    return torch.randn(batch_size, amp_obs_dim)


# Mock environment wrapper that provides AMP observations
class AMPEnvironmentWrapper:
    def __init__(self, base_env):
        self.base_env = base_env
        
    def reset(self):
        state, info = self.base_env.reset()
        # Add AMP observations to info
        info["amp_obs"] = extract_amp_observations({"state": torch.tensor(state, dtype=torch.float32)})
        return state, info
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.base_env.step(action)
        # Add AMP observations to info
        info["amp_obs"] = extract_amp_observations({"state": torch.tensor(next_state, dtype=torch.float32)})
        return next_state, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.base_env, name)


def main():
    """Main training function"""
    print("Starting AMP training for humanoid environment...")
    
    # Environment setup - for demo, we'll use a simple environment
    # In practice, use a humanoid environment like Humanoid-v4
    base_env = GymnasiumWrapper("Humanoid-v4")  # or "HumanoidStandup-v4"
    env = AMPEnvironmentWrapper(base_env)
    
    # Dimensions (adjust based on actual environment)
    obs_dim = 376  # Humanoid observation dimension
    action_dim = 17  # Humanoid action dimension  
    amp_obs_dim = 108  # AMP observation dimension
    
    # Create model configuration
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # Create models
    policy = HumanoidPolicy(model_cfg, obs_dim, action_dim)
    value = HumanoidValue(model_cfg, obs_dim)
    discriminator = HumanoidDiscriminator(model_cfg, amp_obs_dim)
    
    # Configure AMP parameters
    amp_cfg = AMPCfg()
    amp_cfg.device = "cpu"
    amp_cfg.memory_size = 4096
    amp_cfg.rollouts = 64  # Larger rollouts for AMP
    amp_cfg.learning_epochs = 6
    amp_cfg.mini_batches = 4
    amp_cfg.learning_rate = 5e-5  # Lower learning rate for stability
    amp_cfg.mixed_precision = False
    
    # AMP specific parameters
    amp_cfg.amp_batch_size = 512
    amp_cfg.task_reward_weight = 0.0  # Pure style learning
    amp_cfg.style_reward_weight = 1.0
    amp_cfg.discriminator_loss_scale = 5.0
    amp_cfg.discriminator_reward_scale = 2.0
    amp_cfg.discriminator_gradient_penalty_scale = 5.0
    amp_cfg.discriminator_logit_regularization_scale = 0.05
    amp_cfg.discriminator_weight_decay_scale = 0.0001
    
    # GAE parameters
    amp_cfg.discount_factor = 0.99
    amp_cfg.lambda_value = 0.95
    
    # Clipping parameters
    amp_cfg.ratio_clip = 0.2
    amp_cfg.value_clip = 0.2
    amp_cfg.grad_norm_clip = 0.5
    
    # Loss scaling
    amp_cfg.value_loss_scale = 2.5
    amp_cfg.entropy_loss_scale = 0.01
    
    print(f"AMP Configuration:")
    print(f"  - Rollouts: {amp_cfg.rollouts}")
    print(f"  - Learning rate: {amp_cfg.learning_rate}")
    print(f"  - AMP batch size: {amp_cfg.amp_batch_size}")
    print(f"  - Task reward weight: {amp_cfg.task_reward_weight}")
    print(f"  - Style reward weight: {amp_cfg.style_reward_weight}")
    print(f"  - Discriminator loss scale: {amp_cfg.discriminator_loss_scale}")
    
    # Create reference motion collection function
    def collect_reference_motions(batch_size: int) -> torch.Tensor:
        """Collect reference motion data"""
        return create_mock_reference_motions(batch_size, amp_obs_dim)
    
    # Create observation collection function (optional)
    def collect_observation() -> Dict[str, torch.Tensor]:
        """Collect current observation (optional for some environments)"""
        # This can be used for environments that require special observation collection
        return None
    
    # Create AMP agent
    models = {"policy": policy, "value": value, "discriminator": discriminator}
    agent = AMP(
        models=models,
        cfg=amp_cfg,
        amp_observation_space={"amp_obs": amp_obs_dim},
        collect_reference_motions=collect_reference_motions,
        collect_observation=collect_observation
    )
    
    # Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=100000,  # Longer training for AMP
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
        print("This is expected with mock environment - replace with actual humanoid environment")
    
    # Save model
    model_path = "./trained_amp_model.pt"
    agent.save(model_path)
    print(f"Training completed! Model saved to {model_path}")
    
    # Print training tips
    print("\n" + "="*60)
    print("AMP Training Tips:")
    print("="*60)
    print("1. Use a proper humanoid environment (e.g., IsaacGym)")
    print("2. Provide high-quality reference motion data (mocap)")
    print("3. Tune discriminator learning vs policy learning balance")
    print("4. Start with style_reward_weight=1.0, task_reward_weight=0.0")
    print("5. Gradually increase task_reward_weight if task performance needed")
    print("6. Monitor discriminator accuracy - should be around 50-70%")
    print("7. Use appropriate AMP observation features (joint angles, velocities)")
    print("="*60)


if __name__ == "__main__":
    main()