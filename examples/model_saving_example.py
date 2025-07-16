#!/usr/bin/env python3
"""
Example demonstrating the enhanced model saving mechanism in AquaML

This example shows how to:
1. Create and train a PPO agent
2. Save model checkpoints during training
3. Load saved models
4. Use the best model checkpoint feature
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.ppo import PPO, PPOCfg
from AquaML.learning.model.gaussian import GaussianModel


class SimplePolicy(GaussianModel):
    """Simple policy model for demonstration"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # Simple MLP network
        self.net = nn.Sequential(
            nn.Linear(4, 32),  # Assuming 4D observation space
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # Assuming 2D action space
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        states = data_dict["states"]
        mean = self.net(states)
        return {"mean": mean}


class SimpleValue(Model):
    """Simple value model for demonstration"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # Simple MLP network
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        states = data_dict["states"]
        values = self.net(states)
        return {"values": values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Value function doesn't need to act"""
        return self.compute(data_dict)


def demo_model_saving():
    """Demonstrate the model saving functionality"""
    print("=== AquaML Model Saving Demo ===\n")
    
    # 1. Create model configurations
    policy_cfg = ModelCfg(
        device="cpu",
        inputs_name=["states"],
        concat_dict=False
    )
    
    value_cfg = ModelCfg(
        device="cpu", 
        inputs_name=["states"],
        concat_dict=False
    )
    
    # 2. Create models
    policy = SimplePolicy(policy_cfg)
    value = SimpleValue(value_cfg)
    
    print("✓ Created policy and value models")
    
    # 3. Create PPO configuration with checkpoint settings
    ppo_cfg = PPOCfg()
    ppo_cfg.device = "cpu"
    ppo_cfg.memory_size = 1000
    
    # Configure checkpointing
    ppo_cfg.experiment = {
        "directory": "./runs",
        "experiment_name": "ppo_saving_demo",
        "checkpoint_interval": 100,  # Save every 100 timesteps
        "store_separately": False    # Save as single file
    }
    
    # 4. Create PPO agent
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # Initialize the agent
    agent.init({"timesteps": 1000})
    print("✓ Created and initialized PPO agent")
    
    # 5. Demonstrate manual saving
    print("\n--- Manual Saving Demo ---")
    
    # Save individual model
    policy.save("./runs/policy_manual.pt")
    print("✓ Saved policy model manually")
    
    # Save entire agent
    agent.save("./runs/agent_manual.pt")
    print("✓ Saved agent manually")
    
    # 6. Demonstrate checkpoint system
    print("\n--- Checkpoint System Demo ---")
    
    # Simulate training with periodic checkpoints
    for timestep in range(1, 301):
        # Simulate some training data
        dummy_reward = np.random.random() * 10
        
        # Update best checkpoint if this is a good reward
        agent.update_best_checkpoint(timestep, dummy_reward)
        
        # Write checkpoint at intervals
        if timestep % 100 == 0:
            agent.write_checkpoint(timestep, 300)
            print(f"✓ Checkpoint saved at timestep {timestep}")
    
    # 7. Demonstrate loading
    print("\n--- Loading Demo ---")
    
    # Create new models for loading
    new_policy = SimplePolicy(policy_cfg)
    new_value = SimpleValue(value_cfg)
    
    # Load individual model
    new_policy.load("./runs/policy_manual.pt") 
    print("✓ Loaded policy model")
    
    # Load entire agent
    new_models = {"policy": new_policy, "value": new_value}
    new_agent = PPO(new_models, ppo_cfg)
    new_agent.load("./runs/agent_manual.pt")
    print("✓ Loaded agent")
    
    # 8. Show checkpoint files created
    import os
    checkpoint_dir = "./runs/ppo_saving_demo/checkpoints"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"\n--- Created Checkpoint Files ---")
        for file in sorted(files):
            print(f"  • {file}")
    
    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("• Manual model/agent saving and loading")
    print("• Automatic checkpoint system with configurable intervals")
    print("• Best model checkpoint based on performance")
    print("• Error handling and logging")
    print("• Directory management")


if __name__ == "__main__":
    demo_model_saving()