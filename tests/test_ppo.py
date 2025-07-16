"""Test PPO implementation for AquaML"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from loguru import logger
import sys
import os

# Add AquaML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AquaML.learning.reinforcement.on_policy.ppo import PPO, PPOCfg
from AquaML.learning.model.base import Model
from AquaML.learning.model.gaussian import GaussianMixin
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML import coordinator


class PolicyNetwork(GaussianMixin, Model):
    """Policy network for PPO testing"""
    
    def __init__(self, model_cfg: ModelCfg, 
                 observation_dim: int = 4, 
                 action_dim: int = 2,
                 hidden_dim: int = 64,
                 clip_actions: bool = False,
                 clip_log_std: bool = True,
                 min_log_std: float = -20,
                 max_log_std: float = 2,
                 reduction: str = "sum"):
        Model.__init__(self, model_cfg)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Build network
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Log standard deviation parameter
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
        
    def compute(self, data_dict):
        """Compute mean actions and log std from states"""
        # Try to get states from different possible keys
        states = None
        possible_keys = ["states", "observation", "obs"]
        
        for key in possible_keys:
            if key in data_dict:
                states = data_dict[key]
                break
        
        if states is None:
            # If no standard keys, take the first available value
            states = list(data_dict.values())[0]
        
        # If states is a dictionary, extract the observation
        if isinstance(states, dict):
            if "observation" in states:
                states = states["observation"]
            elif "obs" in states:
                states = states["obs"]
            else:
                # Take the first available key
                states = list(states.values())[0]
        
        # Ensure states is properly shaped
        if states.dim() == 1:
            states = states.unsqueeze(0)
            
        mean_actions = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean_actions)
        
        return {"mean_actions": mean_actions, "log_std": log_std}


class ValueNetwork(Model):
    """Value network for PPO testing"""
    
    def __init__(self, model_cfg: ModelCfg, 
                 observation_dim: int = 4, 
                 hidden_dim: int = 64):
        super().__init__(model_cfg)
        
        self.observation_dim = observation_dim
        
        # Build network
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def compute(self, data_dict):
        """Compute value from states"""
        # Try to get states from different possible keys
        states = None
        possible_keys = ["states", "observation", "obs"]
        
        for key in possible_keys:
            if key in data_dict:
                states = data_dict[key]
                break
        
        if states is None:
            # If no standard keys, take the first available value
            states = list(data_dict.values())[0]
        
        # If states is a dictionary, extract the observation
        if isinstance(states, dict):
            if "observation" in states:
                states = states["observation"]
            elif "obs" in states:
                states = states["obs"]
            else:
                # Take the first available key
                states = list(states.values())[0]
        
        # Ensure states is properly shaped
        if states.dim() == 1:
            states = states.unsqueeze(0)
            
        values = self.net(states)
        return {"values": values}
        
    def act(self, data_dict):
        """Act method for value network"""
        outputs = self.compute(data_dict)
        return outputs


class MockEnvironment:
    """Mock environment for testing PPO"""
    
    def __init__(self, observation_dim: int = 4, action_dim: int = 2):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        obs = torch.randn(1, self.observation_dim)
        return {"observation": obs}
        
    def step(self, actions):
        """Step environment"""
        self.step_count += 1
        
        # Generate next observation
        next_obs = torch.randn(1, self.observation_dim)
        
        # Generate reward (simple reward based on actions)
        reward = torch.randn(1, 1)
        
        # Check if episode is done
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return (
            {"observation": next_obs},
            reward,
            torch.tensor([terminated], dtype=torch.bool),
            torch.tensor([truncated], dtype=torch.bool),
            {}
        )


def test_ppo_initialization():
    """Test PPO agent initialization"""
    logger.info("Testing PPO initialization...")
    
    # Initialize coordinator
    coordinator.initialize()
    
    # Create model configuration
    model_cfg = ModelCfg(device="cpu", inputs_name=["states"])
    
    # Create policy and value networks
    policy = PolicyNetwork(model_cfg)
    value = ValueNetwork(model_cfg)
    
    # Create PPO configuration
    ppo_cfg = PPOCfg(
        rollouts=4,
        learning_epochs=2,
        mini_batches=1,
        learning_rate=3e-4,
        device="cpu",
        memory_size=1000
    )
    
    # Create PPO agent
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    assert agent.policy is not None
    assert agent.value is not None
    assert agent.device == "cpu"
    
    logger.info("PPO initialization test passed!")


def test_ppo_act():
    """Test PPO action generation"""
    logger.info("Testing PPO action generation...")
    
    # Initialize coordinator
    coordinator.initialize()
    
    # Create model configuration
    model_cfg = ModelCfg(device="cpu", inputs_name=["states"])
    
    # Create policy and value networks
    policy = PolicyNetwork(model_cfg)
    value = ValueNetwork(model_cfg)
    
    # Create PPO configuration
    ppo_cfg = PPOCfg(
        rollouts=4,
        learning_epochs=2,
        mini_batches=1,
        learning_rate=3e-4,
        device="cpu",
        memory_size=1000
    )
    
    # Create PPO agent
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # Test action generation
    states = {"observation": torch.randn(1, 4)}
    outputs = agent.act(states, timestep=0, timesteps=1000)
    
    assert "actions" in outputs
    assert "log_prob" in outputs
    assert "mean_actions" in outputs
    assert outputs["actions"].shape == (1, 2)
    
    logger.info("PPO action generation test passed!")


def test_ppo_record_transition():
    """Test PPO transition recording"""
    logger.info("Testing PPO transition recording...")
    
    # Initialize coordinator
    coordinator.initialize()
    
    # Create model configuration
    model_cfg = ModelCfg(device="cpu", inputs_name=["states"])
    
    # Create policy and value networks
    policy = PolicyNetwork(model_cfg)
    value = ValueNetwork(model_cfg)
    
    # Create PPO configuration
    ppo_cfg = PPOCfg(
        rollouts=4,
        learning_epochs=2,
        mini_batches=1,
        learning_rate=3e-4,
        device="cpu",
        memory_size=1000
    )
    
    # Create PPO agent
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # Test transition recording
    states = {"observation": torch.randn(1, 4)}
    actions = {"actions": torch.randn(1, 2)}
    rewards = torch.randn(1, 1)
    next_states = {"observation": torch.randn(1, 4)}
    terminated = torch.tensor([False], dtype=torch.bool)
    truncated = torch.tensor([False], dtype=torch.bool)
    
    # First generate actions to set up log_prob
    agent.act(states, timestep=0, timesteps=1000)
    
    # Record transition
    agent.record_transition(
        states, actions, rewards, next_states, 
        terminated, truncated, {}, 0, 1000
    )
    
    # Check that memory has data
    assert agent.memory.position > 0
    
    logger.info("PPO transition recording test passed!")


def test_ppo_training_loop():
    """Test PPO training loop"""
    logger.info("Testing PPO training loop...")
    
    # Initialize coordinator
    coordinator.initialize()
    
    # Create model configuration
    model_cfg = ModelCfg(device="cpu", inputs_name=["states"])
    
    # Create policy and value networks
    policy = PolicyNetwork(model_cfg)
    value = ValueNetwork(model_cfg)
    
    # Create PPO configuration
    ppo_cfg = PPOCfg(
        rollouts=4,  # Small rollouts for faster testing
        learning_epochs=2,
        mini_batches=1,
        learning_rate=3e-4,
        device="cpu",
        memory_size=1000,
        entropy_loss_scale=0.01
    )
    
    # Create PPO agent
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # Create mock environment
    env = MockEnvironment()
    
    # Run training loop
    timesteps = 20
    states = env.reset()
    
    for timestep in range(timesteps):
        # Pre-interaction
        agent.pre_interaction(timestep, timesteps)
        
        # Generate actions
        outputs = agent.act(states, timestep, timesteps)
        actions = {"actions": outputs["actions"]}
        
        # Step environment
        next_states, rewards, terminated, truncated, infos = env.step(actions)
        
        # Record transition
        agent.record_transition(
            states, actions, rewards, next_states,
            terminated, truncated, infos, timestep, timesteps
        )
        
        # Post-interaction (may trigger learning)
        agent.post_interaction(timestep, timesteps)
        
        # Update states
        states = next_states
        
        # Reset environment if episode ended
        if terminated.item() or truncated.item():
            states = env.reset()
    
    logger.info("PPO training loop test passed!")


def test_ppo_save_load():
    """Test PPO save and load functionality"""
    logger.info("Testing PPO save/load...")
    
    # Initialize coordinator
    coordinator.initialize()
    
    # Create model configuration
    model_cfg = ModelCfg(device="cpu", inputs_name=["states"])
    
    # Create policy and value networks
    policy = PolicyNetwork(model_cfg)
    value = ValueNetwork(model_cfg)
    
    # Create PPO configuration
    ppo_cfg = PPOCfg(
        rollouts=4,
        learning_epochs=2,
        mini_batches=1,
        learning_rate=3e-4,
        device="cpu",
        memory_size=1000
    )
    
    # Create PPO agent
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # Save model
    save_path = "/tmp/test_ppo_model.pt"
    agent.save(save_path)
    
    # Create new agent and load
    policy2 = PolicyNetwork(model_cfg)
    value2 = ValueNetwork(model_cfg)
    models2 = {"policy": policy2, "value": value2}
    agent2 = PPO(models2, ppo_cfg)
    agent2.load(save_path)
    
    # Test that models produce same outputs
    states = {"observation": torch.randn(1, 4)}
    
    # Set both agents to eval mode
    agent.policy.eval()
    agent.value.eval()
    agent2.policy.eval()
    agent2.value.eval()
    
    with torch.no_grad():
        outputs1 = agent.policy.act(states)
        outputs2 = agent2.policy.act(states)
        
        # Check that actions are close (not exactly equal due to stochasticity)
        assert torch.allclose(outputs1["mean_actions"], outputs2["mean_actions"], atol=1e-6)
    
    # Clean up
    os.remove(save_path)
    
    logger.info("PPO save/load test passed!")


def run_all_tests():
    """Run all PPO tests"""
    logger.info("Starting PPO tests...")
    
    try:
        test_ppo_initialization()
        test_ppo_act()
        test_ppo_record_transition()
        test_ppo_training_loop()
        test_ppo_save_load()
        
        logger.info("All PPO tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"PPO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run tests
    success = run_all_tests()
    
    if success:
        logger.info("🎉 All PPO tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some PPO tests failed!")
        sys.exit(1)