#!/usr/bin/env python3
"""
Q-Learning Example for AquaML

This example demonstrates how to use the Q-Learning algorithm with AquaML
on a discrete environment (CartPole-v1).

Q-Learning is a model-free, off-policy reinforcement learning algorithm that 
learns the quality of actions, telling an agent what action to take under 
what circumstances.
"""

import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.off_policy.q_learning import QLearning, QLearningCfg
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig


# Define Q-function network
class CartPoleQFunction(Model):
    """Q-function network for CartPole environment"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        # CartPole has 4 state dimensions and 2 discrete actions
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 actions: left or right
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute Q-values for all actions"""
        states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        q_values = self.net(states)
        return {"q_values": q_values}
        
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Select action based on Q-values (epsilon-greedy or greedy)"""
        outputs = self.compute(data_dict)
        q_values = outputs["q_values"]
        
        # Greedy action selection (argmax)
        actions = torch.argmax(q_values, dim=-1, keepdim=True)
        outputs["actions"] = actions
        outputs["max_q_value"] = torch.max(q_values, dim=-1, keepdim=True)[0]
        
        return outputs
    
    def random_act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Random action selection for exploration"""
        batch_size = list(data_dict.values())[0].shape[0]
        random_actions = torch.randint(0, 2, (batch_size, 1), device=self.device)
        return {"actions": random_actions}


def main():
    """Main training function"""
    print("🚀 Starting Q-Learning training on CartPole-v1")
    
    # 1. Create environment
    env = GymnasiumWrapper("CartPole-v1")
    print("✓ Environment created")
    
    # 2. Create model configuration
    model_cfg = ModelCfg(
        device="cpu",  # Use "cuda" if you have GPU
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 3. Create Q-function model
    q_function = CartPoleQFunction(model_cfg)
    print("✓ Q-function model created")
    
    # 4. Configure Q-Learning parameters
    q_cfg = QLearningCfg()
    
    # Basic training parameters
    q_cfg.device = "cpu"
    q_cfg.memory_size = 10000  # Experience buffer size
    
    # Q-Learning specific parameters
    q_cfg.learning_rate = 0.001  # Learning rate for neural network
    q_cfg.discount_factor = 0.99  # Discount factor (gamma)
    
    # Exploration parameters
    q_cfg.random_timesteps = 1000  # Random exploration steps
    q_cfg.learning_starts = 1000  # Start learning after this many steps
    
    # Function approximation settings
    q_cfg.batch_size = 1  # Q-Learning typically uses single-step updates
    
    print("✓ Q-Learning configuration set")
    
    # 5. Create Q-Learning agent
    models = {"policy": q_function}
    agent = QLearning(models, q_cfg)
    print("✓ Q-Learning agent created")
    
    # 6. Create trainer configuration
    trainer_cfg = TrainerConfig(
        timesteps=20000,  # Total training timesteps
        headless=True,    # No rendering
        disable_progressbar=False  # Show progress
    )
    
    # 7. Create trainer and start training
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    print("🎯 Starting training...")
    
    trainer.train()
    
    print("✅ Training completed!")
    
    # 8. Save the trained model
    model_path = "./q_learning_cartpole_model.pt"
    agent.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    # 9. Test the trained model
    print("🧪 Testing trained model...")
    
    # Reset environment for testing
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    # Run a test episode
    while steps < 500:  # CartPole max episode length
        # Convert state to the expected format
        state_dict = {"state": torch.tensor(state, dtype=torch.float32)}
        
        # Get action from trained Q-function (greedy policy)
        with torch.no_grad():
            action_dict = agent.act(state_dict, 10000, 10000)  # Use large timestep to avoid random actions
            action = action_dict["actions"].item()
        
        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"🏆 Test episode: {steps} steps, total reward: {total_reward:.2f}")
    
    # Run multiple test episodes to get average performance
    print("📊 Running multiple test episodes...")
    test_episodes = 10
    total_rewards = []
    total_steps = []
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < 500:
            state_dict = {"state": torch.tensor(state, dtype=torch.float32)}
            
            with torch.no_grad():
                action_dict = agent.act(state_dict, 10000, 10000)
                action = action_dict["actions"].item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        print(f"  Episode {episode + 1}: {episode_steps} steps, reward: {episode_reward:.2f}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_steps = sum(total_steps) / len(total_steps)
    
    print(f"\n📈 Test Results (10 episodes):")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average steps: {avg_steps:.2f}")
    print(f"   Best episode: {max(total_rewards):.2f} reward, {max(total_steps)} steps")
    
    print("\n🎉 Q-Learning example completed successfully!")
    print(f"📊 Training details:")
    print(f"   - Algorithm: Q-Learning with function approximation")
    print(f"   - Environment: CartPole-v1")
    print(f"   - Total timesteps: {trainer_cfg.timesteps}")
    print(f"   - Learning rate: {q_cfg.learning_rate}")
    print(f"   - Discount factor: {q_cfg.discount_factor}")
    print(f"   - Random exploration steps: {q_cfg.random_timesteps}")
    
    # Show Q-function predictions for sample states
    print("\n🔍 Sample Q-function predictions:")
    sample_states = [
        [0.0, 0.0, 0.0, 0.0],  # Balanced state
        [0.1, 0.0, 0.1, 0.0],  # Slightly tilted right
        [-0.1, 0.0, -0.1, 0.0],  # Slightly tilted left
    ]
    
    for i, state in enumerate(sample_states):
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            q_outputs = q_function.compute({"state": state_tensor})
            q_values = q_outputs["q_values"][0]
            preferred_action = torch.argmax(q_values).item()
            
        print(f"   State {i+1}: {state}")
        print(f"   Q-values: [Left={q_values[0]:.3f}, Right={q_values[1]:.3f}]")
        print(f"   Preferred action: {'Left' if preferred_action == 0 else 'Right'}")


if __name__ == "__main__":
    main()