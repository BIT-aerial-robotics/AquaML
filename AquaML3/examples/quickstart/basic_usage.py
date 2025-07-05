#!/usr/bin/env python3
"""
AquaML - Basic Usage Example

This example demonstrates how to use the AquaML framework.
"""

import torch
from typing import Dict, Any

# Import from new architecture
from AquaML.core.coordinator import AquaMLCoordinator
from AquaML.plugins.interface import AlgorithmPlugin, PluginInfo, PluginType
from AquaML.learning.base.learner import BaseLearner


class SimpleRLAgent(BaseLearner):
    """Simple RL agent example"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = torch.nn.Linear(config.get('input_dim', 4), config.get('output_dim', 2))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        self.loss_fn = torch.nn.MSELoss()
    
    def train_step(self, batch: Any) -> Dict[str, Any]:
        """Training step"""
        states, actions, rewards = batch
        
        # Forward pass
        predictions = self.model(states)
        targets = rewards.unsqueeze(1)
        
        # Compute loss
        loss = self.loss_fn(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def predict(self, input_data: Any) -> Any:
        """Make predictions"""
        with torch.no_grad():
            return self.model(input_data)
    
    def evaluate(self, eval_data: Any) -> Dict[str, Any]:
        """Evaluate model"""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_data:
                states, actions, rewards = batch
                predictions = self.model(states)
                targets = rewards.unsqueeze(1)
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'eval_loss': avg_loss}
    
    def save_model(self, path: str) -> None:
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class SimpleRLPlugin(AlgorithmPlugin):
    """Simple RL algorithm plugin"""
    
    @property
    def plugin_info(self) -> PluginInfo:
        return PluginInfo(
            name="simple_rl",
            version="1.0.0",
            description="Simple RL Agent Plugin",
            author="AquaML Team",
            plugin_type=PluginType.ALGORITHM,
            dependencies=[],
            entry_point="AquaML.examples.quickstart.basic_usage"
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin"""
        self.config = config
        print(f"SimpleRL Plugin initialized with config: {config}")
    
    def cleanup(self) -> None:
        """Cleanup plugin"""
        print("SimpleRL Plugin cleaned up")
    
    def get_algorithm_class(self):
        """Get algorithm class"""
        return SimpleRLAgent
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'input_dim': 4,
            'output_dim': 2,
            'learning_rate': 0.001,
            'device': 'cpu'
        }


def create_dummy_data():
    """Create dummy training data"""
    batch_size = 32
    input_dim = 4
    
    # Generate random data
    states = torch.randn(batch_size, input_dim)
    actions = torch.randint(0, 2, (batch_size,))
    rewards = torch.randn(batch_size)
    
    return [(states, actions, rewards)]


def main():
    """Main example function"""
    print("🚀 AquaML - Basic Usage Example")
    
    # 1. Create coordinator
    coordinator = AquaMLCoordinator()
    
    # 2. Configure the system
    config = {
        'plugins': {
            'simple_rl': {
                'path': 'AquaML.examples.quickstart.basic_usage',
                'config': {
                    'input_dim': 4,
                    'output_dim': 2,
                    'learning_rate': 0.001
                }
            }
        },
        'training': {
            'epochs': 10,
            'device': 'cpu'
        }
    }
    
    # 3. Initialize coordinator
    coordinator.initialize(config)
    
    # 4. Register plugin manually (since we don't have plugin loading yet)
    plugin_manager = coordinator.get_plugin_manager()
    if plugin_manager:
        plugin_manager.register_plugin(SimpleRLPlugin, config['plugins']['simple_rl']['config'])
        print("✅ Registered SimpleRL plugin")
    
    # 5. Create and configure agent
    agent_config = config['plugins']['simple_rl']['config']
    agent_config.update(config['training'])
    
    agent = SimpleRLAgent(agent_config)
    coordinator.registry.register('agent', agent)
    print("✅ Created and registered agent")
    
    # 6. Create dummy data
    train_data = create_dummy_data()
    eval_data = create_dummy_data()
    print("✅ Created training data")
    
    # 7. Train the agent
    print("\n🏋️ Starting training...")
    training_results = agent.train(
        train_data=train_data,
        eval_data=eval_data,
        epochs=config['training']['epochs']
    )
    
    print(f"✅ Training completed!")
    print(f"📊 Final training metrics: {training_results['train_metrics'][-1]}")
    print(f"📊 Final evaluation metrics: {training_results['eval_metrics'][-1]}")
    
    # 8. Make predictions
    print("\n🔮 Making predictions...")
    test_input = torch.randn(1, 4)
    prediction = agent.predict(test_input)
    print(f"🎯 Prediction for input {test_input}: {prediction}")
    
    # 9. Show agent state
    print("\n📈 Agent state:")
    state = agent.get_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # 10. Show coordinator status
    print("\n🏗️ Coordinator status:")
    status = coordinator.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 11. Cleanup
    coordinator.shutdown()
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main() 