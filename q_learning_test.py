#!/usr/bin/env python3
"""
Test script for Q-Learning implementation in AquaML
This test verifies that Q-Learning can be imported and instantiated without errors.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_q_learning_import():
    """Test that Q-Learning can be imported"""
    print("Testing Q-Learning import...")
    
    try:
        from AquaML.learning.reinforcement.off_policy.q_learning import QLearning, QLearningCfg
        print("✓ Successfully imported QLearning and QLearningCfg")
        return True
    except Exception as e:
        print(f"❌ Failed to import Q-Learning: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_q_learning_config():
    """Test Q-Learning configuration creation"""
    print("Testing Q-Learning configuration...")
    
    try:
        from AquaML.learning.reinforcement.off_policy.q_learning import QLearningCfg
        
        # Create default config
        cfg = QLearningCfg()
        print(f"✓ Created QLearningCfg with default values")
        print(f"  - discount_factor: {cfg.discount_factor}")
        print(f"  - learning_rate: {cfg.learning_rate}")
        print(f"  - random_timesteps: {cfg.random_timesteps}")
        print(f"  - learning_starts: {cfg.learning_starts}")
        
        # Test custom config
        cfg.learning_rate = 0.1
        cfg.discount_factor = 0.95
        print(f"✓ Modified QLearningCfg values successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test QLearningCfg: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_q_learning_vs_sac():
    """Compare Q-Learning and SAC configs to ensure proper framework integration"""
    print("Comparing Q-Learning and SAC configurations...")
    
    try:
        from AquaML.learning.reinforcement.off_policy.q_learning import QLearningCfg
        from AquaML.learning.reinforcement.off_policy.sac import SACCfg
        
        q_cfg = QLearningCfg()
        sac_cfg = SACCfg()
        
        # Check common attributes
        common_attrs = ['discount_factor', 'device', 'random_timesteps', 'learning_starts']
        
        print("Common attributes comparison:")
        for attr in common_attrs:
            if hasattr(q_cfg, attr) and hasattr(sac_cfg, attr):
                q_val = getattr(q_cfg, attr)
                sac_val = getattr(sac_cfg, attr)
                print(f"  {attr}: Q-Learning={q_val}, SAC={sac_val}")
        
        # Check Q-Learning-specific attributes
        q_specific = ['learning_rate', 'batch_size', 'memory_size']
        
        print("Q-Learning-specific attributes:")
        for attr in q_specific:
            if hasattr(q_cfg, attr):
                val = getattr(q_cfg, attr)
                print(f"  {attr}: {val}")
        
        print("✓ Configuration comparison completed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to compare configurations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_q_learning_model_creation():
    """Test creating Q-Learning agent with a simple model"""
    print("Testing Q-Learning agent creation...")
    
    try:
        import torch
        import torch.nn as nn
        from AquaML.learning.model import Model
        from AquaML.learning.model.model_cfg import ModelCfg
        from AquaML.learning.reinforcement.off_policy.q_learning import QLearning, QLearningCfg
        
        # Create a simple Q-function model
        class SimpleQFunction(Model):
            def __init__(self, model_cfg: ModelCfg):
                super().__init__(model_cfg)
                self.net = nn.Sequential(
                    nn.Linear(4, 32),  # 4 state dimensions
                    nn.ReLU(),
                    nn.Linear(32, 2)   # 2 actions (discrete)
                )
            
            def compute(self, data_dict):
                states = data_dict["state"] if "state" in data_dict else list(data_dict.values())[0]
                if states.dim() == 1:
                    states = states.unsqueeze(0)
                q_values = self.net(states)
                return {"q_values": q_values}
            
            def act(self, data_dict):
                outputs = self.compute(data_dict)
                q_values = outputs["q_values"]
                actions = torch.argmax(q_values, dim=-1, keepdim=True)
                outputs["actions"] = actions
                return outputs
        
        # Create model configuration
        model_cfg = ModelCfg(
            device="cpu",
            inputs_name=["state"],
            concat_dict=False
        )
        
        # Create model
        q_function = SimpleQFunction(model_cfg)
        print("✓ Created simple Q-function model")
        
        # Create Q-Learning configuration
        q_cfg = QLearningCfg()
        q_cfg.device = "cpu"
        q_cfg.learning_rate = 0.1
        q_cfg.discount_factor = 0.99
        print("✓ Created Q-Learning configuration")
        
        # Create Q-Learning agent
        models = {"policy": q_function}
        agent = QLearning(models, q_cfg)
        print("✓ Created Q-Learning agent successfully")
        
        # Test basic functionality
        dummy_states = {"state": torch.randn(1, 4)}
        outputs = agent.act(dummy_states, 0, 1000)
        print(f"✓ Generated action: {outputs['actions'].item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Q-Learning agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Q-Learning Implementation Test ===\n")
    
    tests = [
        test_q_learning_import,
        test_q_learning_config,
        test_q_learning_vs_sac,
        test_q_learning_model_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}\n")
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Q-Learning implementation looks good.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)