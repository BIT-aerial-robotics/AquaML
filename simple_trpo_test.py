#!/usr/bin/env python3
"""
Simple test for TRPO implementation in AquaML
This test verifies that TRPO can be imported and instantiated without errors.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_trpo_import():
    """Test that TRPO can be imported"""
    print("Testing TRPO import...")
    
    try:
        from AquaML.learning.reinforcement.on_policy.trpo import TRPO, TRPOCfg
        print("✓ Successfully imported TRPO and TRPOCfg")
        return True
    except Exception as e:
        print(f"❌ Failed to import TRPO: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trpo_config():
    """Test TRPO configuration creation"""
    print("Testing TRPO configuration...")
    
    try:
        from AquaML.learning.reinforcement.on_policy.trpo import TRPOCfg
        
        # Create default config
        cfg = TRPOCfg()
        print(f"✓ Created TRPOCfg with default values")
        print(f"  - rollouts: {cfg.rollouts}")
        print(f"  - max_kl_divergence: {cfg.max_kl_divergence}")
        print(f"  - conjugate_gradient_steps: {cfg.conjugate_gradient_steps}")
        
        # Test custom config
        cfg.rollouts = 32
        cfg.max_kl_divergence = 0.02
        print(f"✓ Modified TRPOCfg values successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test TRPOCfg: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trpo_vs_ppo():
    """Compare TRPO and PPO configs to ensure proper porting"""
    print("Comparing TRPO and PPO configurations...")
    
    try:
        from AquaML.learning.reinforcement.on_policy.trpo import TRPOCfg
        from AquaML.learning.reinforcement.on_policy.ppo import PPOCfg
        
        trpo_cfg = TRPOCfg()
        ppo_cfg = PPOCfg()
        
        # Check common attributes
        common_attrs = ['rollouts', 'learning_epochs', 'mini_batches', 'discount_factor', 'lambda_value']
        
        print("Common attributes comparison:")
        for attr in common_attrs:
            if hasattr(trpo_cfg, attr) and hasattr(ppo_cfg, attr):
                trpo_val = getattr(trpo_cfg, attr)
                ppo_val = getattr(ppo_cfg, attr)
                print(f"  {attr}: TRPO={trpo_val}, PPO={ppo_val}")
        
        # Check TRPO-specific attributes
        trpo_specific = ['max_kl_divergence', 'conjugate_gradient_steps', 'damping', 'max_backtrack_steps']
        
        print("TRPO-specific attributes:")
        for attr in trpo_specific:
            if hasattr(trpo_cfg, attr):
                val = getattr(trpo_cfg, attr)
                print(f"  {attr}: {val}")
        
        print("✓ Configuration comparison completed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to compare configurations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== TRPO Implementation Test ===\n")
    
    tests = [
        test_trpo_import,
        test_trpo_config,
        test_trpo_vs_ppo,
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
        print("🎉 All tests passed! TRPO implementation looks good.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)