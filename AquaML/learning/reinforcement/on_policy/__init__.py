"""On-policy reinforcement learning algorithms."""

from .ppo import PPO, PPOCfg
from .trpo import TRPO, TRPOCfg

__all__ = [
    'PPO',
    'PPOCfg',
    'TRPO',
    'TRPOCfg',
]
