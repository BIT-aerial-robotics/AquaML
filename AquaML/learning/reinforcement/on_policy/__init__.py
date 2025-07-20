"""On-policy reinforcement learning algorithms."""

from .ppo import PPO, PPOCfg
from .trpo import TRPO, TRPOCfg
from .a2c import A2C, A2CCfg
from .amp import AMP, AMPCfg
from .rpo import RPO, RPOCfg

__all__ = [
    'PPO',
    'PPOCfg',
    'TRPO',
    'TRPOCfg',
    'A2C',
    'A2CCfg',
    'AMP',
    'AMPCfg',
    'RPO',
    'RPOCfg',
]
