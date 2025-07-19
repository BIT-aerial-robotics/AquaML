"""Off-policy reinforcement learning algorithms."""

from .sac import SAC, SACCfg
from .ddpg import DDPG, DDPGCfg

__all__ = ["SAC", "SACCfg", "DDPG", "DDPGCfg"] 