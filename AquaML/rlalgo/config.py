"""
Provides configuration for the RL algorithms.
"""

from dataclasses import dataclass


@dataclass
class LoadFlag:
    actor: bool = False
    critic: bool = False
    state_normalizer: bool = False
    reward_normalizer: bool = False

