

MAIN_THREAD = 1
SUB_THREAD = 2

# RL type
STOCHASTIC = 1
DETERMINISTIC = 2

# RL Model' type
SHARE_ACTOR_CRITIC = 1
SEPARATE_ACTOR_CRITIC = 2

# Machine learning type
REINFORCE_LEANING = 1

# Memory share
NO_SHARE_MEMORY = 0
CREATE_SHARE_MEMORY = 1
LOAD_SHARE_MEMORY = 2

################### Aqua Reinforcement Learning ###################
from AquaML.buffer.RLBuffer import TrajectoryFilterRegister, RLBufferPluginRegister
traj_filter_register = TrajectoryFilterRegister()
rl_buffer_plugin_register = RLBufferPluginRegister()