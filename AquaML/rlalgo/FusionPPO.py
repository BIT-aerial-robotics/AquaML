from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.rlalgo.Parameters import FusionPPO_parameter
from AquaML.DataType import RLIOInfo
import tensorflow as tf

class FusionPPO(BaseRLAlgo):
    def __init__(self,
                 env,
                 rl_io_info: RLIOInfo,
                 parameters: FusionPPO_parameter,
                 actor,
                 critic,
                 computer_type: str = 'PC',
                 name: str = 'PPO',
                 level: int = 0,
                 thread_id: int = -1,
                 total_threads: int = 1, ):
        super().__init__(
            env=env,
            rl_io_info=rl_io_info,
            name=name,
            update_interval=parameters.update_interval,
            computer_type=computer_type,
            level=level,
            thread_ID=thread_id,
            total_threads=total_threads,
        )