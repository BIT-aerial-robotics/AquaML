import numpy as np
from AquaML.buffer.BaseBuffer import BaseBuffer


class OnPolicyDefaultReplayBuffer(BaseBuffer):
    """
    RL 第一个经验池，只是单纯的存储经验，对LSTM输入进行部分处理后续将增加更强的经验池。

    该经验池可以在RL算法中可以创建多个，默认中我们将为根据数据集合分别为actor和critic创建两个经验池。

    """

    def __init__(self, capacity: int, data_names: list):
        super().__init__(
            capacity=capacity,
            data_names=data_names
            )
        
    def _process_data(self, data: dict):
        """
        Processes the data in the buffer.

        This method is called by the `append` method to process the data in the buffer before it is returned. This method
        can be overridden in subclasses to perform additional processing on the data.

        Return like this:
            {
                'state': np.array(...),
                'action': np.array(...),
                'reward': np.array(...),
                'next_state': np.array(...),
                'done': np.array(...)
            }
        """
        
