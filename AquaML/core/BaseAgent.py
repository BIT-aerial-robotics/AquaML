from abc import ABC, abstractmethod

class BaseAgent(ABC):
    
    @abstractmethod
    def summary_io_info(self):
        """
        Agent需要提供自己的输入输出信息。不同的算法汇总方式不一样
        ，因此需要agent自己提供。
        """