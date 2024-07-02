'''
该模块能够让离线强化学习在线运行。
该模块主要包含模型更新进程和环境采样进程，
采样进程和更新进程完全异步执行，采样进程不断采样数据，更新进程不断更新模型。
'''

from AquaML.framework.FrameWorkBase import FrameWorkBase
from AquaML import logger, communicator, settings


class Offline2Online(FrameWorkBase):
    """
    Offline2online用于将离线强化学习转换为在线强化学习。
    """
    
    def __init__(self,
                 
                 ):
        pass
    
    def run(self):
        """
        运行。
        """
        pass