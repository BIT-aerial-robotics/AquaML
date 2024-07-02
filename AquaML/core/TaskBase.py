'''

Task启动模板，用于启动Task。

所有需要启动得任务都要从这里集成。
'''

import abc
from AquaML.communicator.CommunicatorBase import CommunicatorBase

class TaskBase:
    def __init__(self, communicator:CommunicatorBase):
        """
        所有任务的基类，用于定义任务的基本功能。
        
        Args:
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。
        """
        self._communicator = communicator

    def start(self):
        pass

    def stop(self):
        pass