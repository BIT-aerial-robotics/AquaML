'''

所有深度学习框架的离线强化学习算法的基类.
'''

import abc
import time

from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
import keras
import numpy as np
from AquaML.algo_base.AlgoBase import AlgoBase
from AquaML.core.old.FileSystem import FileSystemBase


class OfflineRLBase(AlgoBase,abc.ABC):
    
    def __init__(self, 
                 name:str,
                 # buffer,
                 hyper_params,
                 communicator:CommunicatorBase,
                 data_module:DataModule,
                 file_system:FileSystemBase
                 ):
        """
        Algo算法基类，用于定义Algo算法的基本功能。

        Args:
            name(str): 算法的名称。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystem): 文件系统。用于文件的存储和读取。由系统自动传入。
        """
        super().__init__(
            name=name,
            # buffer=buffer,
            hyper_params=hyper_params,
            communicator=communicator,
            data_module=data_module,
            file_system=file_system
        )

    def init(self, buffer):
        """
        初始化算法。
        """
        self._buffer = buffer

        
        
    def run(self):
        """
        运行算法。
        """
        self._communicator.logger_info('OfflineRLBase: Run OfflineRLBase')
        
        self._data_module.wait_program_start()

        while not self._data_module.get_program_end:
            # time.sleep(0.01)
            data_num = self._buffer.capacity_count if self._buffer.capacity_count < self._buffer._capacity else self._buffer._capacity

            if data_num >= self._hyper_params.update_start:
                loss_tracker = self.optimize(self._buffer)

                self._optimize_times += 1

                # 隔开显示
                print('###################{} optimize times: {}###################'.format(self._name, self._optimize_times))

                # 获取loss信息
                loss_dict = loss_tracker.get_data()
                # self._communicator.logger_success(loss_dict)

                for key, value in loss_dict.items():
                    print('{}: {}'.format(key, value))
                    self._communicator.logger_info('OfflineRLBase: {}: {}'.format(key, value))

                # self.save_cache_models(self.model_dict)
                if self._optimize_times % self._hyper_params.model_save_interval == 0:
                    self.save_checkpoint(self.model_dict)

