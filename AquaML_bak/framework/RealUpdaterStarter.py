'''

此模块用将updater模块和buffer模块进行绑定，然后updater和buffer两个进程里面运行，buffer自动从内存里面获取数据
整理数据，然后传递给updater，updater进行训练，然后将训练好的模型参数传递给buffer，buffer将模型参数保存到硬盘上。
'''

from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.FileSystem import DefaultFileSystem
from AquaML.core.old.DataModule import DataModule
from threading import Thread
from AquaML.buffer.RealCollectBuffer import RealCollectBuffer
from AquaML.buffer.MixtureBuffer import MixtureBuffer


class RealUpdaterStarter:
    
    def __init__(self,
                 policy_updater_name:str,
                 policy_updater,
                 policy_updater_param:dict,
                 capacity:int,
                 data_names_in_buffer:list,
                 data_module:DataModule,
                 communicator:CommunicatorBase,
                 file_system:DefaultFileSystem,
                 offline_dataset_path: str = None,
                 ):
        """
        
        此模块用将updater模块和buffer模块进行绑定，然后updater和buffer两个进程里面运行，buffer自动从内存里面获取数据
        整理数据，然后传递给updater，updater进行训练，然后将训练好的模型参数传递给buffer，buffer将模型参数保存到硬盘上。

        Args:
            policy_updater_name (str): 策略更新的名称。
            policy_updater: 更新算法。
            policy_updater_param (dict): 更新算法的参数。
            capacity (int): buffer的容量。
            data_names_in_buffer (list): buffer中的数据名称。
            data_module (DataModule): 数据模块。
            communicator (CommunicatorBase): 通信模块。
            file_system (DefaultFileSystem): 文件系统。
        """
        communicator.logger_info('RealUpdaterStarter: __init__')
        
        # 三大件
        self._data_module = data_module
        self._communicator = communicator
        self._file_system = file_system
        
        # 混合参数                                          
        self._policy_updater_name = policy_updater_name
        self._capacity = capacity
        self._data_names_in_buffer = data_names_in_buffer
        self._static_data_path = offline_dataset_path

        # 初始化策略更新器
        self._communicator.logger_info('RealUpdaterStarter: Create policy_updater')
        self.policy_updater = policy_updater(
            **policy_updater_param,
            file_system=self._file_system,
            data_module=self._data_module,
            communicator=self._communicator
        )
        
        # 初始化buffer
        if self._static_data_path is None:
            self._communicator.logger_info('RealUpdaterStarter: Create buffer')
            self.buffer = RealCollectBuffer(
                capacity=self._capacity,
                data_names=self._data_names_in_buffer,
                data_module=self._data_module,
                communicator=self._communicator,
                file_system=self._file_system
            )
        else:
            self._communicator.logger_info('MixtureBuffer: Create buffer')
            self.buffer = MixtureBuffer(
                capacity=self._capacity,
                data_names=self._data_names_in_buffer,
                data_module=self._data_module,
                communicator=self._communicator,
                static_data_path=self._static_data_path
            )

        self.policy_updater.init(self.buffer)


        # def run_buffer():
        #     pass
        
    def  run(self):
        """
        运行updater和buffer。
        """
        self._communicator.logger_info('RealUpdaterStarter: run')
        
        self.buffer_thread = Thread(target=self.buffer.run)
        
        self.updater_thread = Thread(target=self.policy_updater.run)
        
        self.buffer_thread.start()
        self.updater_thread.start()

        self._data_module.get_program_running_state().set_data([[True, False]])

        self.buffer_thread.join()
        self.updater_thread.join()

    def __del__(self):
        self._data_module.get_program_running_state().set_data([[True, True]])
        
        