from AquaML.communicator.CommunicatorBase import CommunicatorBase
import os

class DebugCommunicator(CommunicatorBase):
    def __init__(
        self,
        communicator_path: str=None,
        machine_id: int=0,
        compute_engine: str='tensorflow',
        force_run: bool=True
    ):
        """
        
        Debug communicator for debugging purpose.

        Args:
            communicator_path (str, optional): 通讯器的工作路径.当为None时，使用默认路径。 Defaults to None.
            machine_id (int, optional): 机器的ID。用于区分不同的机器. 如果不使用集群，可以忽略此参数。 Defaults to 0.
            compute_engine (str, optional): 计算引擎。默认为tensorflow。当前支持tensorflow、pytorch以及JAX. Defaults to 'tensorflow'.
            force_run (bool, optional): 强制运行模式，即使遇到错误也会忽略错误继续运行。默认为True。此模式仅用于调试目的。请勿在生产环境中使用。 
        """
        super().__init__(
            comunicator_path=communicator_path,
            machine_id=machine_id,
            compute_engine=compute_engine,
            debug_mode=True
        )
        self._force_run = force_run
        self.logger_warning('Force run mode is on. This mode is only for debugging purpose. Please do not use it in production environment.')
        
        self.wait_time_out = 0.0001 # 用于调试的时候，等待时间。降低等待时间，强制跳过等待。用于单进程测试。
        self.logger_warning('Wait time out is set to 0.0001. This mode is only for debugging purpose. Please do not use it in production environment.')
        self.check_time_interval = 0.0001 # 用于调试的时候，检查时间间隔。降低检查时间间隔，强制跳过等待。用于单进程测试。
        self.logger_warning('Check time interval is set to 0.0001. This mode is only for debugging purpose. Please do not use it in production environment.')

    def get_process_id(self):
        return 0

    def get_total_process_num(self):
        return 1

    def barrier(self):
        pass

