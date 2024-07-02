'''
MPI进程间通信

'''

from AquaML.communicator.CommunicatorBase import CommunicatorBase
from mpi4py import MPI

class MPICommunicator(CommunicatorBase):
    def __init__(self,
                 comunicator_path: str = None,
                 machine_id=0,
                 compute_engine: str = 'tensorflow',
                 wait_time_out = 1,  # 等待超时时间
                 check_time_interval = 0.001,
                 detailed_log = False,
                 ):
        """
        用于初始化CommunicatorBase。
        
        args:
            process_id (int): 线程ID，用于区分不同的进程。
            comunicator_path (str): 通讯器的工作路径。
            machine_id(int): 机器的ID。用于区分不同的机器。
            compute_engine(str): 计算引擎。默认为tensorflow。当前支持tensorflow、pytorch以及JAX。
        """

        self.detailed_log = detailed_log
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        super().__init__(comunicator_path,
                         machine_id,
                         compute_engine,
                         wait_time_out,
                         check_time_interval,
                         )

    
    def get_total_process_num(self):
        """
        获取总的进程数。
        """
        return self.size
        
    def get_process_id(self):
        """
        设置进程ID。
        """
        return self.rank
    
    def barrier(self):
        """
        进程同步
        """
        self.comm.Barrier()

    def logger_warning(self, sentence: str):
        """
        用于记录警告。

        默认的警告语句格式为：Machine {} process {}: sentence

        args:
            sentence (str): 警告语句。
        """
        if self.detailed_log:
            self.logger.warning('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_error(self, sentence: str):
        """
        用于记录错误。

        默认的警告语句格式为：Machine {} process {}: sentence

        args:
            sentence (str): 错误语句。
        """

        self.logger.error('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_info(self, sentence: str):
        """
        用于记录信息。

        默认的警告语句格式为：Machine {} process {}: sentence

        args:
            sentence (str): 信息语句。
        """

        if self.detailed_log:
            self.logger.info('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))