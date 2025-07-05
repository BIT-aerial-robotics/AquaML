'''
进程模拟器，用于虚拟化进程，方便对算法进行测试
'''

from AquaML.communicator import CommunicatorBase

class ProcessSimulator(CommunicatorBase):
    def __init__(self,comunicator_path:str ,machine_id=0, compute_engine:str='tensorflow'):
        """
        用于初始化CommunicatorBase。
        
        args:
            process_id (int): 线程ID，用于区分不同的进程。
            comunicator_path (str): 通讯器的工作路径。
            machine_id(int): 机器的ID。用于区分不同的机器。
            compute_engine(str): 计算引擎。默认为tensorflow。当前支持tensorflow、pytorch以及JAX。
        """
        
        super().__init__(comunicator_path, machine_id, compute_engine)
    
    
    def get_total_process(self):
        """
        获取总的进程数。
        """
        return 1
        
    def get_process_id(self):
        """
        设置进程ID。
        """
        return 0
    
    def barrier(self):
        """
        进程同步
        """
        pass