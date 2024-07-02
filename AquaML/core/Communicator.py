'''

用于管理进程的工具，面向下一代框架设计。
'''

from AquaML import logger
from mpi4py import MPI

class Communicator:
    """
    用于控制管理不同进程
    """
    def __init__(self):
        
        # 在该框架中，默认使用MPI进行通信
        # 无论在单进程还是多进程，都会使用mpi4py
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        
    
    @property
    def rank(self)->int:
        """
        
        返回当前进程的rank

        Returns:ßßß
            int: 当前进程的rank
        """
        return self._rank
    
    @property
    def size(self)->int:
        """
        返回总的进程数

        Returns:
            int: 总的进程数
        """
        
        return self._size
    
    ######################### 同步操作 #########################
    def Barrier(self):
        """
        同步操作
        """
        self._comm.Barrier()
        
    
        