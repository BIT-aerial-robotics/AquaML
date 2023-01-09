import numpy as np
from multiprocessing import shared_memory
import warnings

class DataUnit:
    """
    The smallest data storage unit. It can be used in HPC (MPI) and 
    shared memory system.
    """

    def __init__(self, name:str, shape=None, dtype=np.float32, computer_type='PC'):
        """Create data unit. If shape is not none, this data unit is used in main thread.
        The unit is created depend on your computer type. If you use in high performance
        computer(HPC), shared memmory isn't used.

        Args:
            name (str): _description_
            shape (tuple, None): shape of data unit.If shape is none, the unit is in sub thread. 
            (buffersize, dims)
             Defaults to None.
            dtype (nd.type): type of this data unit. Defaults to np.float32.
            computer_type(str):When computer type is 'PC', mutlti thread is based on shared memory.Defaults to 'PC'.
        """

        self.name = name

        self.shape = shape
        self.dtype = dtype

        self.computer_type = computer_type

        if shape is not None:
            self.level = 0
            self.buffer = np.zeros(shape=shape,dtype=self.dtype)
            self.__nbytes = self.buffer.nbytes
        else:
            self.level = 1
            self.buffer = None
            self.__nbytes == None

    
    def create_shared_memory(self):
        """Create shared-memory.
        """
        if self.computer_type == 'HPC':
            warnings.warn("HPC can't support shared memory!")

        if self.level == 0:
            self.shm_buffer = shared_memory.SharedMemory(create=True, size=self.__nbytes, name=self.name)
            self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_buffer)
        else:
            raise Exception("Current thread is sub thread!")
    
    def read_shared_memory(self,shape:tuple):
        if self.computer_type == 'HPC':
            warnings.warn("HPC can't support shared memory!")
        
        if self.level == 1:
            self.__nbytes = self.compute_nbytes(shape)
            self.shape = shape
            self.shm_buffer = shared_memory.SharedMemory(name=self.name, size=self.__nbytes)
            self.buffer = np.ndarray(self.shape,dtype=self.dtype,buffer=self.shm_buffer)
        else:
            raise Exception("Current thread is main thread!")
        
    
    def compute_nbytes(self, shape:tuple)->int:
        """Compute numpy array nbytes.

        Args:
            shape (tuple): _description_

        Returns:
            _type_: int
        """

        a = np.arange(1,  dtype=self.dtype)

        single_nbytes = a.nbytes

        total_szie = 1

        for size in shape:
            total_szie = total_szie*size
        

        total_szie = total_szie*single_nbytes

        return total_szie
    

