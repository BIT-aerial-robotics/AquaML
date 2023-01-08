import numpy as np
from multiprocessing import shared_memory

class DataUnit:
    """
    The smallest data storage unit. It can be used in HPC (MPI) and 
    shared memory system.
    """

    def __init__(self, name:str):
        """

        Parameters
        ----------
        name : str
            The name of the data unit
        """