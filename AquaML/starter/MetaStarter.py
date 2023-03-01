from mpi4py import MPI
import time
import atexit


class MetaStarter:
    def __init__(self,
                 meta_algo,
                 meta_parameter,
                 inner_algo,
                 inner_parameter,
                 name: str = None,
                 mpi_comm=None,
                 ):

        """
        In meta learning, the name of inner loop will be allocated by the meta algorithm.

        In our meta learning, meta algorithm can directly visit the data
        pool and parameters of inner algorithm.

        As for the neural network, the meta algorithm get the weights via load_weights().

        """
        pass
