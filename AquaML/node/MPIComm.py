from mpi4py import MPI


class MPIComm:
    def __init__(self, comm: MPI.COMM_WORLD):
        """"
        MPI communicator. The communication tool for each node.

        :param comm: MPI comm.
        """
        self.comm = comm
        self.rank = comm.rank


    
