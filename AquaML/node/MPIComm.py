from mpi4py import MPI
from AquaML.data.DataPool import DataPool
from AquaML.data.ParamData import ParamData
import numpy as np


class MPIComm:
    def __init__(self, comm: MPI.COMM_WORLD):
        """
        MPI communicator. The communication tool for each node.
        Current just support bcast and gather
        We assume all the root node id is 0.

        :param comm: MPI comm.
        """
        self.comm = comm
        self.rank = comm.rank  # get rank from current group

    def gather_one(self, data_pool: DataPool):
        """
        gather one data pool.
        :param data_pool: In main thread, shape is [size, total_steps, shape]. Sub thread is [total_steps, shape].
        :return: [size, total_steps, shape]. Usually [1:,:,:]
        """

        if self.rank == 0:
            recv = np.empty(shape=data_pool.shapes, dtype=np.float32)
            send = np.zeros(shape=data_pool.shapes[1:], dtype=np.float32)
        else:
            recv = None
            send = data_pool.data

        self.comm.Gather(send, recv, root=0)

        return recv

    def bcast(self, data: ParamData):

        if self.rank == 0:
            buffer = data.data
        else:
            buffer = np.empty(data.shape, dtype=np.float32)

        self.comm.Bcast(buffer)

        return buffer
