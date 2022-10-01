from mpi4py import MPI
import numpy as np
import sys

main_comm = MPI.COMM_WORLD
comm = MPI.Comm.Get_parent()
comm_rank = comm.Get_rank()

data = comm.bcast(None, root=0)

print("{},{}".format(comm_rank, data))
