from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
print(rank)
new_comm = MPI.COMM_SELF.Spawn(sys.executable, args=['slave.py'], maxprocs=3)

data = [1, 2, 3]

new_comm.bcast(data, root=0)
