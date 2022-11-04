# # import tensorflow as tf
#
#
# class Test:
#     def __init__(self, tf_handle):
#         self.tf = tf_handle
#
#         self.a = self.tf.ones(shape=(12, 1), dtype=self.tf.float32)
#
#         @tf_handle.function
#         def multi(b):
#             return b * self.a
#
#         self.multi = multi
#
#
# if __name__ == "__main__":
#     import multiprocessing as mp
#
#
#     def running(id):
#         import os
#         import tensorflow as tf
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
#         test = Test(tf)
#
#         print(test.multi(id))
#
#
#     processes = [mp.Process(target=running, args=(i,)) for i in range(10)]
#
#     for proces in processes:
#         proces.start()
#
#     proces.join()


# class A:
#     def __init__(self, a, b, c):
#         self.a = a
#         self.c = c
#         self.b = b
#
#     def test(self, b):
#         print(b.b)
#
#
# args = {'a': 1, 'b': 2, 'c': 3}
#
# a = A(**args)
#
# print(type(A))
# print(type(a))
# print(a.c)

# from mpi4py import MPI
#
# global_comm = MPI.COMM_WORLD
#
# global_rank = global_comm.Get_rank()
#
# global_size = global_comm.Get_size()
#
# group = global_comm.Get_group()
#
# group = group.Incl([1, 0, 2])
#
# sub_comm = global_comm.Create(group)
#
# if sub_comm != MPI.COMM_NULL:
#     print("{},{}".format(global_rank, sub_comm.Get_rank()))
#
# dic = {'a':1,'b':2}
#
# name = list(dic)
#
# # print('a' in dic)
#
# a = [i for i in range(10)]
#
# print(a)
import numpy as np

a = np.array([1, 2], dtype=np.int8)

print(a.nbytes)
