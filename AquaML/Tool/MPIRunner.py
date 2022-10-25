from AquaML import RLWorker
from AquaML.data.DataCollector import DataCollector
import os
import time


def mkdir(path):
    current = os.getcwd()
    path = current + '/' + path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)


# TODO: Gradually improve plan.
class MPIRunner:
    def __init__(self, policy, algo, algo_args, work_space: str, data_info, env, comm):
        """
        Parallel running. Can be used by all types of algo.
        :param policy: Policy contains neural networks. The format depend on your task, but it has the basic class.
        :param algo: Algo you will use.
        :param algo_args: Args of algo.
        :param work_space: name of your task.
        :param env: environment
        """

        self.policy = policy

        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        if self.rank == 0:
            self.data_collector = DataCollector(data_dic=data_info.data_dic, total_length=data_info.total_length,
                                                name_prefix=work_space, share_memory=True)
        else:
            time.sleep(6)
            self.data_collector = DataCollector(data_dic=data_info.data_dic, total_length=data_info.total_length,
                                                name_prefix=work_space, share_memory=False)
        # self.algo = algo(algo_args=algo_args, data_collector, policy)
