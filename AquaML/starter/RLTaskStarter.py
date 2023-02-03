from AquaML.BaseClass import BaseStarter
from AquaML.DataType import DataInfo, RLIOInfo
from mpi4py import MPI
import time


# TODO: 检查任务级别
class RLTaskStarter(BaseStarter):
    def __init__(self, env,
                 obs_info: DataInfo,
                 model_class_dict: dict,
                 algo,
                 algo_hyperparameter,
                 mpi_comm=None,
                 computer_type: str = 'PC',
                 name=None,
                 ):
        """
        Start a reinforcement learning task.

        Args:
            env : environment. (must be a inherited class of AquaML.BaseClass.RLBaseEnv)
            obs_info (DataInfo): full observation of environment.
            model_class_dict (dict): model class dict. {'actor':actor_class, 'critic':critic_class}. 
                                    They must be inherited class of AquaML.BaseClass.RLBaseModel.
            algo_hyperparameter : This is a structure. See AquaML.rl.algo.Parameters.
            mpi_comm (None or MPI.COMM_WORLD): mpi communicator. If True, use mpi to run the task. Default is False.
            computer_type (str): computer type. Default is 'PC'. It decides the way 
                                 to communicate with each thread.
            name (str, optional): name of the task. Defaults to None. When None, use the algorithm name.
        """
        # TODO: check logics
        # get actor info from model_class_dict
        # just create, do not build
        actor = model_class_dict['actor']()
        actor_out_info = actor.output_info  # dict
        del actor  # delete actor

        # parallel information
        # if using mpi
        if mpi_comm is not None:
            # get numbers of threads
            total_threads = MPI.COMM_WORLD.Get_size()

            # check  buffer size can be divided by mpi size
            buffer_size = (algo_hyperparameter.buffer_size / total_threads) * total_threads
            algo_hyperparameter.buffer_size = buffer_size
            thread_id = MPI.COMM_WORLD.Get_rank()

            # set thread level
            if thread_id == 0:
                level = 0
            else:
                level = 1
        else:
            total_threads = 1
            thread_id = -1
            level = 0

        self.total_threads = total_threads
        self.thread_id = thread_id
        self.level = level
        self.computer_type = computer_type

        # create rl_io_info
        rl_io_info = RLIOInfo(obs_info=obs_info.shape_dict,
                              obs_type_info=obs_info.type_dict,
                              actor_out_info=actor_out_info,
                              reward_info=env.reward_info,
                              buffer_size=algo_hyperparameter.buffer_size
                              )

        # create dict for instancing algorithm
        parallel_args = {
            'total_threads': self.total_threads,
            'thread_id': self.thread_id,
            'level': self.level,
            'computer_type': self.computer_type,
        }

        if name is not None:
            parallel_args['name'] = name

        algo_args = {
            'env': env,
            'rl_io_info': rl_io_info,
            'parameters': algo_hyperparameter,
        }

        model_args = model_class_dict

        algo_args = {**algo_args, **model_args, **parallel_args}

        # create algorithm
        self.algo = algo(**algo_args)

        # initial algorithm
        self.algo.init()

        # store key objects
        self.max_epochs = algo_hyperparameter.n_epochs
        self.mpi_comm = mpi_comm

        # config run function

        if mpi_comm is None:
            self.run = self._run_
        else:
            self.run = self._run_mpi_

    # single thread
    def _run_(self):
        for i in range(self.max_epochs):
            self.algo.worker.roll()
            self.algo.optimize()

        self.algo.close()

    def _run_mpi_(self):
        for i in range(self.max_epochs):
            if self.thread_id == 0:
                self.algo.sync()
            else:
                pass
            self.mpi_comm.Barrier()

            if self.thread_id > 0:
                self.algo.sync()
            else:
                pass
            self.mpi_comm.Barrier()

            self.algo.worker.roll()
            self.mpi_comm.Barrier()

            if self.thread_id == 0:
                self.algo.optimize()
            else:
                pass
            self.mpi_comm.Barrier()
